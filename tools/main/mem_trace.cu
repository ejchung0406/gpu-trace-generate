/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <pthread.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <set>
#include <condition_variable>
#include <functional>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdlib>
#include <zlib.h>
#include <sys/stat.h>
#include <fstream>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"
#include "mem_trace.h"

#define HEX(x)                                                            \
    "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x \
         << std::dec

#define CHANNEL_SIZE (1l << 20)

struct CTXstate {
    /* context id */
    int id;

    /* Channel used to communicate from GPU to CPU receiving thread */
    ChannelDev* channel_dev;
    ChannelHost channel_host;
};

/* lock */
pthread_mutex_t mutex;
pthread_mutex_t file_mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* grid launch id, incremented at every launch */
uint64_t grid_launch_id = 0;

/* # of workers for file i/o? */
const size_t num_threads = 128;

/* Trace file path */
// std::string trace_path = "/fast_data/echung67/trace/nvbit/temp/";
std::string trace_path = "./";

/* To distinguish different Kernels */
class UniqueKernelStore {
public:
    int add(const std::string& str) {
        int new_id = kernels.size();
        kernels.push_back(str);

        std::priority_queue<int, std::vector<int>, std::greater<int>> warp_id;
        std::set<int> warp_id_s;
        std::unordered_map<int, int> instr_count;
        warp_ids.push_back(warp_id);
        warp_ids_s.push_back(warp_id_s);
        instr_counts.push_back(instr_count);
        return new_id;
    }
    const std::string& get_string(int id) const {
        return kernels[id];
    }
    void create_trace_info() {
        for (int i = 0; i < static_cast<int>(kernels.size()); i++){
            // Print the elements in the heap in order
            std::string str = get_string(i);
            std::ofstream file_trace(trace_path + str + "_" + std::to_string(i) + "/" + "trace.txt", std::ios_base::app);
            std::ofstream file_info_trace(trace_path + str + "_" + std::to_string(i) + "/" + "trace_info.txt", std::ios_base::app);
            file_trace << warp_ids[i].size() << std::endl; // Total number of warps
            while (!warp_ids[i].empty()) {
                file_trace << warp_ids[i].top() << " " << "0" << std::endl;
                auto it = instr_counts[i].find(warp_ids[i].top());
                if (it == instr_counts[i].end()) {
                    std::cout << "Element with key=" << it->first << " not found." << std::endl;
                }
                file_info_trace << warp_ids[i].top() << " " << it->second << std::endl; // number of instructions in one trace*.raw file
                warp_ids[i].pop();
            }
            file_trace.close();
            file_info_trace.close();
        }
    }

    /* Warp ids */
    std::vector<std::priority_queue<int, std::vector<int>, std::greater<int>>> warp_ids;
    std::vector<std::set<int>> warp_ids_s;

    /* counting the number of instructions per one trace*.raw */
    std::vector<std::unordered_map<int, int>> instr_counts;
    std::vector<std::string> kernels;
};

class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; i++) {
            threads.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) {
                            return;
                        }
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();
        for (std::thread& thread : threads) {
            thread.join();
        }
    }

    template<class F, class... Args>
    void enqueue(F&& f, Args&&... args) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            tasks.emplace([=] { f(args...); });
        }
        cv.notify_one();
    }

private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex mutex;
    std::condition_variable cv;
    bool stop;
};

bool is_fp(std::string opcode){
    std::size_t dot_pos = opcode.find('.');
    std::string opcode_short = opcode.substr(0, dot_pos);

    auto it = std::find(FP_LIST.begin(), FP_LIST.end(), opcode_short);
    return (it != FP_LIST.end())? true : false;
}

bool is_ld(std::string opcode){
    std::size_t dot_pos = opcode.find('.');
    std::string opcode_short = opcode.substr(0, dot_pos);

    auto it = std::find(LD_LIST.begin(), LD_LIST.end(), opcode_short);
    return (it != LD_LIST.end()) ? true : false;
}

bool is_st(std::string opcode){
    std::size_t dot_pos = opcode.find('.');
    std::string opcode_short = opcode.substr(0, dot_pos);

    auto it = std::find(ST_LIST.begin(), ST_LIST.end(), opcode_short);
    return (it != ST_LIST.end()) ? true : false;
}

bool file_exists(const std::string& file_path) {
    std::ifstream f(file_path);
    return f.good();
}

// Check if the directory exists. If there isn't, make one. 
bool create_a_directory(std::string dir_path, bool print) {
    const char* c_dir_path = dir_path.c_str();
    struct stat info;
    if (stat(c_dir_path, &info) != 0) {
        // Directory doesn't exist, create it
        int result = mkdir(c_dir_path, 0777);
        if (result != 0) {
            std::cerr << "Error: Failed to create directory." << std::endl;
            return 1;
        }
        if (print) std::cout << "Directory " << c_dir_path << " created." << std::endl;
    }
    else if (info.st_mode & S_IFDIR) {
        // Directory exists
        if (print) std::cout << "Directory " << c_dir_path << " already exists." << std::endl;
    }
    else {
        std::cerr << "Error: Path is not a directory." << std::endl;
        return 1;
    }
    return 0;
}

// Remove bracket in kernel name 
std::string rm_bracket (std::string kernel_name){
    size_t pos = kernel_name.find('(');
    return kernel_name.substr(0, pos);
}

// not sure..
std::string cf_type(std::string opcode){ 
    // NOT_CF,  //!< not a control flow instruction
    // CF_BR,  //!< an unconditional branch
    // CF_CBR,  //!< a conditional branch
    // CF_CALL,  //!< a call
    // // below this point are indirect cfs
    // CF_IBR,  //!< an indirect branch // non conditional
    // CF_ICALL,  //!< an indirect call
    // CF_ICO,  //!< an indirect jump to co-routine
    // CF_RET,  //!< a return
    // CF_MITE,  //!< alpha PAL, micro-instruction assited instructions

    std::size_t dot_pos = opcode.find('.');
    std::string opcode_short = opcode.substr(0, dot_pos);
    if (opcode_short == "JMP")
        return "CF_BR";
    else if (opcode_short == "BRA")
        return "CF_CBR";
    else if (opcode_short == "RET")
        return "CF_RET";
    else 
        return "NOT_CF";
}

uint8_t num_dst_reg(mem_access_t* ma){
    std::string opcode = id_to_opcode_map[ma->opcode_id];
    std::size_t dot_pos = opcode.find('.');
    std::string opcode_short = opcode.substr(0, dot_pos);
    if (is_st(opcode) || opcode_short == "BRA" || opcode_short == "EXIT" || opcode_short == "BAR"
                      || opcode_short == "BSSY" || opcode_short == "BSYNC" || opcode_short == "CALL"
                      || opcode_short == "BREAK")
        return 0;
    else 
        return 1;
}

void src_reg(mem_access_t* ma, uint16_t* src_reg_){
    for(int i=num_dst_reg(ma), j=0; i<ma->num_regs; i++, j++){
        src_reg_[j] = ma->reg_id[i];
    }
    return;
}

void dst_reg(mem_access_t* ma, uint16_t* dst_reg_){
    for(int i=0; i<num_dst_reg(ma); i++){
        dst_reg_[i] = ma->reg_id[i];
    }
    return;
}

int num_child_trace(uint64_t* mem_addrs, size_t size, uint32_t active_mask){
    uint64_t min_nonzero = (uint64_t)-1;
    uint64_t max = 0;
    for (int i = 0; i < (int)size; i++) {
        if (mem_addrs[i] != 0 && (active_mask & ( 1 << i )) >> i && mem_addrs[i] < min_nonzero) {
            min_nonzero = mem_addrs[i];
        }
        if ((active_mask & ( 1 << i )) >> i && mem_addrs[i] > max) {
            max = mem_addrs[i];
        }
    }
    // fix me!!
    int num_child = (min_nonzero == (uint64_t)-1) ? 0 : (int)(max - min_nonzero) / 128;
    return (num_child > 32) ? 32 : num_child;
}

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());

    /* set mutex as recursive */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);

    ThreadPool pool(num_threads);

    create_a_directory(rm_bracket(trace_path), false);
    std::ofstream file_kernel_config(trace_path + "kernel_config.txt");
    file_kernel_config << "nvbit" << std::endl;
    file_kernel_config << "14" << std::endl; // GPU Trace version
    file_kernel_config << "-1" << std::endl;
    file_kernel_config.close();
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* Kernel - id mapping */
UniqueKernelStore store;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    // int device_count;
    // cuDeviceGetCount(&device_count);
    // std::cout << device_count << std::endl;
    int device_id = 0; // device_id should be an integer between 0 and device_count - 1
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, device_id);
    int maxBlocksPerMultiprocessor;
    CUresult result = cuDeviceGetAttribute(&maxBlocksPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, cuDevice);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            // continue;
        }

        /* Making proper directories for trace files */
        std::string func_name = nvbit_get_func_name(ctx, f); // this function fetches the argument part too..
        int kernel_id = store.add(rm_bracket(func_name));

        std::string kernel_dir = rm_bracket(trace_path + func_name) + "_" + std::to_string(grid_launch_id);
        create_a_directory(kernel_dir, false);
        // std::string command = "rm " + kernel_dir + "/" + "*.*";
        // int status = system(command.c_str());
        // if (status == 0) {
        //     std::cout << "rm command executed successfully" << std::endl;
        // } else {
        //     std::cout << "rm command failed to execute" << std::endl;
        // }

        std::ofstream file_trace(kernel_dir + "/" + "trace.txt");
        file_trace << "nvbit" << std::endl;
        file_trace << "14" << std::endl; // GPU Trace version
        file_trace << maxBlocksPerMultiprocessor << std::endl;
        file_trace.close();

        /* get vector of instructions of function "f" */
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);

        if (verbose) {
            printf(
                "MEMTRACE: CTX %p, Inspecting CUfunction %p name %s at address "
                "0x%lx\n",
                ctx, f, nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
        }

        uint32_t cnt = 0;
        /* iterate on all the static instructions in the function */
        for (auto instr : instrs) {
            if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
                cnt++;
                continue;
            }
            if (verbose) {
                instr->printDecoded();
            }

            if (opcode_to_id_map.find(instr->getOpcode()) ==
                opcode_to_id_map.end()) {
                int opcode_id = opcode_to_id_map.size();
                opcode_to_id_map[instr->getOpcode()] = opcode_id;
                id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
            }

            int opcode_id = opcode_to_id_map[instr->getOpcode()];
            std::vector<int> reg_num_list;
            // int mref_idx = 0;
            /* iterate on the operands */
            for (int i = 0; i < instr->getNumOperands(); i++) {
                /* get the operand "i" */
                const InstrType::operand_t* op = instr->getOperand(i);

                /* count # of regs */
                if (op->type == InstrType::OperandType::REG || 
                    op->type == InstrType::OperandType::PRED || 
                    op->type == InstrType::OperandType::UREG || 
                    op->type == InstrType::OperandType::UPRED) {
                    // for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
                    //     reg_num_list.push_back(op->u.reg.num + reg_idx);
                    // } what is the purpose of this for-loop?
                    reg_num_list.push_back(op->u.reg.num);
                }
            }

            nvbit_insert_call(instr, "instrument_else", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val32(instr, kernel_id);
            nvbit_add_call_arg_const_val32(instr, opcode_id);
            // }
            /* add "space" for kernel function pointer that will be set
                    * at launch time (64 bit value at offset 0 of the dynamic
                    * arguments)*/
            nvbit_add_call_arg_launch_val64(instr, 0);
            /* add pointer to channel_dev*/
            nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
            /* instruction size */
            nvbit_add_call_arg_const_val32(instr, 4); // 32bit instructions?
            /* PC address */
            nvbit_add_call_arg_const_val64(instr, nvbit_get_func_addr(func) + instr->getOffset());
            /* Branch target address (should i care about predicates?) */
            uint64_t branchAddrOffset = (std::string(instr->getOpcodeShort()) == "BRA") ? 
                instr->getOperand(instr->getNumOperands()-1)->u.imm_uint64.value + nvbit_get_func_addr(func) :
                nvbit_get_func_addr(func) + instr->getOffset() + 0x10;
            nvbit_add_call_arg_const_val64(instr, branchAddrOffset);
            /*bool is_mem*/
            // nvbit_add_call_arg_const_val32(instr, instr->);
            /* MEM access address / reconv(??) address */
            nvbit_add_call_arg_mref_addr64(instr);
            /* MEM access size */
            nvbit_add_call_arg_const_val32(instr, (uint8_t)instr->getSize()); // i'm not sure what this getSize() function exactly does.. is it for getting instruction size or mem_access size?
            /* MEM addr space */
            nvbit_add_call_arg_const_val32(instr, (uint8_t)instr->getMemorySpace());
            /* how many register values are passed next */
            nvbit_add_call_arg_const_val32(instr, reg_num_list.size());
            // std::cout << instr->getSass() << std::endl;
            // std::cout << instr->getSass() << ", reg_num: " << reg_num_list.size() << std::endl;
            for (int num : reg_num_list) {
                /* last parameter tells it is a variadic parameter passed to
                * the instrument function record_reg_val() */
                nvbit_add_call_arg_const_val32(instr, num, true);
                // std::cout << "num: " << num << " ";
            }
            // std::cout << std::endl;
            cnt++;
        }
    }
}

__global__ void flush_channel(ChannelDev* ch_dev) {
    /* set a CTA id = -1 to indicate communication thread that this is the
     * termination flag */
    mem_access_t ma;
    ma.cta_id_x = -1;
    ch_dev->push(&ma, sizeof(mem_access_t));
    /* flush channel */
    ch_dev->flush();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    pthread_mutex_lock(&mutex);

    /* we prevent re-entry on this callback when issuing CUDA functions inside
     * this function */
    if (skip_callback_flag) {
        pthread_mutex_unlock(&mutex);
        return;
    }
    skip_callback_flag = true;

    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel) {
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;

        /* Make sure GPU is idle */
        cudaDeviceSynchronize();
        assert(cudaGetLastError() == cudaSuccess);

        if (!is_exit) {
            /* instrument */
            instrument_function_if_needed(ctx, p->f);

            int nregs = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

            int shmem_static_nbytes = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&shmem_static_nbytes,
                                   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, p->f));

            /* get function name and pc */
            const char* func_name = nvbit_get_func_name(ctx, p->f);
            uint64_t pc = nvbit_get_func_addr(p->f);

            /* set grid launch id at launch time */
            nvbit_set_at_launch(ctx, p->f, &grid_launch_id, sizeof(uint64_t));
            /* increment grid launch id for next launch */
            grid_launch_id++;

            /* enable instrumented code to run */
            nvbit_enable_instrumented(ctx, p->f, true);

            printf(
                "MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - Kernel "
                "name %s - grid launch id %ld - grid size %d,%d,%d - block "
                "size %d,%d,%d - nregs %d - shmem %d - cuda stream id %ld\n",
                (uint64_t)ctx, pc, func_name, grid_launch_id, p->gridDimX,
                p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY,
                p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes,
                (uint64_t)p->hStream);
        } else {
            // save to file
            std::cout << std::endl;
        }
    }
    skip_callback_flag = false;
    pthread_mutex_unlock(&mutex);
}

void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;
    ThreadPool pool(num_threads);

    pthread_mutex_lock(&mutex);
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    ChannelHost* ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);
    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);

    bool done = false;
    while (!done) {
        /* receive buffer from channel */
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                mem_access_t* ma =
                    (mem_access_t*)&recv_buffer[num_processed_bytes];

                /* when we receive a CTA_id_x it means all the kernels
                 * completed, this is the special token we receive from the
                 * flush channel kernel that is issues at the end of the
                 * context */
                if (ma->cta_id_x == -1) {
                    done = true;
                    break;
                }

                std::stringstream ss;

                int kernel_id = static_cast<int>(ma->grid_launch_id);
                std::string kernel_name = store.get_string(kernel_id) + "_" + std::to_string(kernel_id);
                std::string filename = "bin_trace_" + std::to_string(ma->warp_id) + ".txt";
                std::string filename_raw = "bin_trace_" + std::to_string(ma->warp_id) + ".raw";
                // const char * filename_gz = (trace_path +"trace_" + std::to_string(ma->warp_id) + ".gz").c_str();
                std::string opcode = id_to_opcode_map[ma->opcode_id];

                // find element with ma->warp_id in the map. 
                auto itt = store.instr_counts[kernel_id].find(ma->warp_id);
                if (itt != store.instr_counts[kernel_id].end()) {
                    itt->second++;
                } else {
                    store.instr_counts[kernel_id].insert({ma->warp_id, 1});
                }

                std::size_t dot_pos = opcode.find('.');
                std::string opcode_short = opcode.substr(0, dot_pos);
                uint8_t opcode_int = 255;
                auto it = std::find(std::begin(GPU_NVBIT_OPCODE), std::end(GPU_NVBIT_OPCODE), opcode_short);
                if (it != std::end(GPU_NVBIT_OPCODE)) {
                    opcode_int = (uint8_t)std::distance(std::begin(GPU_NVBIT_OPCODE), it);
                }
                uint8_t cf_type_int = 255;
                it = std::find(std::begin(CF_TYPE), std::end(CF_TYPE), cf_type(opcode));
                if (it != std::end(CF_TYPE)) {
                    cf_type_int = (uint8_t)std::distance(std::begin(CF_TYPE), it);
                }
                uint8_t num_dst_reg_ = num_dst_reg(ma);
                uint8_t num_src_reg_ = ma->num_regs - num_dst_reg_;
                uint16_t src_reg_[MAX_GPU_SRC_NUM];
                uint16_t dst_reg_[MAX_GPU_DST_NUM];
                memset(src_reg_, 0, sizeof(src_reg_));
                memset(dst_reg_, 0, sizeof(dst_reg_));
                src_reg(ma, src_reg_);
                dst_reg(ma, dst_reg_);
                uint8_t inst_size = ma->size; // always 4? 8?
                uint32_t active_mask = ma->active_mask;
                uint32_t br_taken_mask = 0; // should be added soon
                uint64_t func_addr = ma->func_addr;
                uint64_t br_target_addr = ma->branch_target_addr;
                uint64_t mem_addr = (is_ld(opcode) || is_st(opcode)) ? ma->mem_addr : 0; // or m_reconv_inst_addr
                uint8_t mem_access_size = ma->mem_access_size; // or m_barrier_id
                uint16_t m_num_barrier_threads = 0; // should be added soon
                uint8_t m_addr_space = ma->m_addr_space; // or m_level (memory barrier level)
                std::string m_addr_space_str = MemorySpaceStr[m_addr_space];
                uint8_t m_cache_level = 0; // should be added soon
                uint8_t m_cache_operator = 0; // should be added soon
                uint64_t mem_addrs[32];
                for (int i = 0; i < 32; i++) mem_addrs[i] = ma->addrs[i];
                // count 1s in active_mask
                uint8_t active_threads = __builtin_popcount(active_mask);
                // children thread number for memory operations
                // if(is_ld(opcode) || is_st(opcode)) br_target_addr = active_threads; 
                
                trace_info_nvbit_small_s cur_trace;
                cur_trace.m_opcode = opcode_int;
                cur_trace.m_is_fp = is_fp(opcode);
                cur_trace.m_is_load = is_ld(opcode);
                cur_trace.m_cf_type = cf_type_int;
                cur_trace.m_num_read_regs = num_src_reg_;
                cur_trace.m_num_dest_regs = num_dst_reg_;
                memcpy(cur_trace.m_src, src_reg_, sizeof(src_reg_));
                memcpy(cur_trace.m_dst, dst_reg_, sizeof(dst_reg_));
                cur_trace.m_size = inst_size;
                cur_trace.m_active_mask = active_mask;
                cur_trace.m_br_taken_mask = br_taken_mask;
                cur_trace.m_inst_addr = func_addr;
                cur_trace.m_br_target_addr = br_target_addr;
                cur_trace.m_mem_addr = mem_addr;
                cur_trace.m_mem_access_size = mem_access_size;
                cur_trace.m_num_barrier_threads = m_num_barrier_threads;
                cur_trace.m_addr_space = m_addr_space;
                cur_trace.m_cache_level = m_cache_level;
                cur_trace.m_cache_operator = m_cache_operator;
                if(store.warp_ids_s[kernel_id].find(ma->warp_id) == store.warp_ids_s[kernel_id].end()) {
                    store.warp_ids[kernel_id].push(ma->warp_id);
                    store.warp_ids_s[kernel_id].insert(ma->warp_id);
                }


                // size = 32 
                size_t size = sizeof(mem_addrs) / sizeof(mem_addrs[0]);
                int num_child_trace_ = num_child_trace(mem_addrs, size, active_mask);
                assert(num_child_trace_ <= 32);
                std::vector<trace_info_nvbit_small_s> children_trace;
                for (int i = 1; i < num_child_trace_; i++){
                    trace_info_nvbit_small_s child_trace;
                    memcpy(&child_trace, &cur_trace, sizeof(child_trace));
                    if (num_child_trace_ == 32){
                        child_trace.m_mem_addr = mem_addrs[i];
                    } else {
                        child_trace.m_mem_addr = mem_addrs[0] + i * 128;
                    }
                    children_trace.push_back(child_trace);
                }
                if (num_child_trace_ && (is_ld(opcode) || is_st(opcode))){
                    // std::cout << "num_child_trace: " << num_child_trace_ << std::endl;
                    cur_trace.m_mem_access_size *= num_child_trace_;
                    mem_access_size *= num_child_trace_;
                }

                pool.enqueue([filename, kernel_name, kernel_id, filename_raw, opcode, opcode_int, cf_type_int, num_src_reg_, num_dst_reg_, inst_size, 
                    src_reg_, dst_reg_, active_mask, br_taken_mask, func_addr, br_target_addr, mem_addr, 
                    mem_access_size, m_num_barrier_threads, m_addr_space, m_addr_space_str, m_cache_level, m_cache_operator, cur_trace, children_trace] {
                    pthread_mutex_lock(&file_mutex);
                    std::ofstream file(trace_path + kernel_name + "/" + filename, std::ios_base::app);
                    file << opcode << std::endl;
                    file << is_fp(opcode) << std::endl;
                    file << is_ld(opcode) << std::endl;
                    file << cf_type(opcode) << std::endl;
                    file << (int)num_src_reg_ << std::endl;
                    file << (int)num_dst_reg_ << std::endl;
                    file << src_reg_[0] << std::endl;
                    file << src_reg_[1] << std::endl;
                    file << src_reg_[2] << std::endl;
                    file << src_reg_[3] << std::endl;
                    file << dst_reg_[0] << std::endl;
                    file << dst_reg_[1] << std::endl;
                    file << dst_reg_[2] << std::endl;
                    file << dst_reg_[3] << std::endl;
                    file << (int)inst_size << std::endl;
                    file << std::hex << active_mask << std::endl;
                    file << std::hex << br_taken_mask << std::endl;
                    file << std::hex << func_addr << std::endl;
                    file << std::hex << br_target_addr << std::endl; 
                    file << std::hex << mem_addr << std::endl;
                    file << (int)mem_access_size << std::endl;
                    file << (int)m_num_barrier_threads << std::endl;
                    file << m_addr_space_str << std::endl;
                    file << (int)m_cache_level << std::endl;
                    file << (int)m_cache_operator << std::endl;
                    file << std::endl;
                    if(is_ld(opcode) || is_st(opcode)) { //children threads for ld/store
                        for (int i = 0; i < (int)children_trace.size(); i++){
                            file << opcode << " (child)" << std::endl;
                            file << is_fp(opcode) << std::endl;
                            file << is_ld(opcode) << std::endl;
                            file << cf_type(opcode) << std::endl;
                            file << (int)num_src_reg_ << std::endl;
                            file << (int)num_dst_reg_ << std::endl;
                            file << src_reg_[0] << std::endl;
                            file << src_reg_[1] << std::endl;
                            file << src_reg_[2] << std::endl;
                            file << src_reg_[3] << std::endl;
                            file << dst_reg_[0] << std::endl;
                            file << dst_reg_[1] << std::endl;
                            file << dst_reg_[2] << std::endl;
                            file << dst_reg_[3] << std::endl;
                            file << (int)inst_size << std::endl;
                            file << std::hex << active_mask << std::endl;
                            file << std::hex << br_taken_mask << std::endl;
                            file << std::hex << func_addr << std::endl;
                            file << std::hex << br_target_addr << std::endl; 
                            file << std::hex << mem_addr + (i+1) * 128 << std::endl;
                            file << (int)mem_access_size << std::endl;
                            file << (int)m_num_barrier_threads << std::endl;
                            file << m_addr_space_str << std::endl;
                            file << (int)m_cache_level << std::endl;
                            file << (int)m_cache_operator << std::endl;
                            file << std::endl;
                        }
                    }
                    file.close();
                
                    std::ofstream file_raw(trace_path + kernel_name + "/" + filename_raw, std::ios::binary | std::ios_base::app);
                    file_raw.write(reinterpret_cast<const char*>(&cur_trace), sizeof(cur_trace));
                    
                    if(is_ld(opcode) || is_st(opcode)) { //children threads for ld/store
                        for (int i = 0; i < (int)children_trace.size(); i++){
                            file_raw.write(reinterpret_cast<const char*>(&children_trace[i]), sizeof(children_trace[i]));
                        }
                    }
                    file_raw.close();
                    pthread_mutex_unlock(&file_mutex);
                });
                // if(is_ld(opcode) || is_st(opcode)) {//children threads for ld/store
                //     pool.enqueue([filename, filename_raw, opcode,mem_addrs]{
                //         for (int i=0;i<32;i++) if(mem_addrs[i]!=0) {
                //             std::ofstream file(trace_path + filename, std::ios_base::app);
                //             file << opcode << std::endl;
                //             file << 1 << std::endl; // set is_fp=1 for children
                //             // set the rest of the fields to be 0
                //             for(i=0;i<17;i++) file << 0 << std::endl;
                //             file << std::hex << mem_addrs[i] << std::endl;
                //             for(i=0;i<5;i++) file << 0 << std::endl;
                //             file << std::endl;
                //             file.close();
                //         }
                //     });
                // }
                num_processed_bytes += sizeof(mem_access_t);
            }
        }
    }

    // Print the elements in the heap in order
    store.create_trace_info();

    int idx = 0;
    for (auto s: store.kernels) {
        std::ofstream file_kernel_config(trace_path + "kernel_config.txt", std::ios_base::app);
        file_kernel_config << trace_path + s + "_" + std::to_string(idx) + "/trace.txt" << std::endl;
        file_kernel_config.close();
        idx++;
    }

    free(recv_buffer);
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    if (verbose) {
        printf("MEMTRACE: STARTING CONTEXT %p\n", ctx);
    }
    CTXstate* ctx_state = new CTXstate;
    assert(ctx_state_map.find(ctx) == ctx_state_map.end());
    ctx_state_map[ctx] = ctx_state;
    cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                                 ctx_state->channel_dev, recv_thread_fun, ctx);
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);

    skip_callback_flag = true;
    if (verbose) {
        printf("MEMTRACE: TERMINATING CONTEXT %p\n", ctx);
    }
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* flush channel */
    flush_channel<<<1, 1>>>(ctx_state->channel_dev);
    /* Make sure flush of channel is complete */
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);

    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback_flag = false;
    delete ctx_state;
    pthread_mutex_unlock(&mutex);
}
