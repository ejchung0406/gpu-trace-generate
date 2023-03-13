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
#include <condition_variable>
#include <functional>
#include <algorithm>

#define MAX_GPU_SRC_NUM 4
#define MAX_GPU_DST_NUM 4

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

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

/* OPCODE list */
std::vector<std::string> FP_LIST;
std::vector<std::string> LD_LIST;
std::vector<std::string> ST_LIST;

/* grid launch id, incremented at every launch */
uint64_t grid_launch_id = 0;

/* # of thread */
const int THREAD_NUM = 100;
const size_t num_files = THREAD_NUM;
const size_t num_threads = 8;

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

int num_dst_reg(mem_access_t* ma){
    std::string opcode = id_to_opcode_map[ma->opcode_id];
    std::size_t dot_pos = opcode.find('.');
    std::string opcode_short = opcode.substr(0, dot_pos);
    if (opcode_short == "BRA" || opcode_short == "EXIT" || is_st(opcode))
        return 0;
    else 
        return 1;
}

void src_reg(mem_access_t* ma, int* src_reg_){
    for(int i=num_dst_reg(ma), j=0; i<ma->num_regs; i++, j++){
        src_reg_[j] = ma->reg_id[i];
    }
    return;
}

void dst_reg(mem_access_t* ma, int* dst_reg_){
    for(int i=0; i<num_dst_reg(ma); i++){
        dst_reg_[i] = ma->reg_id[i];
    }
    return;
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

    // std::ifstream file("insts.txt");
    // std::vector<std::string> GPU_OPCODE_LIST;
    // std::string line;
    std::ifstream file_fl("/home/echung67/nvbit_release/tools/main/insts/floating.txt");
    std::string line_fl;
    std::ifstream file_ld("/home/echung67/nvbit_release/tools/main/insts/load.txt");
    std::string line_ld;
    std::ifstream file_st("/home/echung67/nvbit_release/tools/main/insts/store.txt");
    std::string line_st;
    // while (std::getline(file, line)) {
    //     GPU_OPCODE_LIST.push_back(line);
    // }
    while (std::getline(file_fl, line_fl)) {
        FP_LIST.push_back(line_fl);
    }
    while (std::getline(file_ld, line_ld)) {
        LD_LIST.push_back(line_ld);
    }
    while (std::getline(file_st, line_st)) {
        ST_LIST.push_back(line_st);
    }
    // for (const auto& l : LD_LIST) {
    //     std::cout << l << std::endl;
    // }

    ThreadPool pool(num_threads);
    for (size_t i = 0; i < num_files; i++) {
        std::string filename = "trace_" + std::to_string(i) + ".txt";
        pool.enqueue([filename, i] {
            std::ofstream file("/home/echung67/nvbit_release/tools/main/trace/" + filename);
            file << "Thread " << i << std::endl;
            file.close();
        });
    }
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

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
            // if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
            //     instr->getMemorySpace() == InstrType::MemorySpace::NONE ||
            //     instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT) {
            //     cnt++;
            //     continue;
            // }
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
                    op->type == InstrType::OperandType::PRED) {
                    for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
                        reg_num_list.push_back(op->u.reg.num + reg_idx);
                    }
                }
            }

                // if (op->type == InstrType::OperandType::MREF) {
                //     /* insert call to the instrumentation function with its
                //      * arguments */
                //     nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
                //     /* predicate value */
                //     nvbit_add_call_arg_guard_pred_val(instr);
                //     /* opcode id */
                //     nvbit_add_call_arg_const_val32(instr, opcode_id);
                //     /* memory reference 64 bit address */
                //     nvbit_add_call_arg_mref_addr64(instr, mref_idx);
                //     mref_idx++;
                // } else {
            nvbit_insert_call(instr, "instrument_else", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val32(instr, opcode_id);
            // }
            /* add "space" for kernel function pointer that will be set
                    * at launch time (64 bit value at offset 0 of the dynamic
                    * arguments)*/
            nvbit_add_call_arg_launch_val64(instr, 0);
            /* add pointer to channel_dev*/
            nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
            /* size */
            nvbit_add_call_arg_const_val32(instr, instr->getSize());
            /* how many register values are passed next */
            nvbit_add_call_arg_const_val32(instr, reg_num_list.size());
            std::cout << instr->getSass() << ", reg_num: " << reg_num_list.size() << std::endl;
            for (int num : reg_num_list) {
                /* last parameter tells it is a variadic parameter passed to
                * the instrument function record_reg_val() */
                nvbit_add_call_arg_const_val32(instr, num, true);
                std::cout << "num: " << num << " ";
            }
            std::cout << std::endl;
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
                // ss << "CTX " << HEX(ctx) << " - grid_launch_id "
                //    << ma->grid_launch_id << " - CTA " << ma->cta_id_x << ","
                //    << ma->cta_id_y << "," << ma->cta_id_z << " - warp "
                //    << ma->warp_id << " - " << id_to_opcode_map[ma->opcode_id]
                //    << " - ";

                // for (int i = 0; i < 32; i++) {
                //     ss << HEX(ma->addrs[i]) << " ";
                // }

                std::string filename = "trace_" + std::to_string(ma->thread_id) + ".txt";
                std::string opcode = id_to_opcode_map[ma->opcode_id];
                int num_dst_reg_ = num_dst_reg(ma);
                int num_src_reg_ = ma->num_regs - num_dst_reg_;
                int src_reg_[MAX_GPU_SRC_NUM];
                int dst_reg_[MAX_GPU_DST_NUM];
                memset(src_reg_, 0, sizeof(src_reg_));
                memset(dst_reg_, 0, sizeof(dst_reg_));
                src_reg(ma, src_reg_);
                dst_reg(ma, dst_reg_);
                int size = ma->size;

                ss << "Thread id: " << ma->thread_id << ", Opcode: " << opcode << ", isFp: " << is_fp(opcode)
                   << ", isLoad: " << is_ld(opcode) << ", cfType: " << cf_type(opcode);
                // printf("%s\n", ss.str().c_str());
                pool.enqueue([filename, opcode, num_src_reg_, num_dst_reg_, size, src_reg_, dst_reg_] {
                    std::ofstream file("/home/echung67/nvbit_release/tools/main/trace/" + filename, std::ios_base::app);
                    file << opcode << std::endl;
                    file << is_fp(opcode) << std::endl;
                    file << is_ld(opcode) << std::endl;
                    file << cf_type(opcode) << std::endl;
                    file << num_src_reg_ << std::endl;
                    file << num_dst_reg_ << std::endl;
                    file << src_reg_[0] << std::endl;
                    file << src_reg_[1] << std::endl;
                    file << src_reg_[2] << std::endl;
                    file << src_reg_[3] << std::endl;
                    file << dst_reg_[0] << std::endl;
                    file << dst_reg_[1] << std::endl;
                    file << dst_reg_[2] << std::endl;
                    file << dst_reg_[3] << std::endl;
                    file << size << std::endl;
                    file << std::endl;
                    file.close();
                });

                num_processed_bytes += sizeof(mem_access_t);
            }
        }
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
