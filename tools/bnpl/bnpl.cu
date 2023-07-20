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
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <cstring>
#include <map>
#include <set>
#include <unordered_set>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
pthread_mutex_t mutex;

/* opcode to id map */
std::map<std::string, int> opcode_to_id_map;

struct Argument_s {
    std::string type;
    size_t pointerLevel;
    std::string value;
};

void nvbit_at_init() {
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 1, "Enable verbosity inside the tool");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

/* Set to store register IDs with the pointer */
std::unordered_set<int> pointerRegIDs; 

std::vector<Argument_s> extractArgumentValues(const std::string& func_name, void** args) {
    std::vector<Argument_s> argumentValues;

    char* string_inside_parentheses = nullptr;
    size_t start_pos = func_name.find('(');
    size_t end_pos = func_name.find(')');
    if (start_pos != std::string::npos && end_pos != std::string::npos && end_pos > start_pos + 1) {
        std::string substring = func_name.substr(start_pos + 1, end_pos - start_pos - 1);
        string_inside_parentheses = new char[substring.length() + 1];
        std::strcpy(string_inside_parentheses, substring.c_str());
    }

    // Split the func_name into individual tokens
    char* token = std::strtok(string_inside_parentheses, ",");
    while (token != nullptr) {
        std::string token_str(token);
        // Remove leading and trailing whitespaces from each token
        size_t start = token_str.find_first_not_of(" ");
        size_t end = token_str.find_last_not_of(" ");
        token_str = token_str.substr(start, end - start + 1);

        // Extract the type and pointer level from the token
        size_t typeEnd = token_str.find_last_of('*');
        std::string type;
        size_t pointerLevel;
        
        if (typeEnd != std::string::npos) {
            type = token_str.substr(0, typeEnd + 1);
            pointerLevel = token_str.substr(typeEnd).length();
        } else {
            type = token_str;
            pointerLevel = 0;
        }

        std::cout << "\"" << type << "\"" << std::endl;

        std::string value;
        if (pointerLevel == 0) {
            value = "None"; // Dummy value for non-pointer values
        } else if (pointerLevel > 0) {
            void** argValue = static_cast<void**>(args[argumentValues.size()]);
            value = std::to_string(reinterpret_cast<uintptr_t>(*argValue));
        }

        Argument_s new_entry;
        new_entry.pointerLevel = pointerLevel;
        new_entry.type = type;
        new_entry.value = value;
        argumentValues.push_back(new_entry);
        token = std::strtok(nullptr, ",");
    }

    delete[] string_inside_parentheses;
    return argumentValues;
}

void instrument_function_if_needed(CUcontext ctx, CUfunction func, void** kernelParams) {
    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    // std::vector<int> kernel_arg_size_v = nvbit_get_kernel_argument_sizes(func);
    std::string func_name(nvbit_get_func_name(ctx, func));
    std::vector<Argument_s> kernel_args_v = extractArgumentValues(func_name, kernelParams);

    /* iterate on function */
    for (auto f : related_functions) {
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, func);
        int num_of_args = kernel_args_v.size();
        std::set<int> reg_num_list;
        
        for (auto instr : instrs) {
            std::cout << "reg_num_list: ";
            for (const auto& element : reg_num_list) {
                std::cout << element << " ";
            }
            std::cout << std::endl;
            instr->print();

            if (num_of_args > 0 && strcmp(instr->getOpcodeShort(), "LDC") == 0){
                // How do we know that the value in CBANK is a pointer?
                // Assume every argument is loaded with the same order as the function name
                if (kernel_args_v[kernel_args_v.size() - num_of_args].pointerLevel > 0){
                    const InstrType::operand_t *op_src = instr->getOperand(1);
                    if (op_src->type == InstrType::OperandType::CBANK) {
                        const InstrType::operand_t *op_dst = instr->getOperand(0);

                        // Assume we are using only the first 32-bit register (which contains the lower 32bit of the mem addr)
                        // for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
                        reg_num_list.insert(op_dst->u.reg.num);

                        nvbit_insert_call(instr, "arg_check", IPOINT_AFTER);
                        nvbit_add_call_arg_reg_val(instr, op_dst->u.reg.num);
                        nvbit_add_call_arg_reg_val(instr, op_dst->u.reg.num + 1);
                        // }
                    }
                }
                num_of_args--;
                continue;
            }

            

            // if (strcmp(instr->getOpcodeShort(), "MOV") == 0) {
            //     for (int i = 1; i < instr->getNumOperands(); i++) {
            //         const InstrType::operand_t *op = instr->getOperand(i);
            //         if (op->type == InstrType::OperandType::REG) {
            //             if (reg_num_list.find(op->u.reg.num) != reg_num_list.end()) {
            //                 const InstrType::operand_t *op_dst = instr->getOperand(0);
            //                 reg_num_list.insert(op_dst->u.reg.num);
            //             }

            //             // for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
            //             //     if (reg_num_list.find(op->u.reg.num + reg_idx) != reg_num_list.end()) {
            //             //         const InstrType::operand_t *op_dst = instr->getOperand(0);
            //             //         for (int reg_idx_ = 0; reg_idx_ < instr->getSize() / 4; reg_idx_++) {
            //             //             reg_num_list.insert(op_dst->u.reg.num + reg_idx_);
            //             //         }
            //             //     }
            //             // }
            //         }
            //     }
            //     continue;
            // }

            for (int i = 0; i < instr->getNumOperands(); i++) {
                const InstrType::operand_t *op = instr->getOperand(i);
                if (op->type == InstrType::OperandType::REG) {
                    // for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
                    //     if (reg_num_list.find(op->u.reg.num + reg_idx) != reg_num_list.end()) {
                    //         break;
                    //     }
                    // }
                    if (reg_num_list.find(op->u.reg.num) != reg_num_list.end()) {
                        // if (instr->getSize() == 8) {
                        nvbit_insert_call(instr, "bound_check", IPOINT_BEFORE);
                        nvbit_add_call_arg_guard_pred_val(instr);
                        nvbit_add_call_arg_reg_val(instr, op->u.reg.num);
                        nvbit_add_call_arg_reg_val(instr, op->u.reg.num + 1);
                        nvbit_add_call_arg_const_val64(instr, 12345);
                        // }
                    }
                }
            }
        }
    }
}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids which are defined in tools_cuda_api_meta.h.
 * This call back is triggered both at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, p->f, p->kernelParams);
            nvbit_enable_instrumented(ctx, p->f, true);
        } else {
            CUDA_SAFECALL(cudaDeviceSynchronize());
            pthread_mutex_unlock(&mutex);
        }
    }
}
