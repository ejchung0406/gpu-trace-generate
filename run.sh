# cd ./tools/main
# make
# cd -

# CUDA_INJECTION64_PATH=./tools/instr_count_bb/instr_count_bb.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/instr_count_cuda_graph/instr_count_cuda_graph.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/mem_printf/mem_printf.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/mov_replace/mov_replace.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/opcode_hist/opcode_hist.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/record_reg_vals/record_reg_vals.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/mem_trace/mem_trace.so ./test-apps/vectormultadd/vectormultadd > output.txt
# CUDA_INJECTION64_PATH=./tools/instr_count/instr_count.so ./test-apps/vectormultadd/vectormultadd
# CUDA_INJECTION64_PATH=./tools/main/main.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/main/main.so ./test-apps/vectormultadd/vectormultadd

####################################################################################
# uncomment one of these and run ```bash run.sh```

# name="bfs"
# dataset=/fast_data/echung67/rodinia-data/bfs/graph65536.txt
# subdir="graph65536"

# name="backprop"
# dataset=1024
# subdir=1024

# name="gaussian"
# dataset="-f /fast_data/echung67/rodinia-data/gaussian/matrix3.txt"
# subdir="matrix3"

# name="vectoraddd"
# dataset=""
# subdir=""

# name="vectormultaddd"
# dataset=""
# subdir=""

####################################################################################

# if [ -z "$dataset" ]; then
#   rm -rf /fast_data/trace/nvbit/${name}
#   echo -e "1\n/fast_data/trace/nvbit/${name}/kernel_config.txt" > /fast_data/echung67/macsim/bin/trace_file_list
# else
#   rm -rf /fast_data/trace/nvbit/${name}/${subdir}
#   echo -e "1\n/fast_data/trace/nvbit/${name}/${subdir}/kernel_config.txt" > /fast_data/echung67/macsim/bin/trace_file_list
# fi
# cd ./tools/main
# make
# cd -
# CUDA_INJECTION64_PATH=./tools/main/main.so /fast_data/echung67/gpu-rodinia/bin/linux/cuda/${name} ${dataset}
# ./tools/main/compress
# if [ -z "$dataset" ]; then
#   mv /fast_data/trace/nvbit/temp /fast_data/trace/nvbit/${name}
# else
#   mkdir /fast_data/trace/nvbit/${name}
#   mv /fast_data/trace/nvbit/temp /fast_data/trace/nvbit/${name}/${subdir}
# fi

#################################################################

