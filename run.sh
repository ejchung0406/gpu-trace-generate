# CUDA_INJECTION64_PATH=./tools/instr_count_bb/instr_count_bb.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/instr_count_cuda_graph/instr_count_cuda_graph.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/mem_printf/mem_printf.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/mov_replace/mov_replace.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/opcode_hist/opcode_hist.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/record_reg_vals/record_reg_vals.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/mem_trace/mem_trace.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/instr_count/instr_count.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/main/main.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/main/main.so ./test-apps/vectormultadd/vectormultadd

# name="bfs"
# dataset=/fast_data/echung67/rodinia-data/bfs/graph4096.txt

# name="backprop"
# dataset=1024

# name="gaussian"
# dataset="-f /fast_data/echung67/rodinia-data/gaussian/matrix4.txt"

name="vectoraddd"
dataset=""

# name="vectormultaddd"
# dataset=""

########################=###########################################################

rm -rf /fast_data/trace/nvbit/${name}
cd ./tools/main
make
cd -
echo -e "1\n/fast_data/trace/nvbit/${name}/kernel_config.txt" > /fast_data/echung67/macsim/bin/trace_file_list
CUDA_INJECTION64_PATH=./tools/main/main.so /fast_data/echung67/gpu-rodinia/bin/linux/cuda/${name} ${dataset}
./tools/main/compress
mv /fast_data/trace/nvbit/temp /fast_data/trace/nvbit/${name}