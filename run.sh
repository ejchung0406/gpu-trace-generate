# cd ./tools/main
# make
# cd -

# CUDA_INJECTION64_PATH=./tools/instr_count_bb/instr_count_bb.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/instr_count_cuda_graph/instr_count_cuda_graph.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/mem_printf/mem_printf.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/mov_replace/mov_replace.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/opcode_hist/opcode_hist.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/record_reg_vals/record_reg_vals.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/mem_trace/mem_trace.so ./test-apps/vectormultadd/vectormultadd
# CUDA_INJECTION64_PATH=./tools/instr_count/instr_count.so ./test-apps/vectormultadd/vectormultadd
# CUDA_INJECTION64_PATH=./tools/main/main.so ./test-apps/vectoradd/vectoradd
# CUDA_INJECTION64_PATH=./tools/main/main.so ./test-apps/vectormultadd/vectormultadd

####################################################################################
# uncomment one of these and run ```bash run.sh```

# this directory should match with trace_path in mem_trace.cu and compress.cc !!
tracedir='/fast_data/echung67/trace/nvbit/'

# name="backprop"
# dataset=1024
# subdir=1024

name="bfs"
dataset=/fast_data/echung67/rodinia-data/bfs/graph1k.txt
subdir="graph1k"

# name="euler3d"
# dataset=/fast_data/echung67/rodinia-data/cfd/fvcorr.domn.097K
# subdir="fvcorr.domn.097K"

# name="gaussian"
# dataset="-f /fast_data/echung67/rodinia-data/gaussian/matrix3.txt"
# subdir="matrix3"

# name="heartwall"
# dataset="/fast_data/echung67/rodinia-data/heartwall/test.avi 10"
# subdir="frames10"

# name="hotspot"
# dataset="512 512 100 /fast_data/echung67/rodinia-data/hotspot/temp_512 /fast_data/echung67/rodinia-data/hotspot/power_512 none"
# subdir="r512h512i100"

# name="lavaMD"
# dataset="-boxes1d 1" 
# subdir="1"

# name="lud_cuda"
# dataset="-i /fast_data/echung67/rodinia-data/lud/256.dat" 
# subdir="256"

# name="nn"
# dataset="/fast_data/echung67/rodinia-data/nn/inputGen/list512k.txt -r 30 -lat 30 -lng 90" 
# subdir="512k"

# name="needle"
# dataset="128 10" 
# subdir="128"

# name="srad_v1"
# dataset="6 0.5 64 64" 
# subdir="6"

# name="srad_v2"
# dataset="64 64 0 32 0 32 0.5 10" 
# subdir="10"

# name="sc_gpu"
# dataset="10 20 16 64 16 100 none none 1" 
# subdir="10-20-16-64-16-100"

# name="sc_gpu"
# dataset="3 3 4 16 16 4 none none 1"
# subdir="3-3-4-16-16-4"

# name="sc_gpu"
# dataset="2 5 4 16 16 32 none none 1"
# subdir="2-5-4-16-16-32"

# name="particlefilter_naive"
# dataset="-x 128 -y 128 -z 10 -np 1000" 
# subdir="100"

# name="particlefilter_float"
# dataset="-x 64 -y 64 -z 5 -np 1" 
# subdir="1"

# name="pathfinder"
# dataset="1024 20 10" 
# subdir="10"

# name="vectoraddd"
# dataset=""
# subdir=""

# name="vectormultaddd"
# dataset=""
# subdir=""

####################################################################################

if [ -z "$dataset" ]; then
  rm -rf ${tracedir}${name}
  echo -e "1\n${tracedir}${name}/kernel_config.txt" > /fast_data/echung67/macsim/bin/trace_file_list
else
  rm -rf ${tracedir}${name}/${subdir}
  echo -e "1\n${tracedir}${name}/${subdir}/kernel_config.txt" > /fast_data/echung67/macsim/bin/trace_file_list
fi
cd ./tools/main
make
cd -
CUDA_INJECTION64_PATH=./tools/main/main.so /fast_data/echung67/gpu-rodinia/bin/linux/cuda/${name} ${dataset}
./tools/main/compress
if [ -z "$dataset" ]; then
  mv ${tracedir}temp ${tracedir}${name}
else
  mkdir ${tracedir}${name}
  mv ${tracedir}temp ${tracedir}${name}/${subdir}
fi
cd /fast_data/echung67/macsim/bin
./macsim
cd -

####################################################################################

