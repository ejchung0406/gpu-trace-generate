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

run() {
  echo "dataset: $dataset"
  echo "subdir: $subdir"
  
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
}

####################################################################################

# decomment one of these and run ```bash run.sh```

# this directory should match with trace_path in /tools/main/mem_trace.cu and /tools/main/compress.cc !!
tracedir='/fast_data/echung67/trace/nvbit/'

name="backprop"
# dataset="128"
# subdir="128"
# run
dataset="256"
subdir="256"
run
# dataset="512"
# subdir="512"
# dataset="1024"
# subdir="1024"

# name="bfs"
# dataset="/fast_data/echung67/rodinia-data/bfs/graph1k.txt"
# subdir="graph1k"
# run
# dataset="/fast_data/echung67/rodinia-data/bfs/graph2k.txt"
# subdir="graph2k"
# run
# dataset="/fast_data/echung67/rodinia-data/bfs/graph4k.txt"
# subdir="graph4k"
# run
# dataset="/fast_data/echung67/rodinia-data/bfs/graph8k.txt"
# subdir="graph8k"
# run
# dataset="/fast_data/echung67/rodinia-data/bfs/graph16k.txt"
# subdir="graph16k"
# run
# dataset="/fast_data/echung67/rodinia-data/bfs/graph32k.txt"
# subdir="graph32k"
# run
# dataset="/fast_data/echung67/rodinia-data/bfs/graph64k.txt"
# subdir="graph64k"
# run

# name="euler3d"
# dataset="/fast_data/echung67/rodinia-data/cfd/fvcorr.domn.097K"
# subdir="fvcorr.domn.097K"
# run

# name="gaussian"
# dataset="-f /fast_data/echung67/rodinia-data/gaussian/matrix3.txt"
# subdir="matrix3"
# run
# dataset="-f /fast_data/echung67/rodinia-data/gaussian/matrix4.txt"
# subdir="matrix4"
# run
# dataset="-f /fast_data/echung67/rodinia-data/gaussian/matrix16.txt"
# subdir="matrix16"
# run

# name="heartwall"
# dataset="/fast_data/echung67/rodinia-data/heartwall/test.avi 10"
# subdir="frames10"
# run

# name="hotspot"
# dataset="512 512 1000 /fast_data/echung67/rodinia-data/hotspot/temp_512 /fast_data/echung67/rodinia-data/hotspot/power_512 none"
# subdir="r512h512i1000"
# run

# name="lavaMD"
# dataset="-boxes1d 1" 
# subdir="1"
# run
# dataset="-boxes1d 2" 
# subdir="2"
# run

# name="lud_cuda"
# dataset="-i /fast_data/echung67/rodinia-data/lud/64.dat" 
# subdir="64"
# run
# dataset="-i /fast_data/echung67/rodinia-data/lud/256.dat" 
# subdir="256"
# run
# dataset="-i /fast_data/echung67/rodinia-data/lud/512.dat" 
# subdir="512"
# run

# name="nn"
# dataset="/fast_data/echung67/rodinia-data/nn/inputGen/list64k.txt -r 30 -lat 30 -lng 90" 
# subdir="64k"
# run
# dataset="/fast_data/echung67/rodinia-data/nn/inputGen/list128k.txt -r 30 -lat 30 -lng 90" 
# subdir="128k"
# run
# dataset="/fast_data/echung67/rodinia-data/nn/inputGen/list256k.txt -r 30 -lat 30 -lng 90" 
# subdir="256k"
# run
# dataset="/fast_data/echung67/rodinia-data/nn/inputGen/list512k.txt -r 30 -lat 30 -lng 90" 
# subdir="512k"
# run
# dataset="/fast_data/echung67/rodinia-data/nn/inputGen/list1024k.txt -r 30 -lat 30 -lng 90" 
# subdir="1024k"
# run

# name="needle"
# dataset="32 10" 
# subdir="32"
# run
# dataset="64 10" 
# subdir="64"
# run
# dataset="128 10" 
# subdir="128"
# run

# name="srad_v1"
# dataset="3 0.5 64 64" 
# subdir="3"
# run
# dataset="10 0.5 64 64" 
# subdir="10"
# run

# name="srad_v2"
# dataset="64 64 0 32 0 32 0.5 10" 
# subdir="10"
# run

# name="sc_gpu"
# dataset="10 20 16 64 16 100 none none 1" 
# subdir="10-20-16-64-16-100"
# run
# dataset="3 3 4 16 16 4 none none 1"
# subdir="3-3-4-16-16-4"
# run
# dataset="2 5 4 16 16 32 none none 1"
# subdir="2-5-4-16-16-32"
# run

# name="particlefilter_naive"
# dataset="-x 128 -y 128 -z 10 -np 1000" 
# subdir="1000"
# run

# name="particlefilter_float"
# dataset="-x 64 -y 64 -z 5 -np 10" 
# subdir="10"
# run

# name="pathfinder"
# dataset="1024 20 10"
# subdir="10"
# run
# dataset="1024 20 50"
# subdir="50"
# run
# dataset="1024 20 100"
# subdir="100"
# run

# name="dwt2d"
# dataset="/fast_data/echung67/rodinia-data/dwt2d/rgb.bmp -d 1024x1024 -f -5 -l 3"
# subdir="1024"
# run 
# dataset="/fast_data/echung67/rodinia-data/dwt2d/192.bmp -d 192x192 -f -5 -l 3"
# subdir="192"
# run

# name="vectoradd"
# dataset="4096"
# subdir="4096"
# run
# dataset="65536"
# subdir="65536"
# run

# name="vectormultadd"
# dataset="4096"
# subdir="4096"
# run
# dataset="16384"
# subdir="16384"
# run

# name="graphbig_bfs_topo_atomic"

####################################################################################
