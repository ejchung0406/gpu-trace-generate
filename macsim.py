#!/usr/bin/python
#########################################################################################
# Author: Jaekyu Lee (kacear@gmail.com)
# Date: 7/17/2011
# Description:
#   submit a batch of macsim simulation jobs
#   see README for examples
#
# Note:
#   1. when you specify -cmd option, please remove all '--'
#   2. all .out .stat.out. files will be automatically gzipped unless specifying -disable-gzip
#   3. by specifying -nproc, you can choose number of nodes to be allocated to your jobs
#########################################################################################

import sys
import os
import argparse
import subprocess

def process_options():
  parser = argparse.ArgumentParser(description='run.py')
  parser.add_argument('-proc', action='store', default='1', dest='nproc', help='number of processors that this job requires')
  return parser

def trace_available(file_path):
  if not os.path.exists(file_path):
    print(f"File path {file_path} doesn't exist.")
    return False
  
  with open(file_path, "r") as file:
    contents = file.read()
  
  if "Success" in contents:
    return True
  
  print(f"trace {file_path} is not available.")
  return False

def rodinia(argv):
  global args

  # parse arguments
  parser = process_options()
  args = parser.parse_args()
  current_dir = os.getcwd()

  ## path to binary
  macsim_files = ["/fast_data/echung67/macsim/bin/macsim",
                  "/fast_data/echung67/macsim/bin/params.in",
                  "/fast_data/echung67/macsim/bin/trace_file_list"]
  trace_path_base = "/fast_data/echung67/trace/nvbit/"
  rodinia_bin = "/fast_data/echung67/gpu-rodinia/bin/linux/cuda/"
  nvbit_bin = "/fast_data/echung67/nvbit_release/tools/main/main.so"
  compress_bin = "/fast_data/echung67/nvbit_release/tools/main/compress"
  result_dir = os.path.join(current_dir, "run")

  benchmark_names = [
    # Rodinia
    # "backprop",
    # "bfs",
    # "dwt2d",
    # "euler3d",
    # "gaussian",
    # "heartwall",
    # "hotspot",
    # "lavaMD",
    # "lud_cuda",
    # "needle",
    # "nn",
    # "particlefilter_float",
    # "particlefilter_naive",
    # "pathfinder",
    # "sc_gpu",
    # "srad_v1",
    # "srad_v2",

    # GraphBig
    # "graphbig_bfs_topo_atomic", // To-do: add bin path

    # Gunrock


    # ETC
    "vectoradd",
    # "vectormultadd",
  ]

  benchmark_subdir = {
    # Rodinia
    "backprop": ["128", "256", "512", "1024", "2048", "4096", "8192", "16384",
                 "32768", "65536", "131072", "262144", "524288", "1048576"],
    "bfs": ["graph1k", "graph2k", "graph4k", "graph8k", "graph16k", "graph32k",
              "graph64k", "graph128k", "graph256k", "graph512k"],
    "dwt2d": ["192", "1024"],
    "euler3d": ["fvcorr.domn.097K"],
    "gaussian": ["matrix3", "matrix4", "matrix16", "matrix32", "matrix48", "matrix64", "matrix80", "matrix96", "matrix112", "matrix128"],
    "heartwall": ["frames10"],
    "hotspot": ["r512h512i100", "r512h512i1000", "r512h2i2"],
    "lavaMD": ["1", "2", "3", "5", "7", "10"],
    "lud_cuda": ["64", "256", "512"],
    "needle": ["32", "64", "128"],
    "nn": ["64k", "128k", "256k", "512k", "1024k", "2048k", "4096k", "8192k", "16384k", "32768k"],
    "particlefilter_float": ["10"],
    "particlefilter_naive": ["1000"],
    "pathfinder": ["10", "50", "100"],
    "sc_gpu": ["2-5-4-16-16-32", "3-3-4-16-16-4", "10-20-16-64-16-100"],
    "srad_v1": ["3", "6", "10"],
    "srad_v2": ["10"],  

    # GraphBig
    # "graphbig_bfs_topo_atomic": [""],

    # Gunrock


    # ETC
    "vectoradd": ["4096", "16384", "65536"],
    "vectormultadd": ["4096", "16384", "65536"]
  }

  for bench_name in benchmark_names:
    bench_subdirs = benchmark_subdir[bench_name]
    for bench_subdir in bench_subdirs:
      # if (bench_name != "bfs" and bench_name != "backprop"): continue
      # create the result directory
      # assume nvbit.py has been run 
      subdir = os.path.join(result_dir, bench_name, bench_subdir)
      os.chdir(subdir)

      if not (trace_available(os.path.join(subdir, "nvbit_result.txt"))):
        continue

      for macsim_file in macsim_files:
        os.system(f"cp {macsim_file} {subdir}")
      # os.system(f"cp {nvbit_bin} {subdir}")
      # os.system(f"cp {compress_bin} {subdir}")

      python_file = os.path.join(subdir, "macsim.py")
      with open(python_file, "w") as f:
        f.write("import os\n\n")
        f.write("with open(\"trace_file_list\", \"w\") as f:\n")
        f.write(f"    f.write(\"1\\n\" + os.path.join(\"{trace_path_base}\", \"{bench_name}\", \"{bench_subdir}\", \"kernel_config.txt\"))\n\n")
        f.write("os.system('./macsim > macsim_result.txt')\n")

      # Execute nvbit python script
      subprocess.Popen(["nohup python3 macsim.py"], shell=True, cwd=subdir)
  return

def fast_tf(argv, fast=True):
  global args

  # parse arguments
  parser = process_options()
  args = parser.parse_args()
  current_dir = os.getcwd()

  ## path to binary
  macsim_files = ["/fast_data/echung67/macsim/bin/macsim",
                  "/fast_data/echung67/macsim/bin/params.in",
                  "/fast_data/echung67/macsim/bin/trace_file_list"]
  if fast:
    trace_path_base = "/fast_data/echung67/trace/nvbit/"
  else:
    trace_path_base = "/data/echung67/trace/nvbit/"
  tf_bin = "/fast_data/echung67/FastTransformer/bin"
  if fast:
    result_dir = os.path.join(current_dir, "run")
  else:
    result_dir = os.path.join("/data/echung67/", "run")

  benchmark_names = [
    # FasterTransformer
    # "bert_example",
    # "decoding_example",
    # # "gpt_example",
    # # "layernorm_test",
    # "swin_example",
    "vit_example",
    # "wenet_decoder_example",
    # "wenet_encoder_example",
    # "xlnet_example",
  ]

  # benchmark_configs = {
  #   "bert_example": ["32 12 32 12 64 0 0"],
  #   "decoding_example": ["4 1 8 64 2048 30000 6 32 32 512 0 0.6 1"],
  #   "gpt_example": [""],
  #   "layernorm_test": ["1 1024 1"],
  #   "swin_example": ["2 1 0 8 256 32"],
  #   "vit_example": ["32 384 16 768 12 12 1 0"],
  #   "wenet_decoder_example": ["16 12 256 4 64 1"],
  #   "wenet_encoder_example": ["16 12 256 4 64 1"],
  #   "xlnet_example": ["8 12 128 12 64 0"],
  # }

  benchmark_configs = {
    "bert_example": ["1 1 32 4 64 0 0"],
    "decoding_example": ["1 1 4 32 16 100 1 32 32 16 0 0.6 1"],
    "swin_example": ["1 1 0 8 192 1"],
    "vit_example": ["1 32 16 16 4 1 1 0"],
    "wenet_decoder_example": ["1 1 32 4 64 0"],
    "wenet_encoder_example": ["1 1 32 4 64 0"],
    "xlnet_example": ["1 1 32 4 64 0"],
  }

  max_inst = 20

  for bench_name in benchmark_names:
    bench_config = benchmark_configs[bench_name]
    for bench_conf in bench_config:
      # create the result directory
      subdir = os.path.join(result_dir, bench_name, f"{max_inst}")
      # if (bench_name != "bfs" and bench_name != "backprop"): continue
      # create the result directory
      # assume nvbit.py has been run 
      os.chdir(subdir)

      for macsim_file in macsim_files:
        os.system(f"cp {macsim_file} {subdir}")

      if not (trace_available(os.path.join(subdir, "nvbit_result.txt"))):
        continue

      python_file = os.path.join(subdir, "macsim.py")
      with open(python_file, "w") as f:
        f.write("import os\n\n")
        f.write("with open(\"trace_file_list\", \"w\") as f:\n")
        f.write(f"    f.write(\"1\\n\" + os.path.join(\"{trace_path_base}\", \"{bench_name}\", \"{max_inst}\", \"kernel_config.txt\"))\n\n")
        f.write("os.system('./macsim > macsim_result.txt')\n")

      # Execute nvbit python script
      subprocess.Popen(["nohup python3 macsim.py"], shell=True, cwd=subdir)
  return

def tango(argv, fast=True):
  global args

  # parse arguments
  parser = process_options()
  args = parser.parse_args()
  current_dir = os.getcwd()

  ## path to binary
  macsim_files = ["/fast_data/echung67/macsim/bin/macsim",
                  "/fast_data/echung67/macsim/bin/params.in",
                  "/fast_data/echung67/macsim/bin/trace_file_list"]
  if fast:
    trace_path_base = "/fast_data/echung67/trace/nvbit/"
  else:
    trace_path_base = "/data/echung67/trace/nvbit/"
  tango_bin = "/fast_data/echung67/Tango/GPU/"
  if fast:
    result_dir = os.path.join(current_dir, "run")
  else:
    result_dir = os.path.join("/data/echung67/", "run")

  benchmark_names = [
    # FasterTransformer
    ["AlexNet", "AN"],
    # ["CifarNet", "CN"],
    # ["GRU", "GRU"],
    # ["LSTM", "LSTM"],
    # ["ResNet", "RN"],
    # ["SqueezeNet", "SN"],
  ]

  benchmark_configs = {
    "AlexNet": ["100"],
    "CifarNet": ["100"],
    "GRU": [""],
    "LSTM": ["100"],
    "ResNet": [""],
    "SqueezeNet": ["100"],
  }

  for bench_name in benchmark_names:
    bench_config = benchmark_configs[bench_name[0]]
    for bench_conf in bench_config:
      # if (bench_name != "bfs" and bench_name != "backprop"): continue
      # create the result directory
      # assume nvbit.py has been run 
      subdir = os.path.join(result_dir, bench_name[0], "default")
      os.chdir(subdir)

      for macsim_file in macsim_files:
        os.system(f"cp {macsim_file} {subdir}")
      
      if not (trace_available(os.path.join(subdir, "nvbit_result.txt"))):
        continue

      python_file = os.path.join(subdir, "macsim.py")
      with open(python_file, "w") as f:
        f.write("import os\n\n")
        f.write("with open(\"trace_file_list\", \"w\") as f:\n")
        f.write(f"    f.write(\"1\\n\" + os.path.join(\"{trace_path_base}\", \"{bench_name[0]}\", \"default\", \"kernel_config.txt\"))\n\n")
        f.write("os.system('./macsim > macsim_result.txt')\n")

      # Execute nvbit python script
      subprocess.Popen(["nohup python3 macsim.py"], shell=True, cwd=subdir)
  return

if __name__ == '__main__':
  # rodinia(sys.argv)
  # tango(sys.argv, fast=False)
  fast_tf(sys.argv, fast=False)
