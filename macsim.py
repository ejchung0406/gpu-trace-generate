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

def main(argv):
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
    "backprop",
    "bfs",
    # "dwt2d",
    # "euler3d",
    "gaussian",
    "heartwall",
    "hotspot",
    "lavaMD",
    "lud_cuda",
    "needle",
    "nn",
    "particlefilter_float",
    "particlefilter_naive",
    "pathfinder",
    "sc_gpu",
    "srad_v1",
    "srad_v2",
    
    # GraphBig
    # "graphbig_bfs_topo_atomic", // add bin path ??

    # Gunrock


    # ETC
    "vectoradd",
    "vectormultadd"]

  benchmark_dataset = {
    # Rodinia
    "backprop": ["128", "256", "512", "1024"],
    "bfs": ["/fast_data/echung67/rodinia-data/bfs/graph1k.txt",
            "/fast_data/echung67/rodinia-data/bfs/graph2k.txt",
            "/fast_data/echung67/rodinia-data/bfs/graph4k.txt",
            "/fast_data/echung67/rodinia-data/bfs/graph8k.txt",
            "/fast_data/echung67/rodinia-data/bfs/graph16k.txt",
            "/fast_data/echung67/rodinia-data/bfs/graph32k.txt"],
    "dwt2d": ["/fast_data/echung67/rodinia-data/dwt2d/192.bmp -d 192x192 -f -5 -l 3",
              "/fast_data/echung67/rodinia-data/dwt2d/rgb.bmp -d 1024x1024 -f -5 -l 3"],
    "euler3d": ["/fast_data/echung67/rodinia-data/cfd/fvcorr.domn.097K"],
    "gaussian": ["-f /fast_data/echung67/rodinia-data/gaussian/matrix3.txt",
                 "-f /fast_data/echung67/rodinia-data/gaussian/matrix4.txt",
                 "-f /fast_data/echung67/rodinia-data/gaussian/matrix16.txt"],
    "heartwall": ["/fast_data/echung67/rodinia-data/heartwall/test.avi 10"],
    "hotspot": ["512 512 100 /fast_data/echung67/rodinia-data/hotspot/temp_512 /fast_data/echung67/rodinia-data/hotspot/power_512 none",
                "512 512 1000 /fast_data/echung67/rodinia-data/hotspot/temp_512 /fast_data/echung67/rodinia-data/hotspot/power_512 none",
                "512 2 2 /fast_data/echung67/rodinia-data/hotspot/temp_512 /fast_data/echung67/rodinia-data/hotspot/power_512 none"],
    "lavaMD": ["-boxes1d 1",
               "-boxes1d 2",
               "-boxes1d 3"],
    "lud_cuda": ["-i /fast_data/echung67/rodinia-data/lud/64.dat",
                 "-i /fast_data/echung67/rodinia-data/lud/256.dat",
                 "-i /fast_data/echung67/rodinia-data/lud/512.dat"],
    "needle": ["32 10",
               "64 10",
               "128 10"],
    "nn": ["/fast_data/echung67/rodinia-data/nn/inputGen/list64k.txt -r 30 -lat 30 -lng 90",
           "/fast_data/echung67/rodinia-data/nn/inputGen/list128k.txt -r 30 -lat 30 -lng 90",
           "/fast_data/echung67/rodinia-data/nn/inputGen/list256k.txt -r 30 -lat 30 -lng 90",
           "/fast_data/echung67/rodinia-data/nn/inputGen/list512k.txt -r 30 -lat 30 -lng 90",
           "/fast_data/echung67/rodinia-data/nn/inputGen/list1024k.txt -r 30 -lat 30 -lng 90"],
    "particlefilter_float": ["-x 64 -y 64 -z 5 -np 10"],
    "particlefilter_naive": ["-x 128 -y 128 -z 10 -np 1000"],
    "pathfinder": ["1024 20 10",
                   "1024 20 50",
                   "1024 20 100"],
    "sc_gpu": ["2 5 4 16 16 32 none none 1",
               "3 3 4 16 16 4 none none 1",
               "10 20 16 64 16 100 none none 1"],
    "srad_v1": ["3 0.5 64 64",
                "6 0.5 64 64",
                "10 0.5 64 64"],
    "srad_v2": ["64 64 0 32 0 32 0.5 10"],
    
    # GraphBig
    # "graphbig_bfs_topo_atomic": [""],

    # Gunrock


    # ETC
    "vectoradd": ["4096", "16384", "65536"],
    "vectormultadd": ["4096", "16384", "65536"]
  }

  benchmark_subdir = {
    # Rodinia
    "backprop": ["128", "256", "512", "1024"],
    "bfs": ["graph1k", "graph2k", "graph4k", "graph8k", "graph16k", "graph32k"],
    "dwt2d": ["192", "1024"],
    "euler3d": ["fvcorr.domn.097K"],
    "gaussian": ["matrix3", "matrix4", "matrix16"],
    "heartwall": ["frames10"],
    "hotspot": ["r512h512i100", "r512h512i1000", "r512h2i2"],
    "lavaMD": ["1", "2", "3"],
    "lud_cuda": ["64", "256", "512"],
    "needle": ["32", "64", "128"],
    "nn": ["64k", "128k", "256k", "512k", "1024k"],
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
    bench_datasets = benchmark_dataset[bench_name]
    bench_subdirs = benchmark_subdir[bench_name]
    for bench_dataset, bench_subdir in zip(bench_datasets, bench_subdirs):
      # if (bench_name != "bfs" and bench_name != "backprop"): continue
      # create the result directory
      # assume nvbit.py has been run 
      subdir = os.path.join(result_dir, bench_name, bench_subdir)
      os.chdir(subdir)

      for macsim_file in macsim_files:
        os.system(f"cp {macsim_file} {subdir}")
      os.system(f"cp {nvbit_bin} {subdir}")
      os.system(f"cp {compress_bin} {subdir}")

      python_file = os.path.join(subdir, "macsim.py")
      with open(python_file, "w") as f:
        f.write("import os\n\n")
        f.write("with open(\"trace_file_list\", \"w\") as f:\n")
        f.write(f"    f.write(\"1\\n\" + os.path.join(\"{trace_path_base}\", \"{bench_name}\", \"{bench_subdir}\", \"kernel_config.txt\"))\n\n")
        f.write("os.system('./macsim > macsim_result.txt')\n")

      # Execute nvbit python script
      subprocess.Popen(["nohup python3 macsim.py"], shell=True, cwd=subdir)
  return

if __name__ == '__main__':
  main(sys.argv)
