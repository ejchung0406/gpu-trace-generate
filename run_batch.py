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


#########################################################################################
# Last modified by: Sam Jijina
# Reason: SLURM support added. Torque support removed.
#########################################################################################


import sys
import os
import argparse
import bench_common
import trace_common


#########################################################################################
# global variables
#########################################################################################
hparch_account = ''
coc_account = ''


#########################################################################################
# sanity check
#########################################################################################
def sanity_check():
  global args


  if not args.suite:
    print('error: suite not defined')
    exit(0)
  
  if not args.dir:
    print('error: dir not defined')
    exit(0)

  if not args.suite in bench_common.SUITES:
    print('error: suite %s does not exist in bench_common.py.' % (args.suite))
    exit(0)

  if not args.bin and os.path.exists('./macsim') == False:
    print('error: binary does not exist')
    exit(0)

  if args.bin and os.path.exists(args.bin) == False:
    print('error: binary does not exist')
    exit(0)

  if not args.param and os.path.exists('./params.in') == False:
    print('error: params.in does not exist')
    exit(0)

  if args.param and os.path.exists(args.param) == False:
    print('error: %s does not exist' % args.param)
    exit(0)

  if not 'SIM_RESULT_DIR' in os.environ:
    print('error: define SIM_RESULT_DIR env variable')
    exit(0)

  if args.remote_run and not args.remote_dir:
    print('error: define remote option and remote dir')
    exit(0)



#########################################################################################
# create an argument parser
#########################################################################################
def process_options():
  parser = argparse.ArgumentParser(description='run_batch.py')
  parser.add_argument('-proc', action='store', default='1', dest='nproc', help='number of processors that this job requires')
  parser.add_argument('-suite', action='store', dest='suite', help='suite to run')
  parser.add_argument('-bin', action='store', dest='bin', help='macsim binary to run')
  parser.add_argument('-param', action='store', dest='param', help='params.in file to run')
  parser.add_argument('-dir', action='store', dest='dir', help='output directory')
  parser.add_argument('-cmd', action='append', nargs='*', dest='cmd', help='additional command')
  parser.add_argument('-disable-gzip', action='store_true', dest='disable_gzip', help='disable output compression')
  parser.add_argument('-remote', action='store_true', dest='remote', help='sushi cluster')
  parser.add_argument('-remote-run', action='store_true', dest='remote_run', help='sushi cluster')
  parser.add_argument('-remote-dir', action='store', dest='remote_dir', help='sushi cluster')
  parser.add_argument('-add', action='store_true', dest='add', help='add')

  return parser


#########################################################################################
# main function
#########################################################################################
def main(argv):
  global args

  # parse arguments
  parser = process_options()
  args = parser.parse_args()
  
  sanity_check()

  current_dir = os.getcwd()

  ## binary
  if not args.bin:
    bin = current_dir + '/macsim'
  else:
    bin = args.bin

  ## config file
  if not args.param:
    param = current_dir + '/params.in'
  else:
    param = args.param

  ## config file
  args.cmd = sum(args.cmd, [])

  ## result dir setup
  result_dir = os.environ['SIM_RESULT_DIR'] + '/' + args.dir
  if os.path.exists(result_dir):
    if not args.add:
      if sys.version_info < (3, 2):
        answer = raw_input('warning: directory %s exists. overwrite? (y/n) ' % (result_dir))
      else:
        answer = input('warning: directory %s exists. overwrite? (y/n) ' % (result_dir))
      if answer == 'y' or answer == 'yes':
        os.system('rm -rf %s' % (result_dir))
      else:
        exit(0)

  if not args.add or not os.path.exists(result_dir):
    os.system('mkdir -p %s' % (result_dir))
  os.system('cp %s %s/macsim' % (bin, result_dir))
  os.system('cp %s %s/params.in' % (param, result_dir))

  coc_machine_name = ''
  hparch_machine_name = ''
  if coc_account == '':
    coc_machine_name = 'hamaguri.cc.gatech.edu'
    hparch_machine_name = 'purpleconeflower.cc.gt.atl.ga.us'
  else:
    assert(hparch_account != '')
    coc_machine_name = '%s@hamaguri.cc.gatech.edu' % coc_account
    hparch_machine_name = '%s@cherry.cc.gt.atl.ga.us' % hparch_account      ## Modified to cherry machine. Cherry is the Slurm Master controller for now.

  if args.remote:
    os.system('ssh %s mkdir -p remote_dir' % coc_machine_name)
    os.system('scp %s %s:~/remote_dir' % (bin, coc_machine_name))
    os.system('scp %s %s:~/remote_dir/params.in' % (param, coc_machine_name))
    cmd = sys.argv
    cmd[0] = 'run_batch.py'
    cmd.remove('-remote')
    if not '-param' in cmd:
      cmd += ['-param', 'remote_dir/params.in']
    else:
      cmd[cmd.index('-param')+1] = 'remote_dir/params.in'

    if not '-cmd' in cmd:
      cmd += ['-cmd', '\"\"%s\"\"' % cmd[cmd.index('-cmd')+1]]
    else:
      cmd[cmd.index('-cmd')+1] = '\"\"%s\"\"' % cmd[cmd.index('-cmd')+1]

    if not '-bin' in cmd:
      cmd += ['-bin', 'remote_dir/macsim']
    else:
      cmd[cmd.index('-bin')+1] = 'remote_dir/macsim'


    cmd += ['-remote-run']
    cmd += ['-remote-dir', result_dir]
    print('ssh %s %s' % (coc_machine_name, ' '.join(cmd)))
    os.system('ssh %s %s' % (coc_machine_name, ' '.join(cmd)))
  
  for ii in range(0, len(args.cmd)):
    args.cmd[ii] = '--%s' % args.cmd[ii]
  args.cmd = ' '.join(args.cmd)

  print(args.cmd)


  for bench in bench_common.SUITES[args.suite]:
    # create the result directory
    if args.remote_run:
      remotedir = args.remote_dir + '/' + bench
    subdir = result_dir + '/' + bench
    os.system('mkdir -p %s' % (subdir))


    # write a script file to submit
    file_name = subdir + '/run.py'
    file = open(file_name, 'w')
    file.write('#!/usr/bin/python\n\n')
    file.write('import os\n')
    file.write('import glob\n')
    file.write('import sys\n\n')
    file.write('ppid = os.getppid()\n')
    file.write('test_dir = \'/tmp/macsim_\' + \'%s_\' + str(ppid) + \'/%s\'\n' % (args.dir, bench))
    file.write('os.chdir(\'%s\')\n' % (subdir))
    file.write('os.system(\'/bin/uname -a\')\n')
    file.write('os.system(\'/bin/mkdir -p \%s\' % (test_dir))\n')
    file.write('os.system(\'/bin/cp ../macsim \%s\' % (test_dir))\n')
    file.write('os.system(\'/bin/cp ../params.in \%s\' % (test_dir))\n')

    # create trace_file_list
    bench = bench[:bench.find('@')]
    bench_list = bench.split('_')
    file.write('\nnum_appl = %d\n' % len(bench_list))
    file.write('appl_list = []\n')
    for ii in range(0, len(bench_list)):
      if args.remote_run:
        file.write('appl_list.append(\'/net/comparch03/hparch%s\')\n' % 
            (trace_common.TRACE_FILE['%s@ref' % (bench_list[ii])]))
      else:
        file.write('appl_list.append(\'%s\')\n' % (trace_common.TRACE_FILE['%s@ref' % (bench_list[ii])]))

    file.write('for ii in range(0, num_appl):\n')
    file.write('  if os.path.exists(\'/trace_local%s\' % appl_list[ii]):\n')
    file.write('    appl_list[ii] = \'/trace_local%s\' % appl_list[ii]\n\n')

    file.write('f = open(\'%s/trace_file_list\' % test_dir, \'w\')\n')
    file.write('f.write(\'%d\\n\' % num_appl)\n')
    file.write('for ii in range(0, num_appl):\n')
    file.write('  f.write(\'%s\\n\' % appl_list[ii])\n')
    file.write('f.close()\n\n')


    file.write('os.chdir(\'%s\' % (test_dir))\n')
    if args.cmd:
      file.write('os.system(\'./macsim %s\')\n' % (args.cmd))
    else:
      file.write('os.system(\'./macsim\')\n')
    if args.disable_gzip == False:
      file.write('os.system(\'/bin/gzip --best *.stat.out.* *.out\')\n')
      file.write('os.system(\'/bin/mv *.gz %s\')\n' % (subdir))
      if args.remote_run:
        file.write('os.system(\'scp %s/*.gz %s/qsub* %s:%s\')\n' % (subdir, subdir, hparch_machine_name, remotedir))
        
    else:
      file.write('os.system(\'/bin/mv *.out %s\')\n' % (subdir))
      if args.remote_run:
        file.write('os.system(\'scp %s/*.out %s/qsub* %s:%s\')\n' % (subdir, subdir, hparch_machine_name, remotedir))
    file.write('os.system(\'/bin/rm -rf %s\' % (test_dir))\n')
    file.close()
    
    # make the script file executable
    os.system('chmod +x %s' % (file_name))
    
    if args.remote:
      continue


    # qsub command
    cmd = []
    cmd += ['qsub']
    cmd += ['run.py']
    cmd += ['-V -m n']
    cmd += ['-o', '%s/qsub.stdout' % (subdir)]
    cmd += ['-e', '%s/qsub.stderr' % (subdir)]
    if not args.remote_run:
      cmd += ['-q', 'pool1']
    cmd += ['-N', '%s_%s' % (args.dir, bench)]
    cmd += ['-l', 'nodes=1:ppn=%s' % (args.nproc)]

    os.chdir('%s' % (subdir))
    os.system('/bin/echo \'%s\' > %s/RUN_CMD' % (' '.join(cmd), subdir))
    os.system('%s | tee %s/JOB_ID' % (' '.join(cmd), subdir))


#########################################################################################
if __name__ == '__main__':
  main(sys.argv)
    
