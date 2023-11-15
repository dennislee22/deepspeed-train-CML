import subprocess, sys, os
CDSW_APP_PORT=os.environ['CDSW_APP_PORT']
command = "tensorboard --load_fast=false --logdir trainlogs --host 127.0.0.1 --port $CDSW_APP_PORT"
subprocess.run(command, shell = True, executable="/bin/bash")

