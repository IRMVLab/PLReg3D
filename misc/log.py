import atexit
import os
from datetime import datetime
from dateutil import tz
import argparse
import shutil

log_dir=None
LOG_FOUT=None
inited=False

def get_log_dir():
    return log_dir

def setup_log(args, eval_mode=False):
    global LOG_FOUT,log_dir,inited,start_time
    if inited:
        return
    inited=True
    config = args.config.split('/')[-1].split('.')[0].replace('config_baseline','cb')
    model_config = args.model_config.split('/')[-1].split('.')[0]    
    tz_sh = tz.gettz('Asia/Shanghai')
    now = datetime.now(tz=tz_sh)
    if (not os.path.exists("./tf_logs")):
        os.mkdir("./tf_logs")
    dir = '{}-{}-{}'.format(config, model_config, now.strftime("%m%d-%H%M%S"))
    if eval_mode:
        dir = '{}-eval'.format(dir)
    log_dir = os.path.join("./tf_logs", dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)    
    os.system('rm -r {}'.format(os.path.join("./tf_logs", 'latest')))
    os.system("cd tf_logs && ln -s {} {} && cd ..".format(dir, "latest"))

    start_time= now
    LOG_FOUT = open(os.path.join(log_dir, 'log_train.txt'), 'w')
    log_string('log dir: {}'.format(log_dir))

def log_string(out_str, end = '\n'):
    LOG_FOUT.write(out_str)
    LOG_FOUT.write(end)
    LOG_FOUT.flush()
    print(out_str, end=end, flush=True)

def log_silent(out_str, end = '\n'):
    LOG_FOUT.write(out_str)
    LOG_FOUT.write(end)
    LOG_FOUT.flush()