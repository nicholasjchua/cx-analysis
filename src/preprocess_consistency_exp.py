import argparse
import os.path
import src.config
from src.connectome import *

def main():
    fp = handle_args().cfg_path
    cfg = src.config.parse_cfg_file(fp)



    C = Connectome(cfg)








def handle_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("cfg_path", help="Path to .json containing configurations", type=str)
    a = ap.parse_args()

    a.cfg_path = os.path.expanduser(a.cfg_path)
    if not os.path.exists(a.cfg_path):
        raise Exception(f'Directory not found: {a.cfg_path}')
    else:
        return a

main()