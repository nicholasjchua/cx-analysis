import argparse
from src.utils import save_preprocessed_connectome
import os.path
import src.config
from src.connectome import *

def main():
    fp = handle_args().cfg_path
    cfg = src.config.parse_cfg_file(fp)

    C = Connectome(cfg)
    save_preprocessed_connectome(C)

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