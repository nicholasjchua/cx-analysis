import argparse
import os.path

import src.config
from src.connectome import Connectome
from src.utils import load_preprocessed_connectome

def main():
    analysis_dir = handle_args().cfg_path
    cfg = src.config.parse_cfg_file(analysis_dir)

    C = Connectome(cfg)
    if cfg.save:
        C.save_connectome()
        C.save_linkdf(cfg.out_dir)



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