#!/usr/bin/env python
import argparse
from pprint import pprint
import os.path
from glob import glob
import sys

sys.path.append(os.path.expanduser('~/src/cx-analysis/src'))

from config import parse_cfg_file
from connectome import Connectome
from utils import load_preprocessed_connectome

def main():
    analysis_dir = handle_args().cfg_path
    cfg = parse_cfg_file(analysis_dir)
    if len(glob(os.path.join(analysis_dir, "*preprocessed.pickle"))) == 0:
        C = Connectome(cfg)
        if cfg.save:
            C.save_connectome()

    else:
        print(f"Preprocessed connectome data found at: {analysis_dir}. "
              f"Use an empty directory to preform a new fetch")
        C = load_preprocessed_connectome(cfg.out_dir)
    C.save_linkdf()
    C.save_cxdf()
    '''
    C.save_linkdf(cfg.out_dir)
    C.save_cxdf()
    '''
    print(C.adj_mat[0])

    # Do anything else with C

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
