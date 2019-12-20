import json
import os
import os.path
import logging
from glob import glob
from typing import List, Tuple, Dict
from typing import Dict

class Config:

    def __init__(self, path: str=""):

        self.cm_url, \
        self.cm_token, \
        self.p_id, \
        self.annot, \
        self.subtypes, \
        self.expected_n, \
        self.min_cx, \
        self.save, \
        self.log, \
        self.out_dir, \
        self.restrict \
            = parse_cfg_file(path)

    def __str__(self) -> str:
        s = f"cm_url: {self.cm_url}\n" \
            f"annot: {self.annot}\n" \
            f"min_cx: {self.min_cx}\n" \
            f"save: {self.save}\n" \
            f"log: {self.log}\n" \
            f"out_dir: {self.log}\n" \
            f"subtypes: {self.subtypes}\n" \
            f"expected_n: {self.expected_n}\n" \
            f"restrict: {self.restrict}\n"

        return s

    def cm_access(self) -> Tuple:

        return self.cm_url, self.cm_token, self.p_id


def parse_cfg_file(path: str="") -> Tuple:
    """
    parse_config_file
    :param path: str (optional), path to a json containing analysis parameters
    :return cfg: Dict
    """

    if path == "":  # the default file will cause it to crash rn
        fp = 'config/default_cfg.json'
    elif path.split('.')[-1] == 'json':
        fp = os.path.join(os.path.expanduser(path))
        path = os.path.split(fp)[0]
    else:  # if path to directory given
        fp = glob(os.path.join(os.path.expanduser(path), '*.json'))[0]
    print(fp)

    with open(fp) as cfg_file:
        cfg = json.load(cfg_file)

    # Check that the config file has all the necessary info
    if cfg['cm_url'] is "":
        cm_url = os.environ['CM_URL']
    else:
        cm_url = cfg['cm_url']
        print("WARNING: Using Catmaid URL from config file. Remember to gitignore this if private.")

    if cfg['cm_token'] is "":
        cm_token = os.environ['CM_TOKEN']
    else:
        cm_token = cfg['cm_token']
        print("WARNING: User access token found in config file. Remember to gitignore this if private.")

    if type(cfg['annot']) != str:
        raise Exception("A valid Catmaid annotation is required to fetch desired skeletons")
    else:
        annot = cfg['annot']

    if type(cfg['p_id']) != int:
        raise Exception("Project ID is an integer")
    else:
        p_id = cfg["p_id"]

    if type(cfg['min_cx']) != int:
        raise Exception("Connection count threshold is an integer")
    else:
        min_cx = cfg['min_cx']

    if not (cfg['save'] and cfg['log']):
        save = False
        log = False
        out_dir = ""
        raise Warning("Data and logs will not be saved based on options listed in config file")
    else:
        save = cfg["save"]
        log = cfg["log"]
        if cfg['out_dir'] == "":
            out_dir = path
        else:
            out_dir = cfg['out_dir']

    if type(cfg["subtypes"]) != list:
        raise Exception("Subtype categories need to be strings inside a list")
    else:
        subtypes = cfg["subtypes"]

    if len(subtypes) != len(cfg["expected_n"]):
        raise Exception("expected_n should be a the same length as subtypes")
    else:
        expected_n = cfg['expected_n']

    if cfg['restrict_skeletons']['restrict_tags'] == []:
        restrict = False

    else:
        restrict = cfg['restrict_skeletons']

    return cm_url, cm_token, p_id, annot, subtypes, expected_n, min_cx, save, log, out_dir, restrict

