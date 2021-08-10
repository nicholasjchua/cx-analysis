#!/usr/bin/env python

from collections.abc import Mapping
import json
import os
import os.path
import logging
from glob import glob
from typing import List, Tuple, Dict
from typing import Dict
from collections import namedtuple

_Config = namedtuple(
    'Config',
    [
        'source',
        'cm_url',
        'cm_token',
        'p_id',
        'annot',
        'cell_types',
        'expected_n',
        'group_by',
        'annotator_initials',
        'min_cx',
        'log',
        'save_preprocessed',
        'out_dir',
        'restrict',
    ]
)

##################
# The Config object holds information representing experimental configuration,
# common to most code used in this library.

class Config(_Config):

    def cm_access(self) -> Tuple:
        return self.cm_url, self.cm_token, self.p_id


##################
# Class Methods

def parse_cfg_file(path: str="") -> Config:
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

    with open(fp) as cfg_file:
        cfg = json.load(cfg_file)

    ##### CHECK CONFIGS #####
    ### URL TO CATMAID SERVER ###
    if cfg['cm_url'] == "":
        cm_url = os.environ['CM_URL']
    else:
        cm_url = cfg['cm_url']
        print("WARNING: Using Catmaid URL from config file. Remember to gitignore this if private.")
        
    ### USER ACCESS TOKEN FOR CATMAID ###
    if cfg['cm_token'] == "":
        cm_token = os.environ['CM_TOKEN']
    else:
        cm_token = cfg['cm_token']
        print("WARNING: User access token found in config file. Remember to gitignore this if private.")
    
    ### ANNOTATION ASSOCIATED WITH THE NEURONS YOU WANT TO QUERY ###
    if type(cfg['annot']) != str:
        raise Exception("A valid Catmaid annotation is required to fetch desired skeletons")
    else:
        annot = cfg['annot']
    
    ### CATMAID PROJECT ID ###
    if type(cfg['p_id']) != int:
        raise Exception("Project ID is an integer")
    else:
        p_id = cfg["p_id"]

    ### SAVE PREPROCESSED DATA ? ###
    if cfg['save_preprocessed'] is False:
        save_preprocessed = False
        out_dir = ""
        raise Warning("Data will not be saved based on options listed in config file")
    else:
        save_preprocessed = True
        # If no output dir given, save data at location of cfg file
        if cfg['out_dir'] == "":
            out_dir = path
        else:
            out_dir = cfg['out_dir']
    
    ### LOG (NOT YET IMPLEMENTED) ###
    if cfg['log'] is False:
        log = False
        raise Warning("Log will not be kept based on options listed in config file")
    else:
        log = True
    
    ### CELLTYPE ANNOTATIONS ###
    # Each skeleton queried from CATMAID must be annotated with only one of these labels
    if type(cfg["cell_types"]) != list:
        raise Exception("Celltype categories need to be strings inside a list")
    else:
        cell_types = cfg['cell_types']
    
    ### EXPECTED NUMBER OF EACH CATEGORY ###
    if len(cell_types) != len(cfg["expected_n"]):
        raise Exception("expected_n should be a the same length as subtypes")
    else:
        expected_n = cfg['expected_n']

    ### GROUPING ###
    # ommatidia position on the hex grid
    if cfg['group_by'] == 'om':
        group_by = 'om'
        annotator_initials = []  # don't need this if grouping by ommatidia
    elif cfg['group_by'] == 'annotator':
        if len(cfg['annotator_initials']) > 1:
            group_by = 'annotator'
            annotator_initials = cfg['annotator_initials']  # for validation experiment
        else:
            raise Exception("Annotator initials missing from config file (required when group_by=annotator)")
    else:
        raise Exception("Invalid 'group_by' argument. Needs to be 'annotator' or 'om'. Former also requires"
                        "a list of annotator initials in the config file")
    
    ### RESTRICT ANALYSIS ON A TAGGED SEGMENT OF THE SKELETON ###
    # For a list of specified cell_types, restrict analysis to the segment contained between their root node and
    # the specified restrict_tag
    restrict = cfg['restrict_skeletons']

    return Config(
        source=fp,
        out_dir=out_dir,
        save_preprocessed=save_preprocessed,
        cm_url=cm_url,
        cm_token=cm_token,
        p_id=p_id,
        annot=annot,
        cell_types=cell_types,
        expected_n=expected_n,
        group_by=group_by,
        annotator_initials=annotator_initials,
        log=log,
        restrict=restrict,
    )

