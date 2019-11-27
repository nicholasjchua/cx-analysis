import json
import os
import os.path
from typing import Dict

def parse_config_file(p: str="") -> Dict:
    """
    parse_config_file
    :param p: str (optional), path to a json containing analysis parameters
    :return cfg: Dict
    """
    if p == "":
        p = 'config/default_cfg.json'
    else:
        p = os.path.join(os.path.expanduser(p), 'nc_cfg.json')

    with open(p) as cfg_file:
        cfg = json.load(cfg_file)
    print(os.environ[])
    if cfg['project_url'] is "":
        cfg['project_url'] = os.environ['$CM_URL']
    else:
        raise Warning("Using Catmaid URL from config file. Remember to gitignore this if private.")

    if cfg['user_token'] is "":
        cfg['user_token'] = os.environ['$CM_TOKEN']
    else:
        raise Warning("User access token found in config file. Remember to gitignore this if private.")

    return cfg

