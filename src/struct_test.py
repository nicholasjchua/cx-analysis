import os.path
from src.Connectome import*
from config.cfg_parser import parse_config_file

cfg = parse_config_file('~/Data/191127_lamina_configs/')

a = Connectome(cfg)

pprint(a.neuron_names)
pprint(a.skel_data)


