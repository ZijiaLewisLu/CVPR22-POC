from yacs.config import CfgNode
import json
from .default import get_cfg_defaults
import os

def _hparam_helper(cfg: CfgNode) -> dict:
    D = {}
    for k, v in cfg.items():
        # k = capitalize(k)
        if not isinstance(v, CfgNode):
            D[k] = v
        else:
            d = _hparam_helper(v)
            d = { "%s.%s" % (k, k2): v for k2, v in d.items() }
            D.update(d)
            
    return D

def tb_type_convert(x):
    import torch
    t = type(x)
    if t in [int, float, bool, str, torch.Tensor]:
        return x
    else:
        return str(x)

def generate_hparam_dict(cfg: CfgNode, remove_leaf=True, type_convert=True) -> dict:
    D = {}
    for k, v in cfg.items():
        if not isinstance(v, CfgNode):
            if remove_leaf:
                continue
            else:
                D[k] = v
        else:
            d = _hparam_helper(v)
            d = { "%s.%s" % (k, k2): v for k2, v in d.items() }
            D.update(d)

    if type_convert:
        D = { k:tb_type_convert(v) for k, v in D.items() }
            
    return D

def dict2cfg(cfg_dict:dict) -> CfgNode:
    cfg = CfgNode()
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            v = dict2cfg(v)
        cfg[k] = v
    return cfg

def generate_diff_dict(default: CfgNode, cfg: CfgNode, include_missing=False) -> dict :
    """
    include_missing = False
        if a key is missing in cfg,
        it assumes the value matches with that of default
    """

    diff = {}
    for k, v in cfg.items():
        if k not in default and (not include_missing):
            continue
        if isinstance(v, CfgNode):
            subdiff = generate_diff_dict(default[k], cfg[k], include_missing=include_missing)
            if len(subdiff) > 0:
                diff[k] = subdiff
        else:
            if v != default[k]:
                diff[k] = v
    
    return diff

def capitalize(string):
    return string[0].upper() + string[1:]

def diff2expname(diff: dict, remove_leaf=False):
    string = ""
    for k, v in diff.items():
        if isinstance(v, dict):
            v = diff2expname(v, remove_leaf=False) # when recursive call, always false
            string += "%s[%s]-" % (k, v)
        elif not remove_leaf:
            if isinstance(v, bool):
                v = str(v)[0]
            string += "%s:%s-" % (k, v)
    
    string = string[:-1] # remove last dash
    return string

def generate_expgroup_name(default:CfgNode, cfg:CfgNode) -> str:
    """
    use the different params between `default` and `cfg` 
    to generate experiment name
    to help noting down which params are changed.
    """
    exp_name = []

    # 1. add cfg_file setting
    for f in cfg.cfg_file:
        f = os.path.basename(f)
        f = '.'.join(f.split('.')[:-1])
        exp_name.append(f)


    # 2. add different params 
    diff = generate_diff_dict(default, cfg)
    prune = {}
    for k, v in diff.items():
        if not isinstance(v, dict):
            continue # remove leaf args, which are unimportant ones
        if k == 'dataset':
            continue # no need to add dataset information
        prune[capitalize(k)] = v
    diff_string = diff2expname(prune)
    if len(diff_string) > 0:
        exp_name.append(diff_string)
    if len(cfg.mark) > 0: # if there is attentional mark, add it
        exp_name.append(cfg.mark)

    exp_name = '-'.join(exp_name)
    return exp_name


def int2float_check(x, default_value):
    """
    convert the value of a parameter from int to float if its default value is float
    """
    if isinstance(default_value, float) and "." not in x:
        try:
            int(x) # first check if x can convert to int
            x = x + '.0' # if can convert, change to float match str
        except ValueError:
            pass # cannot convert, pass on to cfg to throw error
    return x


def parse_command_set_as_dict(set_cfgs):
    assert len(set_cfgs) / 2 == 0
    d = {}
    for i in range(len(set_cfgs)//2):
        k = set_cfgs[i*2]
        v = set_cfgs[i*2+1]
        d[k] = v
    return d

def setup_cfg(cfg_file=[], set_cfgs=None, runid=-1) -> CfgNode:
    """
    update default cfg according to cmd line input
    and automatic generate experiment name
    """

    cfg = get_cfg_defaults()

    # preprocess set_cfgs to convert int2float
    L = len(set_cfgs)
    default_hparam = generate_hparam_dict(cfg, remove_leaf=False, type_convert=False)
    for i in range(L//2):
        k = set_cfgs[i*2]
        v = set_cfgs[i*2+1]
        tgt = default_hparam[k]
        v = int2float_check(v, tgt)
        set_cfgs[i*2+1] = v

    # update cfg
    for f in cfg_file: # if no config file, this is empty list
        cfg.merge_from_file(f)
    default = cfg.clone()
    if set_cfgs is not None:
        cfg.merge_from_list(set_cfgs)
    cfg.cfg_file = cfg_file
    cfg.set_cfgs = set_cfgs

    # generate experiment name
    ## use dataset to set path 
    path = "%s/Split%d/" % (cfg.dataset.name, cfg.dataset.split)
    cfg.expgroup = generate_expgroup_name(default, cfg)

    # make sure each run saved to a different folder.
    if runid < 0:
        runid = os.getpid()
    cfg.runid = "R%d" % runid
    cfg.logdir = path + cfg.expgroup + "-" + cfg.runid

    cfg.freeze()

    return cfg


def load_cfg_json(cfg_file):

    # cfg_file = os.path.join(folder, 'args.json')
    with open(cfg_file, 'r') as fp:
        cfg_dict = json.load(fp)
    cfg = dict2cfg(cfg_dict)

    default = get_cfg_defaults()
    for f in cfg.cfg_file:
        default.merge_from_file(f)

    expgroup_name = generate_expgroup_name(default, cfg)

    return expgroup_name, cfg