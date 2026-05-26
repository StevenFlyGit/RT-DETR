""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import copy
import re
import yaml 
from typing import Any, Dict, Optional, List

from .workspace import GLOBAL_CONFIG

__all__ = [
    'load_config', 
    'merge_config', 
    'merge_dict', 
    'parse_cli',
]


INCLUDE_KEY = '__include__'


def load_config(file_path, cfg=dict()):
    """load config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files"

    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)
        if file_cfg is None:
            return {}

    if INCLUDE_KEY in file_cfg:
        base_yamls = list(file_cfg[INCLUDE_KEY])
        for base_yaml in base_yamls:
            if base_yaml.startswith('~'):
                base_yaml = os.path.expanduser(base_yaml)

            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            with open(base_yaml) as f:
                base_cfg = load_config(base_yaml, cfg)
                merge_dict(cfg, base_cfg)

    return merge_dict(cfg, file_cfg)


VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def _get_by_path(cfg: Dict, path: str):
    cur = cfg
    for p in path.split('.'):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def _resolve_once(obj, cfg: Dict):
    if isinstance(obj, str):
        def _repl(m):
            key = m.group(1)
            val = _get_by_path(cfg, key)
            if val is None:
                return m.group(0)
            if isinstance(val, (dict, list)):
                return str(val)
            return str(val)

        return VAR_PATTERN.sub(_repl, obj)

    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = _resolve_once(v, cfg)
        return obj

    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _resolve_once(obj[i], cfg)
        return obj

    return obj


def resolve_references(cfg: Dict):
    # Repeatedly resolve until stable or max iterations reached
    max_iter = 10
    for _ in range(max_iter):
        before = repr(cfg)
        _resolve_once(cfg, cfg)
        after = repr(cfg)
        if before == after:
            break


def merge_dict(dct, another_dct, inplace=True) -> Dict:
    """merge another_dct into dct
    """
    def _merge(dct, another) -> Dict:
        for k in another:
            if (k in dct and isinstance(dct[k], dict) and isinstance(another[k], dict)):
                _merge(dct[k], another[k])
            else:
                dct[k] = another[k]

        return dct
    
    if not inplace:
        dct = copy.deepcopy(dct)
    
    return _merge(dct, another_dct)


def dictify(s: str, v: Any) -> Dict:
    if '.' not in s:
        return {s: v}
    key, rest = s.split('.', 1)
    return {key: dictify(rest, v)}


def parse_cli(nargs: List[str]) -> Dict:
    """
    parse command-line arguments
        convert `a.c=3 b=10` to `{'a': {'c': 3}, 'b': 10}`
    """
    cfg = {}
    if nargs is None or len(nargs) == 0:
        return cfg

    for s in nargs:
        s = s.strip()
        k, v = s.split('=', 1)
        d = dictify(k, yaml.load(v, Loader=yaml.Loader))
        cfg = merge_dict(cfg, d)

    return cfg



def merge_config(cfg, another_cfg=GLOBAL_CONFIG, inplace: bool=False, overwrite: bool=False):
    """
    Merge another_cfg into cfg, return the merged config

    Example:

        cfg1 = load_config('./rtdetrv2_r18vd_6x_coco.yml')
        cfg1 = merge_config(cfg, inplace=True)

        cfg2 = load_config('./rtdetr_r50vd_6x_coco.yml')
        cfg2 = merge_config(cfg2, inplace=True)

        model1 = create(cfg1['model'], cfg1)
        model2 = create(cfg2['model'], cfg2)
    """
    def _merge(dct, another):
        for k in another:
            if k not in dct:
                dct[k] = another[k]
            
            elif isinstance(dct[k], dict) and isinstance(another[k], dict):
                _merge(dct[k], another[k])   
            
            elif overwrite:
                dct[k] = another[k]

        return cfg
    
    if not inplace:
        cfg = copy.deepcopy(cfg)

    return _merge(cfg, another_cfg)
