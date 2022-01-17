import os
import logging
import importlib
from argparse import Namespace
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime

from typing import List, Union, Any
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, log_config=None, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['training']['save_dir'])

        exper_name = self.config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        setup_logging(self.log_dir, log_config=log_config)

        # configure logging module
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple) and not isinstance(args, Namespace):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)
        if hasattr(args, 'config') and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))
        log_config = None
        if args.logger:
            log_config = read_json(args.logger)

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(
            args, _get_opt_name(opt.flags)) for opt in options}

        #
        return cls(config, log_config=log_config, resume=resume, modification=modification)

    def init_handle(self, path: Union[str, List[Union[str, int]]], module: Union[str, Any], handle: str, *args, **kwargs):
        """
        Selects `_init_obj` or `_init_ftn` as appropriate.
        Also checks under specified `path` list of sub-keys.
        Does not recursively sets args and kwargs.
        Args:
            path (Union[str, List[Union[str, int]]]): [description]
            module ([type]): [description]
        """
        if isinstance(path, List):
            root = self[path[0]]
            for i in range(1, len(path)):
                root = root[path[i]]
        else:
            root = self[path]
            path = [path]

        ftn_kwargs = dict(
            root['kwargs']) if 'kwargs' in root.keys() else dict()
        assert all([k not in ftn_kwargs for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        ftn_kwargs.update(kwargs)
        ftn_args = list(root['args']) if 'args' in root.keys() else args
        # force rec for now
        rec = True
        if rec:
            for k, v in ftn_kwargs.items():
                if isinstance(v, dict) and 'handle' in v:
                    # recursively init the arg
                    ftn_kwargs[k] = self.init_handle(
                        [*path, 'kwargs', k], v['module'], v['handle'])
            for k in range(len(ftn_args)):
                v = ftn_args[k]
                if isinstance(v, dict) and 'handle' in v:
                    # recursively init the arg
                    ftn_args[k] = self.init_handle(
                        [*path, 'args', k], v['module'], v['handle'])
        ftn_name = root['type']
        ftn_handles = ['obj', 'ftn']
        if isinstance(module, str):
            try:
                module = importlib.import_module(module)
            except ImportError:
                assert False, f"Could not import module {module}"
        assert handle in ftn_handles, f"'{ftn_handle}' handle not recognised"
        if handle == 'obj':
            return getattr(module, ftn_name)(*ftn_args, **ftn_kwargs)
        elif handle == 'ftn':
            # hack: give the function handle the same name as the original
            ftn = getattr(module, ftn_name)
            ftn_n = partial(ftn, *args, **ftn_kwargs)
            ftn_n.__name__ = ftn.__name__
            return ftn_n

    def init_obj(self, path: Union[str, List[Union[str, int]]], module, *args, **kwargs):
        """
        TODO
        """
        return self.init_handle(path, module, 'obj', *args, **kwargs)

    def init_ftn(self, path: Union[str, List[Union[str, int]]], module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        return self.init_handle(path, module, 'ftn', *args, **kwargs)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options


def _update_config(config, modification):
    if modification is None:
        return config
    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
