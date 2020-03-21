import os
import numpy as np
import random
from pathlib import Path

import torch

import dataclasses
import yaml
import inspect

import datetime

import logging
from logging import getLogger, FileHandler, StreamHandler, Formatter

def set_seed(seed=42, num_gpu=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if num_gpu > 0:
        torch.cuda.manual_seed_all(seed)

@dataclasses.dataclass
class YamlConfig:
    """https://qiita.com/kzmssk/items/483f25f47e0ed10aa948"""

    def save(self, config_path: Path, file_name: str="config"):
        """ Export config as YAML file """
        assert config_path.parent.exists(), f'directory {config_path.parent} does not exist'

        def convert_dict(data):
            for key, val in data.items():
                if isinstance(val, Path):
                    data[key] = str(val)
                if isinstance(val, dict):
                    data[key] = convert_dict(val)
            return data

        with open(config_path / file_name, 'w', encoding='utf-8') as f:
            yaml.dump(convert_dict(dataclasses.asdict(self)), f, allow_unicode=True)

    @classmethod
    def load(cls, config_path: Path):
        """ Load config from YAML file """
        assert config_path.exists(), f'YAML config {config_path} does not exist'

        def convert_from_dict(parent_cls, data):
            for key, val in data.items():
                child_class = parent_cls.__dataclass_fields__[key].type
                if child_class == Path:
                    data[key] = Path(val)
                if inspect.isclass(child_class) and issubclass(child_class, YamlConfig):
                    data[key] = child_class(**convert_from_dict(child_class, val))
            return data

        with open(config_path) as f:
            config_data = yaml.full_load(f)
            # recursively convert config item to YamlConfig
            config_data = convert_from_dict(cls, config_data)
            return cls(**config_data)


class BaseLogger(object):
    def __init__(self, log_dir=None, exp_ver=None):
        '''
        Parameters
        ==========
        log_dir : ログフォルダのパス
          ex : ./log/日付(yyyy-mm-dd)
        exp_ver : ログフォルダ内の実験フォルダ名
          デフォルト : 時間(hh-MM)

        Examples
        =========
        log_dir = Path(__file__).resolve().parent / 'log' #実行ファイルがあるフォルダまでの絶対パス
        baselogger = BaseLogger()
        logger = baselogger.create_logger()
        
        logger.info("INFOレベルはストリームにもファイルにもloggingされる")
        logger.debug("DEBUGレベルはファイルにだけloggingされる")
        '''
        self.date =  datetime.datetime.now().strftime("%Y-%m-%d")
        self.time = datetime.datetime.now().strftime("%H-%M")
        
        if log_dir is None:
            self.log_dir = Path(os.path.join("./log", self.date))
            
        #実験フォルダがない場合はディレクトリを作成する
        if not self.log_dir.exists():
            self.log_dir.mkdir()
        
        if exp_ver is None:
            self.exp_ver = self.time

        self.exp_dir = Path(os.path.join(self.log_dir / self.exp_ver))

        #実験サブフォルダがない場合はディレクトリを作成する
        if not self.exp_dir.exists():
            self.exp_dir.mkdir()
            
    @property
    def dump_path(self):
        return self.exp_dir
        
    def create_logger(self, logname="exp", sh=True, fh=True):
        '''
        loogerの作成
        '''
        #log_fileのfullpath
        self.logfile_path = Path(self.log_dir / self.exp_ver / logname)

        #ログファイル名
        log_file = Path('{}.log'.format(self.logfile_path)).resolve()

        logger_ = getLogger(logname)
        logger_.setLevel(logging.DEBUG)

        log_formatter = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

        if fh:
            file_handler = FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(log_formatter)
            logger_.addHandler(file_handler)

        if sh:
            stream_handler = StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(log_formatter)
            logger_.addHandler(stream_handler)
            
        return logger_