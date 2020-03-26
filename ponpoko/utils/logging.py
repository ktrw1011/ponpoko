import os
from pathlib import Path
import datetime
import logging
from logging import getLogger, FileHandler, StreamHandler, Formatter

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