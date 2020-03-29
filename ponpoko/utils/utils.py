import os
from pathlib import Path
import numpy as np
import random
import torch
import subprocess

def set_seed(seed=42, num_gpu=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if num_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def gpu_info():
    gpu_info = subprocess.check_output('nvidia-smi')
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
        print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, ')
        print('and then re-execute this cell.')
    else:
        print(gpu_info)

def chdir_colab(dir:Path, base=Path("drive/My Drive/Colab Notebooks")):
    #実行ディレクトリをファイルがある場所に変更
    path_len = len(list(Path(Path.cwd()).parents))
    if path_len == 1:
        from google.colab import drive
        drive.mount('/content/drive')
        os.chdir(Path(base)/Path(dir))

    print(Path.cwd())