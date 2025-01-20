import numpy as np
import torch as th
import random


def set_proc_and_seed(seed: int):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
