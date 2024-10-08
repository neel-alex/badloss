import random

import torch
import numpy as np


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed=seed)
    random.seed(seed)
