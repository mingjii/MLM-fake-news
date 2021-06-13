import torch
import numpy as np
import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Disable cuDNN benchmark for deterministic selection on algorithm.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
