from dataclasses import dataclass


@dataclass
class SetupConfig:
    rank: int = 0
    world_size: int = 1
    # avoid duplicate with train_config.local_rank
    local_rank: int = 0
    distributed: bool = True
    dist_backend: str = 'nccl'
    dist_url: str = 'env://'
    device: str = ''  # 'cuda' or 'cpu' or ''(system default)
    # avoid duplicate with train_config.seed
    random_seed: int = 42
