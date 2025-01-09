import os
from controlllm.configs import TrainConfig


MODULE = 'controlllm' if 'MODULE' not in os.environ else os.environ['MODULE']

LOG_BASE_DIR = os.path.abspath(os.path.join(f"{os.environ.get('MODEL_PATH', TrainConfig().output_dir)}", MODULE)) \
    if 'LOG_BASE_DIR' not in os.environ else os.environ['LOG_BASE_DIR']
