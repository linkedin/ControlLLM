import os
import sys
import logging
import portalocker
from controlllm.utils import constants
from controlllm.utils import setup_utils

# from transformers import logging as transformers_logging

# # Set the logger to INFO to ensure logging of training loss
# transformers_logging.set_verbosity_info()

# Set up default logger with logging level to INFO and format, print to both console and log file
# Set up root logger
root = logging.getLogger()
root.setLevel(logging.INFO)

# Determine rank for distributed setups
rank = int(os.environ.get("RANK", 0)) if 'WORLD_SIZE' in os.environ else 0

# Custom formatter to include rank
class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.rank = rank
        return super().format(record)


# Custom file handler to lock file while writing
# Note that This would help in ensuring that unnecessary attempts to write to or interact with the log file are avoided.
# Especially important in a multi-node setup where such interactions might lead to NFS issues like stale file handles.
# Fix issue: e.g. none master rank's code logging.info(msg) still tries to access the nfs file system for log file though the log level is set to ERROR not INFO
class ConditionalFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        if self.stream is None:
            self.stream = self._open()
        if record.levelno >= self.level:
            portalocker.lock(self.stream, portalocker.LOCK_EX)
            try:
                super().emit(record)
            finally:
                portalocker.unlock(self.stream)


formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(rank)s - %(message)s')

# Stream handler setup
shandler = logging.StreamHandler(sys.stdout)
shandler.setLevel(logging.INFO)
shandler.setFormatter(formatter)
root.addHandler(shandler)

# File handler setup
log_dir = os.path.abspath(os.path.join(constants.LOG_BASE_DIR, os.path.dirname(constants.MODULE)))
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f'{os.path.basename(constants.MODULE)}.log')
fhandler = ConditionalFileHandler(log_file_path)
fhandler.setFormatter(formatter)
fhandler.setLevel(logging.DEBUG)
root.addHandler(fhandler)

# Distributed setup utilities
if 'WORLD_SIZE' in os.environ:
    setup_utils.setup_for_distributed(rank == 0)
