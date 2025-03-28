# this applies monkey patching to multiprocess module to fix the issue with terminating the pool, dataset.map can hang forever without this patch
import controlllm.utils.multiprocess_custom
import os
# don't try to download datasets from internet, it takes too long to fail and fallback to cache
os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ['HF_HUB_OFFLINE'] = "1"
# set the NCCL timeout to 36000 seconds (60 * 10 minutes) to debug nccl
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # Make NCCL operations blocking
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # avoid memory fragmentation following https://github.com/pytorch/pytorch/issues/130330
os.environ["NCCL_TIMEOUT"] = "36000"
# when using ``sharding_strategy=ShardingStrategy.HYBRID_SHARD``
#     with the sharding process group being intra-node and the
#     replication process group being inter-node, setting
#     ``NCCL_CROSS_NIC=1`` can help improve the all-reduce times over
#     the replication process group for some cluster setups.
# os.environ["NCCL_CROSS_NIC"] = "1"
# uncomment the following two to print and debug the NCCL based Distribution Training issues
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_DISABLED"] = "true"  # disable wandb, only enable it when GPU cluster of training is set up with https access
import fire
import warnings
warnings.filterwarnings("ignore")  # suppress unnecessary warnings from torch and tensorflow

# apply triton kernel customizations, it is particularly useful for llama 3 models as vocab size is 13k+
from controlllm.utils.triton_kernels import apply_model_customizations
apply_model_customizations()

from controlllm.utils import setup_utils
setup_utils.apply_custom_load_dataset()

from controlllm.utils.config_utils import Configs
from controlllm.utils.loading_utils import ModelLoader
from controlllm.utils.dataset_utils import DataLoaderWrapper
from controlllm.utils.train_utils import CustomTrainerNative, CustomTrainerTransformers


def main(**kwargs):

    # Load the default configuration parameters and update them with the command line arguments
    configs = Configs(**kwargs)

    # Set it up to enable runing and debugging in cpu, gpu or distributed mode
    setup_utils.init_distributed_mode(configs.setup_config)

    # Load the pre-trained model and tokenizer
    model_loader = ModelLoader(configs)

    # Load and preprocess the datasets
    data_loader = DataLoaderWrapper(configs, model_loader.tokenizer, model_loader.model)

    # Start the training process
    if configs.train_config.trainer == "native":
        # Train the model
        trainer = CustomTrainerNative(
            model=model_loader.model,
            train_dataloader=data_loader.train_dataloader,
            eval_dataloader=data_loader.eval_dataloader,
            tokenizer=model_loader.tokenizer,
            optimizer=model_loader.optimizer,
            lr_scheduler=model_loader.scheduler,
            train_config=configs.train_config,
            fsdp_config=configs.fsdp_config if configs.train_config.enable_fsdp else None,
            local_rank=configs.setup_config.local_rank,
            rank=configs.setup_config.rank,
            wandb_run=setup_utils.setup_wandb(configs.wandb_config, configs.train_config, configs.fsdp_config, configs.setup_config.rank),
            tb_writer=setup_utils.setup_tensorboard(configs.train_config, configs.setup_config.rank)
        )

        # Train the model with step/epoch based eval and save
        trainer.train()

    else:
        # Initialize Trainer with DistributedDataParallel of the datase, rewite the train_sampler to make it distributed
        trainer = CustomTrainerTransformers(
            model=model_loader.model,
            args=configs.train_config,
            train_dataloader=data_loader.train_dataloader,
            eval_dataloader=data_loader.eval_dataloader,
            # deepspeed and fsdp handle optimizer and scheduler differently by overriding `create_optimizer_and_scheduler` method.
            optimizers=(model_loader.optimizer, model_loader.scheduler),
            processing_class=model_loader.tokenizer,
            local_rank=configs.setup_config.local_rank,
            rank=configs.setup_config.rank,
            train_config=configs.train_config,
            # use transformers trainer's default resume_from_checkpoint if the checkpoint folder is not saved by native trainer
            resume_from_checkpoint=model_loader.resume_checkpoint_path if model_loader.resume_from_transformers_checkpoint else None,
            wandb_run=setup_utils.setup_wandb(configs.wandb_config, configs.train_config, configs.fsdp_config, configs.setup_config.rank),
            tb_writer=setup_utils.setup_tensorboard(configs.train_config, configs.setup_config.rank)
        )

        # add callback for saving and evaluation per save_steps and eval_steps
        trainer.add_callback(model_loader.save_model_callback(trainer, save_steps=configs.train_config.save_steps))
        trainer.add_callback(model_loader.eval_callback(trainer, eval_steps=configs.train_config.eval_steps))
        if model_loader.profiler_callback:
            trainer.add_callback(model_loader.profiler_callback())

        # Train the model with step/epoch based eval and save
        trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
