import dataclasses
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dargparser import dArg, dargparse
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import (
    LightningEnvironment,
    SLURMEnvironment,
)
from lightning.pytorch.strategies import DDPStrategy
from loguru import logger
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

import wandb
from dlib.frameworks.lightning import CUDAMetricsCallback
from dlib.frameworks.pytorch import get_rank, set_torch_file_sharing_strategy_to_system
from dlib.frameworks.wandb import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    WandbCleanupDiskAndCloudSpaceCallback,
    check_checkpoint_path_for_wandb,
    check_for_wandb_checkpoint_and_download_if_necessary,
)
from src.data_loading import LMDataModule, SyntheticDataset, SyntheticDataModule, produce_example
from src.helpers import (
    choose_auto_accelerator,
    choose_auto_devices,
    handle_batch_size_logic_,
    log_slurm_info,
)
from src.model import BasicLM
from src.benchmarks import SamplesPerSecondBenchmark, GpuMetricsBenchmark


@dataclass
class TrainingArgs:
    model_name_or_path: str = dArg(
        default="roberta-base",
        help="HuggingFace model identifier. This is used to construct the model architecture and load pretrained weights if not specified otherwise.",  # noqa: E501
        aliases="--model",
    )
    language_modeling_strategy: Literal["mlm", "clm"] = dArg(
        default="mlm",
        help="Whether to train a masked language model or a causal language model.",
    )
    resume_training: bool = dArg(
        default=False,
        help="Whether to resume training from checkpoint or only load the weights. If true, `--checkpoint_path` must be specified.",
        aliases="--resume",
    )
    checkpoint_path: str | None = dArg(
        default=None,
        help="Path to a saved pytorch-lightning checkpoint. Use the wandb:<wandb-run-id> syntax to load a checkpoint from W&B.",  # noqa: E501
        aliases="--checkpoint",
    )
    tokenizer_path: str | None = dArg(
        default=None,
        help="Path to a directory containing a saved Huggingface PreTrainedTokenizer.",
        aliases="--tokenizer",
    )
    data_dir: str = dArg(
        default="./data",
        help="Path to the data directory. By default, expects a train.txt and dev.txt file inside the directory.",  # noqa: E501
        aliases="-d",
    )
    train_file: str = dArg(default="train.txt")
    dev_file: str = dArg(default="dev.txt")
    line_by_line: bool = dArg(
        default=False, help="Process dataset line by line instead of chunking."
    )
    language: str = dArg(
        default=None,
        help="If specified, the data is expected to lie inside a subdirectory with this name.",
        aliases=["--lang", "--lg", "-l"],
    )
    max_sequence_length: int = dArg(
        default=512,
        help="Sequence length for dataset tokenization.",
        aliases=["--seq_len", "--block_size"],
    )
    overwrite_data_cache: bool = dArg(
        default=False, help="Overwrite the cached preprocessed datasets or not.", aliases="--odc"
    )
    conserve_disk_space: bool = dArg(
        default=False, help="Cleanup cache files whenever possible to save disk space."
    )
    data_preprocessing_only: bool = dArg(
        default=False, help="Exit the script after data preprocessing. Do not start training."
    )

    ####### Hardware ###########
    accelerator: Literal["cuda", "cpu", "tpu", "mps", "auto"] = dArg(
        default="auto",
        help='Hardware accelerator to use. If "auto", will auto-detect available hardware accelerator.',
    )
    distributed_strategy: Literal[
        "ddp", "fsdp", "ddp_smart", "ddp_spawn", "ddp_fork", "auto"
    ] = dArg(
        default="auto",
        help="Distributed training strategy to use. If `auto`, will select automatically (no distributed strategy is used when using a single device).",
        aliases="--ds",
    )
    num_devices: int = dArg(
        default=-1,
        aliases=["--devices", "--nd"],
        help="Number of devices to use for distributed training. If -1, will use all available devices (CUDA) or an accelerator-specific default.",
    )
    cuda_device_ids: list[int] = dArg(
        default=[],
        aliases="--gpu_ids",
        help="Specific CUDA devices (selected by specified indices) to use. Overwrites `--num_devices`. Requires CUDA on the host system.",
    )
    workers: int = dArg(
        default=4,
        help="Number of workers for dataloaders. *Every device* will use that many workers.",
        aliases="-w",
    )
    preprocessing_workers: int = dArg(
        default=-1,
        help="Number of workers for preprocessing the datasets. If -1, use all available CPUs.",
        aliases="--pw",
    )
    precision: Literal["16-mixed", "bf16-mixed", 32] = dArg(
        default=32,
        help="Floating point precision to use during training. Might require specific hardware.",
    )
    precision_type: Literal["medium", "high", "highest"] = dArg(
        default="high",
        help="internal precision of float32 matrix multiplications"
    )
    compile: bool = dArg(
        default=False,
        help="Whether to compile the model with `torch.compile`. Requires torch>=2.0",
    )

    ####### General training ###########
    training_goal: int = dArg(
        default=10_000_000, help="Number training goal units to train for.", aliases="--tg"
    )
    training_goal_unit: Literal["samples", "tokens", "optimizer-steps"] = dArg(
        default="samples", help="Unit of training_goal."
    )
    val_frequency: float = dArg(
        default=0.05,
        help="Period in training goal units between two validations. If <1, compute as fraction of training_goal",
        aliases="--vfq",
    )
    model_log_frequency: float = dArg(
        default=0.1,
        help="Period in training goal units between two model checkpoints. If <1, compute as fraction of training_goal",
        aliases="--mfq",
    )
    val_before_training: bool = dArg(default=True, help="Run one validation epoch before training.")
    val_only: bool = dArg(default=False, help="Run one validation epoch before training.")
    batch_size_per_device: int = dArg(
        default=42,
        help="Batch size per device. If effective_batch_size is specified, this is the maximum batch size per device (you should then increase this in powers of two until you get CUDA OOM errors).",  # noqa: E501
        aliases="-b",
    )
    effective_batch_size: int | None = dArg(
        default=None,
        help="If set, try to auto-infer batch_size_per_device and gradient_accumulation_steps based on number of devices given by --num_devices.",  # noqa: E501
        aliases=["--eb"],
    )
    learning_rate: float = dArg(default=5e-5, aliases="--lr")
    lr_warmup: float = dArg(
        default=0.1,
        help="Number of training goal units to do a learning rate warmup. If <1, compute as fraction of training_goal.",  # noqa: E501
    )
    lr_schedule: Literal[
        "cosine", "linear", "reduce_on_plateau", "constant", "cosine_with_restarts", "polynomial"
    ] = dArg(default="cosine", help="Learning rate schedule.")
    weight_decay: float = dArg(default=0.0, aliases="--wd")
    gradient_clipping: float | None = dArg(default=None, aliases="--gc")
    gradient_accumulation_steps: int = dArg(default=1, aliases=["--gas", "--accum"])
    train_only_embeddings: bool = dArg(
        default=False,
        help="Train only the embedding layer of the model and keep the other transformer layers frozen.",  # noqa: E501
        aliases="--only_embeddings",
    )
    from_scratch: bool = dArg(
        default=False, help="Do not use pre-trained weights to intialize the model."
    )
    from_scratch_embeddings: bool = dArg(
        default=False, help="Do not use pre-trained weights to intialize the token embeddings."
    )

    def __post_init__(self):
        if self.val_frequency < 1:
            self.val_frequency = int(self.training_goal * self.val_frequency)
        if self.model_log_frequency < 1:
            self.model_log_frequency = int(self.training_goal * self.model_log_frequency)
        if self.lr_warmup < 1:
            self.lr_warmup = int(self.training_goal * self.lr_warmup)
        if self.cuda_device_ids:
            if self.num_devices != -1:
                logger.warning(
                    f"Overwriting --num_devices={self.num_devices} with {len(self.cuda_device_ids)} because of --cuda_device_ids={self.cuda_device_ids}"
                )
            self.num_devices = len(self.cuda_device_ids)
        if self.preprocessing_workers == -1:
            # Set to all available CPUs, handle SLURM case when only some CPUs are available to the job
            self.preprocessing_workers = int(
                os.environ.get("SLURM_JOB_CPUS_PER_NODE", multiprocessing.cpu_count())
            )


@dataclass
class MiscArgs:
    seed: int | None = None
    force_deterministic: bool = dArg(
        default=False, help="Force PyTorch operations to be deterministic."
    )
    offline: bool = dArg(default=False, help="Disable W&B online syncing.")
    fast_dev_run: bool = dArg(
        default=False, help="Do fast run through training and validation with reduced sizes."
    )
    wandb_run_name: str | None = dArg(
        default=None, help="Run name for the W&B online UI.", aliases="-n"
    )
    wandb_tags: list[str] = dArg(default=[])
    wandb_project: str = dArg(default=None)
    too_many_open_files_fix: bool = dArg(
        default=False,
        help='Apply fix to circumvent "Too many open files" error caused by the PyTorch Dataloader when using many workers or large batches.',  # noqa: E501
        aliases="--open_files_fix",
    )
    synthetic_data: bool = dArg(
        default=True, help="Decide whether or not to use synthetic data for your experiments. You do not need to specify a specific data_dir when using this option."
    )
    sd_size: int = dArg(
        default=10000000, help="Sice of the synthetic dataset."
    )
    sd_seq_len: int = dArg(
        default = 0, help= "Sequence length of the synthetic data." 
    )
    sd_load_time_mean: float = dArg(
        default=0, help="Mean loading time for the fetching of synthetic data."
    )


@logger.catch(reraise=True)
def main(parsed_arg_groups: tuple[TrainingArgs, MiscArgs]):
    current_process_rank = get_rank()
    args, misc_args = parsed_arg_groups

    ################ Apply fixes ##############
    if misc_args.too_many_open_files_fix:
        logger.info("Setting torch sharing strategy to 'file_system'")
        set_torch_file_sharing_strategy_to_system()

    ############# Seed ##############
    misc_args.seed = seed_everything(workers=True, seed=misc_args.seed)

    ############# Construct W&B Logger ##############
    if misc_args.offline or misc_args.fast_dev_run or args.data_preprocessing_only:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb_extra_args = dict(
        name=misc_args.wandb_run_name,
    )
    if (
        args.checkpoint_path
        and args.resume_training
        and check_checkpoint_path_for_wandb(args.checkpoint_path)
    ):
        logger.info("Resuming training from W&B")
        wandb_extra_args = dict(
            id=check_checkpoint_path_for_wandb(args.checkpoint_path), resume="must"
        )  # resume W&B run
    else:
        args.resume_training = False

    wandb_logger = WandbLogger(
        project=misc_args.wandb_project or WANDB_PROJECT,
        entity=WANDB_ENTITY,
        log_model="all",
        tags=misc_args.wandb_tags,
        **wandb_extra_args,
    )

    ########### define summary metrics for benchmarking ###########
    wandb_logger.experiment.define_metric("SamplesPerSecond", summary="mean")
    wandb_logger.experiment.define_metric("TokensPerSecond", summary="mean")

    ########### Specifiy auto arguments ###########
    if args.accelerator == "auto":
        args.accelerator = choose_auto_accelerator()
    if args.num_devices == -1:
        args.num_devices = choose_auto_devices(args.accelerator)
    if args.cuda_device_ids:
        cuda_device_count = torch.cuda.device_count()
        if cuda_device_count < len(args.cuda_device_ids):
            raise ValueError(
                f"Requested {len(args.cuda_device_ids)} CUDA GPUs but only {cuda_device_count} are available."
            )
    effective_batch_size_per_step = handle_batch_size_logic_(args)

    ########### Log config ###########
    for arg_group in parsed_arg_groups:
        wandb_logger.log_hyperparams(dataclasses.asdict(arg_group))
        if current_process_rank == 0:
            logger.info(arg_group)

    if current_process_rank == 0 and not args.resume_training and not misc_args.offline:
        if misc_args.wandb_run_name is None:
            logger.warning(
                "No run name specified with `--wandb_run_name`. Using W&B default (randomly generated name)."
            )
        else:
            wandb_logger.experiment.name = (
                misc_args.wandb_run_name + "-" + wandb_logger.version
            )  # Append id to name for easier recognition in W&B UI

    IS_ON_SLURM = SLURMEnvironment.detect()
    if IS_ON_SLURM and current_process_rank == 0:
        log_slurm_info()
    ########### Calulate training constants ###########

    if args.training_goal_unit == "samples":
        goal_units_per_optimizer_step = args.effective_batch_size
        goal_units_per_forward_pass = effective_batch_size_per_step
    elif args.training_goal_unit == "tokens":
        goal_units_per_optimizer_step = args.effective_batch_size * args.max_sequence_length
        goal_units_per_forward_pass = effective_batch_size_per_step * args.max_sequence_length
    elif args.training_goal_unit == "optimizer-steps":
        goal_units_per_optimizer_step = 1
        goal_units_per_forward_pass = 1 / args.gradient_accumulation_steps
    else:
        raise ValueError(f"Unknown training goal unit: {args.training_goal_unit}")

    # Lightning does `gradient_accumulation_steps` many forward passes per trainer step (step := optimization step)
    args.training_goal = int(args.training_goal / goal_units_per_optimizer_step)
    val_frequency_in_optimization_steps = int(args.val_frequency / goal_units_per_optimizer_step)

    # val_frequency in lightning is every forward pass NOT optimization step # NOTE: as of June 2023
    args.val_frequency = int(args.val_frequency / goal_units_per_forward_pass)
    args.model_log_frequency = int(args.model_log_frequency / goal_units_per_optimizer_step)
    args.lr_warmup = int(args.lr_warmup / goal_units_per_optimizer_step)

    if misc_args.synthetic_data == True:
        os.environ["TOKENIZERS_PARALLELISM"] = "False" # make sure there won't be any deadlocks when loading data by turning parallel tokenization off

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_name_or_path, use_fast=True
    )

    vocab_size = len(tokenizer)  # NOTE: tokenizer.vocab_size returns size without added vocab
    checkpoint_callback = ModelCheckpoint(
        filename="snap-{step}-samples-{progress/samples}-{progress/tokens}-loss-{val/loss:.2f}",
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
        every_n_train_steps=args.model_log_frequency,
    )
    wandb_disk_cleanup_callback = WandbCleanupDiskAndCloudSpaceCallback(
        cleanup_local=True, cleanup_online=False, size_limit=20
    )

    ################# Construct model ##############
    model_extra_args = dict(
        effective_batch_size_per_step=effective_batch_size_per_step, vocab_size=vocab_size
    )

    # Resume from checkpoint if specified
    if args.checkpoint_path:
        args.checkpoint_path = check_for_wandb_checkpoint_and_download_if_necessary(
            args.checkpoint_path, wandb_logger.experiment
        )

        if args.resume_training:  # load weights, optimizer states, scheduler state, ...\
            model = BasicLM.load_from_checkpoint(
                args.checkpoint_path,
                effective_batch_size_per_step=effective_batch_size_per_step,
            )
            print(model.samples_processed)
        else:  # load only weights
            model = BasicLM(training_args=args, **model_extra_args)
            torch_load = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(torch_load["state_dict"], strict=False)
            model.samples_processed = torch.tensor(0.0)
            model.tokens_processed = torch.tensor(0.0)
    else:
        model = BasicLM(training_args=args, **model_extra_args)

    if args.train_only_embeddings:
        if get_rank() == 0:
            logger.info("Training only embedding layer")
        for param in model.model.parameters():
            param.requires_grad = False
        model.model.get_input_embeddings().weight.requires_grad = True

    if args.from_scratch_embeddings:
        torch.nn.init.xavier_uniform_(model.model.get_input_embeddings().weight)
        # torch.nn.init.normal_(self.model.get_input_embeddings().weight) # alternative

    if current_process_rank == 0:
        model.on_train_start = lambda: logger.info(
            f"Total optimizer steps: {args.training_goal} | "
            f"LR warmup steps: {args.lr_warmup} | "
            f"Validation Frequency: {val_frequency_in_optimization_steps} | "
            f"Model Log Frequencey: {args.model_log_frequency} | "
            f"Effective batch size: {args.effective_batch_size}"
        )
    wandb_logger.watch(model, log="gradients", log_freq=500, log_graph=False)

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision(args.precision_type)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"The current torch version ({torch.__version__}) does not have support for compile."  # noqa: E501
                "Please install torch >= 2.0 or disable compile."
            )
        model = torch.compile(model)

    #################### Construct dataloaders & trainer #################
    if misc_args.synthetic_data == True:
        example = produce_example(tokenizer=tokenizer, sequence_length=(misc_args.sd_seq_len if misc_args.sd_seq_len > 0 else args.max_sequence_length))
        synthetic_dataset = SyntheticDataset(dataset_size=misc_args.sd_size, example=example, load_time=misc_args.sd_load_time_mean)
        dm = SyntheticDataModule(training_args=args, misc_args=misc_args, dataset=synthetic_dataset, tokenizer=tokenizer)
    else:
        dm = LMDataModule(training_args=args, misc_args=misc_args)
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [wandb_disk_cleanup_callback, lr_monitor]
    samplesPerSecondBenchmark = SamplesPerSecondBenchmark(args.max_sequence_length)
    gpuMetricsBenchmark = GpuMetricsBenchmark()
    benchmarks = [samplesPerSecondBenchmark, gpuMetricsBenchmark]
    callbacks.extend(benchmarks)

    if args.accelerator == "cuda":
        callbacks.append(CUDAMetricsCallback())

    # "smart" DDP skipping the find_unused_parameters step - slightly faster
    distributed_strategy = (
        DDPStrategy(find_unused_parameters=False)
        if args.accelerator == "cuda" and args.distributed_strategy == "ddp_smart"
        else args.distributed_strategy
    )

    plugins = None
    if IS_ON_SLURM:
        logger.info("Disabling SLURMEnvironment (we use lightning's native DDP launcher)")
        plugins = [LightningEnvironment()]

    # Initialize trainer
    trainer = Trainer(
        max_steps=args.training_goal,
        check_val_every_n_epoch=None,  # validation based on steps instead of epochs
        devices=args.cuda_device_ids or args.num_devices,
        accelerator=args.accelerator,
        strategy=distributed_strategy,
        logger=wandb_logger,
        deterministic=misc_args.force_deterministic,
        callbacks=callbacks,
        plugins=plugins,
        enable_checkpointing=False,
        precision=args.precision,
        gradient_clip_val=args.gradient_clipping,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        fast_dev_run=misc_args.fast_dev_run,
        inference_mode=not args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other  # noqa: E501
        num_sanity_val_steps=0,
    )
   
    if args.val_before_training and not args.resume_training:
        # TODO: we could use a new trainer with Trainer(devices=1, num_nodes=1) to prevent samples from possibly getting replicated with DistributedSampler here.  # noqa: E501
        logger.info(f"Rank {current_process_rank} | Validation before training...")
        val_result = trainer.validate(model, dm)
        print(val_result)
        if args.val_only:
            exit(0)

    logger.info(f"Rank {current_process_rank} | Starting training...")
    trainer.fit(model, dm, ckpt_path=args.checkpoint_path if args.resume_training else None)
    if trainer.interrupted and IS_ON_SLURM:
        logger.error(
            "Detected keyboard interrupt, not trying to save latest checkpoint right now because we detected SLURM and do not want to drain the node..."
        )
    else:
        if current_process_rank == 0:
            wandb_logger.experiment.summary["Number of Workers"] = args.workers
            wandb_logger.experiment.summary["Compile"] = args.compile
            wandb_logger.experiment.summary["Precision"] = args.precision
            wandb_logger.experiment.summary["Means"] = gpuMetricsBenchmark.compute_means()
            logger.success("Saving finished!")


if __name__ == "__main__":
    parsed_arg_groups = dargparse(dataclasses=(TrainingArgs, MiscArgs))
    main(parsed_arg_groups)
