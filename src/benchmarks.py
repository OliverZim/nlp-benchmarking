# collection of useful benchmarks
from __future__ import annotations

from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import time
# used for nvitop callback
from lightning.pytorch.utilities import rank_zero_only  # pylint: disable=import-error
from lightning.pytorch.utilities.exceptions import (  # pylint: disable=import-error
    MisconfigurationException,
)
from nvitop.api import libnvml
from nvitop.callbacks.utils import get_devices_by_logical_ids, get_gpu_stats
from nvitop import Device, CudaDevice, MiB


class SamplesPerSecondBenchmark(Callback):
    def __init__(self, max_sequence_length, batch_interval=50):
        super().__init__()
        self.batch_interval = batch_interval    # defines the limit of batches for when the metric is computed the next time
        self.start_time = None
        self.num_samples = 0
        self.num_tokens = 0
        self.max_sequence_length = max_sequence_length
        #self.batches_seen = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.num_samples = 0
        self.batch_interval = min(self.batch_interval, len(trainer.train_dataloader)//10)   # set the observation interval to 10% of the batches of one epoch
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # calculate how many samples have been passed so far
        batch_size = batch["input_ids"].size(dim=0)
        sequence_length = batch["input_ids"].size(dim=1)
        self.num_samples += batch_size
        self.num_tokens += batch_size * sequence_length

        # compute and log the samplesPerSecond based on the number of samples
        if (batch_idx+1) % self.batch_interval == 0:  # log every batch_interval batches
            curr_time = time.time()
            elapsed_time = curr_time - self.start_time
            samplesPerSecond = self.num_samples / elapsed_time
            tokensPerSecond = self.num_tokens / elapsed_time
            trainer.logger.experiment.log({"SamplesPerSecond": samplesPerSecond})
            trainer.logger.experiment.log({"TokensPerSecond": tokensPerSecond})


class GpuMetricsBenchmark(Callback):  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        memory_utilization: bool = True,
        gpu_utilization: bool = True,
        intra_step_time: bool = True,
        inter_step_time: bool = True,
        fan_speed: bool = True,
        temperature: bool = True,
        power_usage: bool = True,
        power_relative: bool = True,
    ) -> None:
        super().__init__()

        try:
            libnvml.nvmlInit()
        except libnvml.NVMLError as ex:
            raise MisconfigurationException(
                'Cannot use GpuStatsLogger callback because NVIDIA driver is not installed.',
            ) from ex

        self._memory_utilization = memory_utilization
        self._gpu_utilization = gpu_utilization
        self._intra_step_time = intra_step_time
        self._inter_step_time = inter_step_time
        self._fan_speed = fan_speed
        self._temperature = temperature
        self._power_usage = power_usage
        self._power_relative = power_relative

    def on_train_start(self, trainer, pl_module) -> None:
        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use GpuStatsLogger callback with Trainer that has no logger.',
            )

        if trainer.strategy.root_device.type != 'cuda':
            raise MisconfigurationException(
                f'You are using GpuStatsLogger but are not running on GPU. '
                f'The root device type is {trainer.strategy.root_device.type}.',
            )

        try:
            self._devices = Device.cuda.all()
            self._means = {
            **{"utilization.memory.mean": [0 for i in range(len(self._devices))]},
            **{"utilization.gpu.mean": [0 for i in range(len(self._devices))]},
            **{"memory.used.mean": [0 for i in range(len(self._devices))]},
            **{"memory.free.mean": [0 for i in range(len(self._devices))]},
            **{"fan.speed.mean": [0 for i in range(len(self._devices))]},
            **{"temperature.gpu.mean": [0 for i in range(len(self._devices))]},
            **{"power.used.mean": [0 for i in range(len(self._devices))]},
            **{"power.relative.mean": [0 for i in range(len(self._devices))]},
            }
            self._n_observations = 0

        except (libnvml.NVMLError, RuntimeError) as ex:
            raise ValueError(
                f'Cannot use GpuStatsLogger callback because devices unavailable. '
                f'Received: `gpus={device_ids}`',
            ) from ex

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # pylint: disable=arguments-differ
        if self._intra_step_time:
            self._snap_intra_step_time = time.monotonic()

        logs = self._get_gpu_stats()
        self._n_observations += 1

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # pylint: disable=arguments-differ
        if self._inter_step_time:
            self._snap_inter_step_time = time.monotonic()

        logs = self._get_gpu_stats()
        self._n_observations += 1

        trainer.logger.log_metrics(logs, step=trainer.global_step)
        

    def get_all_gpu_stats(
        self,
        devices: list[Device],
        memory_utilization: bool = True,
        gpu_utilization: bool = True,
        fan_speed: bool = True,
        temperature: bool = True,
        power_usage: bool = True,
        power_relative: bool = True,
        ) -> dict[str, float]:
        """Get the GPU status from NVML queries."""
        stats = {}
        for device_index, device in enumerate(devices):
            prefix = f'gpu_id: {device.cuda_index}'
            if device.cuda_index != device.physical_index:
                prefix += f' (physical index: {device.physical_index})'
            with device.oneshot():
                if memory_utilization or gpu_utilization:
                    utilization = device.utilization_rates()
                    if memory_utilization:
                        memory_util = float(utilization.memory)
                        stats[f"{prefix}/utilization.memory (%)"] = memory_util
                        self._means["utilization.memory.mean"][device_index] += memory_util
                    if gpu_utilization:
                        gpu_util = float(utilization.gpu)
                        stats[f"{prefix}/utilization.gpu (%)"] = gpu_util
                        self._means["utilization.gpu.mean"][device_index] += gpu_util
                if memory_utilization:
                    memory_used = float(device.memory_used()) / MiB
                    memory_free = float(device.memory_free()) / MiB
                    stats[f"{prefix}/memory.used (MiB)"] = memory_used
                    stats[f"{prefix}/memory.free (MiB)"] = memory_free
                    self._means["memory.used.mean"][device_index] += memory_used
                    self._means["memory.free.mean"][device_index] += memory_free
                if fan_speed:
                    fan_speed = float(device.fan_speed())
                    stats[f"{prefix}/fan.speed (%)"] = fan_speed
                    self._means["fan.speed.mean"][device_index] += fan_speed
                if temperature:
                    gpu_temp = float(device.temperature())
                    stats[f"{prefix}/temperature.gpu (C)"] = gpu_temp
                    self._means["temperature.gpu.mean"][device_index] += gpu_temp
                if power_usage:
                    power_used = float(device.power_usage()) / 1000
                    stats[f"{prefix}/power.used (W)"] = power_used
                    self._means["power.used.mean"][device_index] += power_used
                if power_relative:
                    power_relative = float(device.power_usage()/device.power_limit())
                    stats[f"{prefix}/power.relative (%)"] = power_relative
                    self._means["power.relative.mean"][device_index] += power_relative

        return stats

    def _get_gpu_stats(self) -> dict[str, float]:
        """Get the gpu status from NVML queries."""
        return self.get_all_gpu_stats(
            self._devices,
            self._memory_utilization,
            self._gpu_utilization,
            self._fan_speed,
            self._temperature,
            self._power_usage,
            self._power_relative,
        )

    def compute_means(self):
        for key in self._means:
            self._means[key] = [value / self._n_observations for value in self._means[key]]
        return self._means
