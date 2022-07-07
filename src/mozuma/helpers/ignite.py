import dataclasses
from logging import Logger
from typing import Callable

import ignite.distributed as idist
import torch
import torch.distributed as dist
from ignite.engine import Engine, Events
from torch.utils.data.dataloader import DataLoader

from mozuma.predictions import BatchModelPrediction


@dataclasses.dataclass
class ResultsCollector:
    """Helper object used to collect results from each process
    after every iteration.
    """

    output_transform: Callable[..., BatchModelPrediction[torch.Tensor]]
    callbacks_fn: Callable[[torch.Tensor, BatchModelPrediction[torch.Tensor]], None]
    dst_rank: int = 0
    world_size: int = dataclasses.field(
        default_factory=lambda: idist.get_world_size(),
        init=False,
    )
    _is_reduced: bool = dataclasses.field(default=False, init=False)

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self._is_reduced = False

    def collect(self, engine: Engine) -> None:
        if self.world_size > 1:
            self._collect_dist(engine)
        else:
            self._collect_simple(engine)

    def _collect_simple(self, engine: Engine) -> None:
        # Get results from the inference step
        indices, output = engine.state.output

        # In this case, let DataParallel has already gathered the outputs togheter.
        # Thus, we just need to obtain the predictions from it.
        predictions = self.output_transform(output)

        # Applying callbacks on results
        self.callbacks_fn(indices, predictions)

    def _collect_dist(self, engine: Engine) -> None:
        rank = idist.get_rank()

        # Get results from the inference step
        indices, output = engine.state.output

        # Gather outputs
        if not self._is_reduced:
            # Note: using all_gather because nccl doesn't support gather to a single destination
            _gather_outputs = [None for _ in range(self.world_size)]
            dist.all_gather_object(_gather_outputs, output)

            # Gather indices
            _gather_indices = [None for _ in range(self.world_size)]
            dist.all_gather_object(_gather_indices, indices)

        self._is_reduced = True

        # Do the following work on one rank only
        rank = idist.get_rank()
        if rank == self.dst_rank:
            # For each output, transform it to obtain predictions,
            # than apply the callbacks
            for indices, output in zip(_gather_indices, _gather_outputs):
                predictions = self.output_transform(output)

                # Aplly predictions to callbacks
                self.callbacks_fn(indices, predictions)

        else:
            # Free some memory
            del _gather_indices

            # Note: https://discuss.pytorch.org/t/how-to-delete-pytorch-objects-correctly-from-memory/947
            del _gather_outputs

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_STARTED, self.reset)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.collect)


def register_multi_gpu_runner_logger(
    engine: Engine, data_loader: DataLoader, logger: Logger
):
    # Logs basic messages, just like TorchInferenceRunner
    @idist.utils.one_rank_only()
    def on_start(engine):
        engine.state.n_batches = len(data_loader)

    @idist.utils.one_rank_only()
    def on_itertation_started(engine):
        s = engine.state
        logger.debug(f"Sending batch number: {s.iteration}/{s.n_batches}")

    @idist.utils.one_rank_only()
    def on_itertation_completed(engine):
        s = engine.state
        logger.debug(f"Collecting results: {s.iteration}/{s.n_batches}")

    engine.add_event_handler(Events.EPOCH_STARTED, on_start)
    engine.add_event_handler(Events.ITERATION_STARTED, on_itertation_started)
    engine.add_event_handler(Events.ITERATION_COMPLETED, on_itertation_completed)
