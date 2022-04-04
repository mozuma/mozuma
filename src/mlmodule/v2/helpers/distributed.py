from typing import Callable

import ignite.distributed as idist
import torch
import torch.distributed as dist
from ignite.engine import Engine, Events

from mlmodule.v2.base.predictions import BatchModelPrediction
from mlmodule.v2.torch.modules import _ForwardOutputType


class ResultsCollector:
    """Helper object used to collect results from each process
    after every iteration.
    """

    def __init__(
        self,
        output_transform: Callable[
            [_ForwardOutputType], BatchModelPrediction[torch.Tensor]
        ],
        callback_fn: Callable[[torch.Tensor, BatchModelPrediction[torch.Tensor]], None],
        dst_rank: int = 0,
    ) -> None:
        self.output_transform = output_transform
        self.callback_fn = callback_fn
        self.dst_rank = dst_rank

        self.world_size = idist.get_world_size()
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
        self.callback_fn(indices, predictions)

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
                self.callback_fn(indices, predictions)

        else:
            # Free some memory
            del _gather_indices

            # Note: https://discuss.pytorch.org/t/how-to-delete-pytorch-objects-correctly-from-memory/947
            del _gather_outputs

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_STARTED, self.reset)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.collect)
