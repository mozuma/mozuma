import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from mlmodule.types import Metrics, MetricValue

A = TypeVar("A")


@dataclass
class MetricsCollector:
    metrics: Metrics = field(default_factory=dict)

    def track(self, name: str, fun: Callable[..., A], *args, **kwargs) -> A:
        start = time.time()
        ret = fun(*args, **kwargs)
        self.metrics[name] = time.time() - start
        return ret

    def measure(self, name: str) -> "MetricsTimer":
        return MetricsTimer(self, name)

    def add(self, name: str, value: MetricValue) -> None:
        if name in self.metrics:
            warnings.warn(
                f"metric named {name} will overwrite an existing metric ({name}={self.metrics[name]})",
                RuntimeWarning,
            )
        self.metrics[name] = value

    def add_submetrics(self, name: str, sub_metrics: "MetricsCollector"):
        """Merges the sub metrics under the '{name}__' prefix"""
        self.metrics.update({f"{name}__{k}": v for k, v in sub_metrics.metrics.items()})


@dataclass
class MetricsTimer:
    metrics: MetricsCollector
    name: str

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, *_args):
        self.metrics.add(self.name, time.time() - self._start_time)
