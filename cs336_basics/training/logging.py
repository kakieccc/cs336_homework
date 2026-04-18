import csv
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class MetricSnapshot:
    name: str
    value: float
    smoothed: float
    avg: float
    count: int


class SmoothedMetric:
    def __init__(self, window_size: int = 100):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.window: deque[float] = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        self.window.append(value)
        self.total += value
        self.count += 1

    @property
    def smoothed(self) -> float:
        if not self.window:
            return 0.0
        return float(sum(self.window) / len(self.window))

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self.total / self.count)


class TrainLogger:
    def __init__(
        self,
        log_dir: str | os.PathLike,
        run_name: str = "run",
        window_size: int = 100,
        echo: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.window_size = window_size
        self.echo = echo

        self.run_dir = self.log_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        self.meta_path = self.run_dir / "meta.json"

        self.start_time = time.time()
        self._csv_header_written = False
        self._csv_fieldnames: list[str] = []
        self._meters: dict[str, SmoothedMetric] = {}
        self._step_buffer: list[dict[str, float | int | str]] = []
        self._epoch_buffer: dict[str, list[float]] = defaultdict(list)

        meta = {
            "run_name": run_name,
            "created_at_unix": self.start_time,
            "window_size": window_size,
        }
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _to_float(self, value: float | int | torch.Tensor) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("metric tensor must be scalar")
            return float(value.detach().item())
        return float(value)

    def _get_meter(self, metric_name: str) -> SmoothedMetric:
        if metric_name not in self._meters:
            self._meters[metric_name] = SmoothedMetric(window_size=self.window_size)
        return self._meters[metric_name]

    def _write_jsonl(self, row: dict[str, float | int | str]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _write_csv(self, row: dict[str, float | int | str]) -> None:
        current_fields = list(row.keys())
        if (not self._csv_header_written) or (current_fields != self._csv_fieldnames):
            self._csv_fieldnames = current_fields
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
                writer.writeheader()
                for past_row in self._step_buffer:
                    writer.writerow(past_row)
            self._csv_header_written = True
            return

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writerow(row)

    def log_step(
        self,
        step: int,
        metrics: dict[str, float | int | torch.Tensor],
        prefix: str = "train",
    ) -> dict[str, float | int | str]:
        if step < 0:
            raise ValueError("step must be non-negative")

        row: dict[str, float | int | str] = {
            "step": int(step),
            "prefix": prefix,
            "wall_time_s": float(time.time() - self.start_time),
        }

        for metric_name, raw_value in metrics.items():
            value = self._to_float(raw_value)
            meter = self._get_meter(f"{prefix}/{metric_name}")
            meter.update(value)
            row[f"{prefix}/{metric_name}"] = value
            row[f"{prefix}/{metric_name}_smoothed"] = meter.smoothed
            row[f"{prefix}/{metric_name}_avg"] = meter.avg

        self._step_buffer.append(row)
        self._write_jsonl(row)
        self._write_csv(row)

        if self.echo:
            echo_metrics = []
            for metric_name in metrics.keys():
                k = f"{prefix}/{metric_name}_smoothed"
                echo_metrics.append(f"{k}={row[k]:.6f}")
            print(f"[step {step}] " + " ".join(echo_metrics))

        return row

    def add_epoch_metric(self, name: str, value: float | int | torch.Tensor) -> None:
        self._epoch_buffer[name].append(self._to_float(value))

    def flush_epoch(self, epoch: int, prefix: str = "train") -> dict[str, float | int | str]:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")

        row: dict[str, float | int | str] = {
            "step": -1,
            "prefix": f"{prefix}_epoch",
            "epoch": int(epoch),
            "wall_time_s": float(time.time() - self.start_time),
        }

        for name, values in self._epoch_buffer.items():
            if not values:
                continue
            row[f"{prefix}/{name}_epoch_mean"] = float(sum(values) / len(values))
            row[f"{prefix}/{name}_epoch_min"] = float(min(values))
            row[f"{prefix}/{name}_epoch_max"] = float(max(values))

        self._epoch_buffer.clear()
        self._write_jsonl(row)
        self._write_csv(row)
        return row

    def latest(self, metric_name: str) -> MetricSnapshot:
        meter = self._get_meter(metric_name)
        last_value = meter.window[-1] if meter.window else 0.0
        return MetricSnapshot(
            name=metric_name,
            value=float(last_value),
            smoothed=meter.smoothed,
            avg=meter.avg,
            count=meter.count,
        )

    def state_dict(self) -> dict:
        meters_state = {}
        for name, meter in self._meters.items():
            meters_state[name] = {
                "window_size": meter.window_size,
                "window": list(meter.window),
                "total": meter.total,
                "count": meter.count,
            }
        return {
            "start_time": self.start_time,
            "meters": meters_state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.start_time = float(state.get("start_time", self.start_time))
        self._meters.clear()
        for name, meter_state in state.get("meters", {}).items():
            meter = SmoothedMetric(window_size=int(meter_state["window_size"]))
            for value in meter_state.get("window", []):
                meter.window.append(float(value))
            meter.total = float(meter_state.get("total", 0.0))
            meter.count = int(meter_state.get("count", 0))
            self._meters[name] = meter


def create_train_logger(
    log_dir: str | os.PathLike,
    run_name: str = "run",
    window_size: int = 100,
    echo: bool = True,
) -> TrainLogger:
    return TrainLogger(
        log_dir=log_dir,
        run_name=run_name,
        window_size=window_size,
        echo=echo,
    )
