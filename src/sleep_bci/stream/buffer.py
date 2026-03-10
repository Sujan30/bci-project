"""EpochBuffer: accumulates streaming EEG chunks into complete epochs.

Chunks of arbitrary length (e.g. 250 samples at 2.5s) are appended until
a full epoch (e.g. 3000 samples = 30s at 100 Hz) is accumulated, at which
point the complete epoch is yielded for feature extraction and inference.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np


class EpochBuffer:
    """Ring buffer that accumulates EEG chunks and emits complete epochs.

    Args:
        epoch_samples: Number of samples per complete epoch (default 3000 = 30s @ 100 Hz).
        overlap_samples: Samples to retain between epochs for windowed inference (default 0).
    """

    def __init__(self, epoch_samples: int = 3000, overlap_samples: int = 0) -> None:
        if overlap_samples < 0 or overlap_samples >= epoch_samples:
            raise ValueError(
                f"overlap_samples must be in [0, epoch_samples). "
                f"Got overlap_samples={overlap_samples}, epoch_samples={epoch_samples}."
            )
        self.epoch_samples = epoch_samples
        self.overlap_samples = overlap_samples
        self._buffer: list[float] = []
        self._epoch_count: int = 0

    def push(self, chunk: Union[list, np.ndarray]) -> Optional[np.ndarray]:
        """Append a chunk of samples.

        Returns a (epoch_samples,) float32 array when a complete epoch has
        accumulated, or None if more samples are still needed.
        """
        if isinstance(chunk, np.ndarray):
            self._buffer.extend(chunk.ravel().tolist())
        else:
            self._buffer.extend(chunk)

        if len(self._buffer) >= self.epoch_samples:
            epoch = np.array(self._buffer[: self.epoch_samples], dtype=np.float32)
            # Keep overlap for the next epoch window
            self._buffer = self._buffer[self.epoch_samples - self.overlap_samples :]
            self._epoch_count += 1
            return epoch

        return None

    @property
    def epoch_count(self) -> int:
        """Total number of complete epochs emitted so far."""
        return self._epoch_count

    @property
    def buffered_samples(self) -> int:
        """Number of samples currently held in the buffer."""
        return len(self._buffer)

    @property
    def samples_needed(self) -> int:
        """Samples remaining until the next complete epoch."""
        return max(0, self.epoch_samples - len(self._buffer))

    def reset(self) -> None:
        """Clear the buffer and reset epoch count."""
        self._buffer.clear()
        self._epoch_count = 0
