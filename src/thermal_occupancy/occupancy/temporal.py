from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class _TrackState:
    present_streak: int = 0
    absent_streak: int = 0
    occupied: bool = False
    last_seen_frame: int = -1


class TemporalOccupancyStabilizer:
    """Per-track entry/exit debounce expressed in video time."""

    def __init__(
        self,
        *,
        fps: float,
        enter_confirm_sec: float,
        exit_confirm_sec: float,
        state_retention_sec: float,
    ) -> None:
        self.enter_frames = max(1, round(float(enter_confirm_sec) * fps))
        self.exit_frames = max(1, round(float(exit_confirm_sec) * fps))
        self.retention_frames = max(
            self.exit_frames, round(float(state_retention_sec) * fps)
        )
        self._states: dict[int, _TrackState] = {}

    def update(
        self, frame_index: int, inside_track_ids: Iterable[int]
    ) -> tuple[int, ...]:
        inside_ids = {int(track_id) for track_id in inside_track_ids}
        all_ids = set(self._states) | inside_ids

        for track_id in all_ids:
            state = self._states.setdefault(track_id, _TrackState())
            if track_id in inside_ids:
                state.present_streak += 1
                state.absent_streak = 0
                state.last_seen_frame = frame_index
                if state.present_streak >= self.enter_frames:
                    state.occupied = True
            else:
                state.present_streak = 0
                state.absent_streak += 1
                if state.absent_streak >= self.exit_frames:
                    state.occupied = False

        stale_ids = [
            track_id
            for track_id, state in self._states.items()
            if state.last_seen_frame >= 0
            and frame_index - state.last_seen_frame > self.retention_frames
        ]
        for track_id in stale_ids:
            del self._states[track_id]

        return tuple(
            sorted(
                track_id
                for track_id, state in self._states.items()
                if state.occupied
            )
        )

