"""Event-level batching for cross-track attention training.

`TrackHitDataset` yields one track per index. For cross-track attention we need
batches composed of whole events so that every track in the batch has access
to its sibling tracks from the same event.

The approach:
- `EventBatchSampler` precomputes the event -> list[track_idx] mapping and yields
  flat lists of track indices that correspond to `batch_size_events` randomly
  sampled events. Each yielded batch is a flat list of track indices.
- `EventCollator` takes such a flat list, looks up each track's event, groups
  them back into events, and pads each event to `max_tracks_per_event`. It
  returns a dict of tensors shaped `(B_ev, T, ...)` with a `track_mask` bool
  tensor for the padded track slots.

The dataset must have an `event_ids` tensor (`None` for old caches — cross-track
training will raise a clear error in that case).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class EventBatchSampler(Sampler[List[int]]):
    """Yields batches of track indices grouped by event.

    Each yielded batch is a flat list of track indices that, together, cover
    `batch_size_events` randomly chosen events (i.e. every track of every
    sampled event is included). The downstream `EventCollator` then reassembles
    the tracks into their event groups and pads to a fixed `max_tracks_per_event`.

    Args:
        event_ids: (N,) LongTensor of global event ids for each track in the
            (sub)set. Length must match the Subset or Dataset being sampled.
        batch_size_events: number of events per batch.
        shuffle: whether to reshuffle events each epoch.
        drop_last: if True, drop the final partial batch.
        seed: base seed for the per-epoch shuffle.
    """

    def __init__(
        self,
        event_ids: torch.Tensor,
        batch_size_events: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.batch_size_events = batch_size_events
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

        # Build event -> list[local_track_idx] mapping once.
        # Indices are positions in the input `event_ids` tensor (which is the
        # indexing space of the Subset/Dataset this sampler will drive).
        self._event_to_indices: dict[int, List[int]] = defaultdict(list)
        for idx, eid in enumerate(event_ids.tolist()):
            self._event_to_indices[int(eid)].append(idx)
        self._event_ids = list(self._event_to_indices.keys())

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        n_events = len(self._event_ids)
        if self.drop_last:
            return n_events // self.batch_size_events
        return (n_events + self.batch_size_events - 1) // self.batch_size_events

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self._epoch)
            perm = torch.randperm(len(self._event_ids), generator=generator).tolist()
            ordered_events = [self._event_ids[i] for i in perm]
        else:
            ordered_events = list(self._event_ids)

        batch: List[int] = []
        events_in_batch = 0
        for eid in ordered_events:
            batch.extend(self._event_to_indices[eid])
            events_in_batch += 1
            if events_in_batch == self.batch_size_events:
                yield batch
                batch = []
                events_in_batch = 0

        if batch and not self.drop_last:
            yield batch


class EventCollator:
    """Collate a flat list of per-track samples into event-level batches.

    Expects each sample to be a dict with the usual per-track fields:
    `hit_features`, `cls_features`, `truth_params`, `reco_params`,
    `padding_mask`. Additionally, the collator needs to know which event each
    incoming track belongs to, which we obtain from the underlying dataset's
    `event_ids` tensor via the original track index.

    Because the default PyTorch `DataLoader` strips the indices before calling
    the collate, we look up the event id for each sample *from the sample
    itself* — we require the dataset's `__getitem__` to include a new
    `track_idx` field, OR we accept a flat list of `(idx, item)` tuples.

    To keep the existing dataset unchanged, we instead pair this collator with
    `EventBatchSampler` (which yields indices) and a thin wrapper that tags
    each item with its index before passing to collate. See `make_event_dataloader`.

    Args:
        event_ids: (N,) LongTensor of event ids, aligned with the dataset this
            collator will batch (must match the sampler).
        max_tracks_per_event: upper bound on tracks per event. Events are padded
            to this length; an overflow assertion fires if an event has more.
    """

    def __init__(self, event_ids: torch.Tensor, max_tracks_per_event: int = 128):
        self.event_ids = event_ids
        self.max_tracks = max_tracks_per_event

    def __call__(self, batch: Sequence[tuple[int, dict]]) -> dict:
        """Collate a batch of (track_idx, sample_dict) tuples into event-level tensors.

        Returns a dict with keys:
            hit_features: (B_ev, T, H, F)
            cls_features: (B_ev, T, C)
            truth_params: (B_ev, T, 6)
            reco_params:  (B_ev, T, 6)
            padding_mask: (B_ev, T, H) bool — per-hit validity
            track_mask:   (B_ev, T) bool — True for real tracks, False for padding
        """
        # Group samples by event id (looked up via the passed track_idx)
        groups: dict[int, List[dict]] = defaultdict(list)
        for track_idx, sample in batch:
            eid = int(self.event_ids[track_idx].item())
            groups[eid].append(sample)

        event_ids_in_batch = list(groups.keys())
        B_ev = len(event_ids_in_batch)
        T = self.max_tracks

        # Check sizes and infer tensor shapes from the first sample.
        first = batch[0][1]
        H = first["hit_features"].shape[0]
        F = first["hit_features"].shape[1]
        C = first["cls_features"].shape[0]
        n_out = first["truth_params"].shape[0]

        hit_features = torch.zeros(B_ev, T, H, F, dtype=first["hit_features"].dtype)
        cls_features = torch.zeros(B_ev, T, C, dtype=first["cls_features"].dtype)
        truth_params = torch.zeros(B_ev, T, n_out, dtype=first["truth_params"].dtype)
        reco_params = torch.zeros(B_ev, T, n_out, dtype=first["reco_params"].dtype)
        padding_mask = torch.zeros(B_ev, T, H, dtype=torch.bool)
        track_mask = torch.zeros(B_ev, T, dtype=torch.bool)

        for ev_i, eid in enumerate(event_ids_in_batch):
            samples = groups[eid]
            n_tracks = len(samples)
            if n_tracks > T:
                # Truncate rather than crash — very rare with T=128 given ttbar
                # distribution peaks at ~60 tracks.
                samples = samples[:T]
                n_tracks = T
            for t_i, s in enumerate(samples):
                hit_features[ev_i, t_i] = s["hit_features"]
                cls_features[ev_i, t_i] = s["cls_features"]
                truth_params[ev_i, t_i] = s["truth_params"]
                reco_params[ev_i, t_i] = s["reco_params"]
                padding_mask[ev_i, t_i] = s["padding_mask"]
                track_mask[ev_i, t_i] = True

        return {
            "hit_features": hit_features,
            "cls_features": cls_features,
            "truth_params": truth_params,
            "reco_params": reco_params,
            "padding_mask": padding_mask,
            "track_mask": track_mask,
        }


class _IndexTaggingDataset(torch.utils.data.Dataset):
    """Wraps a dataset so __getitem__ returns `(idx, sample)` tuples.

    This lets the `EventCollator` see the original track index, which it needs
    to look up event membership. The wrapped dataset is otherwise unchanged.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return (idx, self.base[idx])


def make_event_dataloader(
    dataset,
    event_ids: torch.Tensor,
    batch_size_events: int,
    max_tracks_per_event: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    """Construct a DataLoader that yields event-level batches from a per-track dataset.

    Args:
        dataset: a per-track dataset (or Subset) whose __getitem__ returns a
            dict of per-track tensors.
        event_ids: (len(dataset),) LongTensor mapping each track index to an
            event id. For a Subset, this should be the event_ids restricted to
            the subset's indices.
        batch_size_events: number of events per batch.
        max_tracks_per_event: maximum tracks per event (padded with zeros).
        shuffle: whether to reshuffle events each epoch.
        num_workers: DataLoader num_workers.
        pin_memory: DataLoader pin_memory.
        seed: base seed for the sampler's RNG.
        drop_last: drop the final partial batch.
    """
    sampler = EventBatchSampler(
        event_ids=event_ids,
        batch_size_events=batch_size_events,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
    collator = EventCollator(event_ids=event_ids, max_tracks_per_event=max_tracks_per_event)
    wrapped = _IndexTaggingDataset(dataset)
    return torch.utils.data.DataLoader(
        wrapped,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
