# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple

import torch

_WORKER_ID = "worker_id"
_FETCHER_STATE = "fetcher_state"
_FETCHER_ENDED = "fetcher_ended"
_DATASET_STATE = "dataset_state"
_DATASET_ITER_STATE = "dataset_iter_state"


def _flatten(data: Any, key_lineage: Tuple = ()) -> Dict[Tuple, Any]:
    # Always return a dict as the result
    # If data is not a dict or if it is an empty dict, then return a dict with key as key_lineage and data as the value
    # If data is a dict with entries, then iterate through it and flatten the keys
    flat_data = {}
    if isinstance(data, dict) and len(data) > 0:
        for key, value in data.items():
            flat = _flatten(value, key_lineage + (key,))
            flat_data.update(flat)
    else:
        flat_data[key_lineage] = data
    return flat_data


def _unflatten(flat_data: Dict[Tuple, Any]):
    nested_data = {}
    for key, value in flat_data.items():
        # Consider case where key is empty tuple, this is the case where original data was not a dict
        if len(key) == 0:
            return value

        prefix = key[0]
        if len(key) == 1:
            nested_data[prefix] = value
            continue

        suffix = key[1:]
        if prefix not in nested_data:
            nested_data[prefix] = {}
        nested_data[prefix][suffix] = value

    # now go through nested_data and unflatten next level of dicts
    for k, v in nested_data.items():
        if isinstance(v, dict):
            nested_data[k] = _unflatten(v)
    return nested_data


class _Tombstone:
    pass


class _IncrementalState:
    def __init__(self, initial_state: Optional[Dict[str, Any]]):
        self.flat_state = _flatten(initial_state)

    def generate_delta(self, new_state: Dict[str, Any]):
        new_flat_state = _flatten(new_state)
        delta_flat_state = {}
        all_keys = set()
        if self.flat_state:
            all_keys = set(self.flat_state.keys())
        all_keys = all_keys.union(new_flat_state.keys())

        for key in all_keys:
            if self.flat_state is None or key not in self.flat_state:
                # New key, retain it
                delta_flat_state[key] = new_flat_state[key]
                continue

            if key not in new_flat_state:
                # Key deletion, put in a tombstone
                delta_flat_state[key] = _Tombstone()
                continue

            prev_value, new_value = self.flat_state[key], new_flat_state[key]
            try:
                if isinstance(prev_value, torch.Tensor) and isinstance(new_value, torch.Tensor):
                    if torch.equal(prev_value, new_value):
                        continue
                elif prev_value == new_value:
                    continue
            except Exception:
                # Fallback to retaining new key/value
                pass
            delta_flat_state[key] = new_value
        # Update internal state to the new state
        self.flat_state = new_flat_state
        return delta_flat_state

    def apply_delta(self, flat_delta_state: Dict[Tuple, Any]) -> None:
        for key, update in flat_delta_state.items():
            if self.flat_state is None:
                self.flat_state = {}

            if isinstance(update, _Tombstone):
                # Remove key if present in the state
                self.flat_state.pop(key, None)
            else:
                self.flat_state[key] = update

    def get_state(self) -> Optional[Dict[str, Any]]:
        return _unflatten(self.flat_state)


class _IncrementalWorkerState:
    def __init__(self, initial_worker_state_dict: Optional[Dict[str, Any]]):
        self._worker_id = None
        self._fetcher_ended = None

        dataset_state = None
        fetcher_iter_state = None
        if initial_worker_state_dict:
            self._worker_id = initial_worker_state_dict[_WORKER_ID]
            dataset_state = initial_worker_state_dict.get(_DATASET_STATE, None)
            fetcher_state = initial_worker_state_dict.get(_FETCHER_STATE, None)
            if fetcher_state is not None:
                self._fetcher_ended = fetcher_state[_FETCHER_ENDED]
                fetcher_iter_state = fetcher_state.get(_DATASET_ITER_STATE, None)

        self._incr_dataset_state = _IncrementalState(dataset_state)
        self._incr_fetcher_iter_state = _IncrementalState(fetcher_iter_state)

    def generate_delta(self, new_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        assert _WORKER_ID in new_state_dict
        self._worker_id = new_state_dict[_WORKER_ID]
        incr_state_dict = {_WORKER_ID: self._worker_id, _FETCHER_STATE: None}

        ds_state = new_state_dict.get(_DATASET_STATE, None)
        if ds_state is not None:
            incr_state_dict[_DATASET_STATE] = self._incr_dataset_state.generate_delta(ds_state)

        fetcher_state = new_state_dict.get(_FETCHER_STATE, None)
        if fetcher_state is not None:
            self._fetcher_ended = fetcher_state[_FETCHER_ENDED]

            delta_iter_state = None
            iter_state = fetcher_state.get(_DATASET_ITER_STATE, None)
            if iter_state is not None:
                delta_iter_state = self._incr_fetcher_iter_state.generate_delta(iter_state)

            incr_state_dict[_FETCHER_STATE] = {
                _DATASET_ITER_STATE: delta_iter_state,
                _FETCHER_ENDED: self._fetcher_ended,
            }
        return incr_state_dict

    def apply_delta(self, delta_state_dict: Dict[str, Any]) -> None:
        self._worker_id = delta_state_dict[_WORKER_ID]
        ds_state = delta_state_dict.get(_DATASET_STATE, None)
        if ds_state is not None:
            self._incr_dataset_state.apply_delta(ds_state)

        fetcher_state = delta_state_dict.get(_FETCHER_STATE, None)
        if fetcher_state is not None:
            self._fetcher_ended = fetcher_state[_FETCHER_ENDED]
            iter_state = fetcher_state.get(_DATASET_ITER_STATE, None)
            if iter_state is not None:
                self._incr_fetcher_iter_state.apply_delta(iter_state)

    def get_state(self) -> Dict[str, Any]:
        fetcher_state = (
            {
                _FETCHER_ENDED: self._fetcher_ended,
                _DATASET_ITER_STATE: self._incr_fetcher_iter_state.get_state(),
            }
            if self._fetcher_ended is not None
            else None
        )
        return {
            _WORKER_ID: self._worker_id,
            _DATASET_STATE: self._incr_dataset_state.get_state(),
            _FETCHER_STATE: fetcher_state,
        }
