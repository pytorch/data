import logging
import math
import os
from copy import deepcopy
from typing import Any, Callable, List

import torch
from torch.distributed import checkpoint
import torch.distributed.tensor as dtensor
import torch.utils.data as data

from .stateful_dataloader import StatefulDataLoader

"""
The following distributed dataloaders are designed around 3 main principles:

1. Efficient, asynchronous operation. Workers on different devices do not communicate. 
2. Modularity. Data loading pipeline is composed of wrapped iterators, the base iterator 
    loading from disk and additional layers adding levels of post-processing (shuffling, 
    packing, padding, rescaling, etc.).
3. Seamless resumption from checkpoint. Each stage of the pipeline maintains an internal 
    state that can be written/read on disk via implemented recursive `state_dict()` and 
    `load_state_dict()` calls. Any values that should be saved to state can be designated
    'state_params' and will be automatically included in the state dict. States must be
    valid targets of torch.tensor().
4. Rescalability. Users can save and load checkpoints to/from different numbers of workers 
    without losing the global state. This is accomplished by splitting the global state over
    a predefined large number of small partitions, each of which tracks its own individual
    state. Rescaling is accomplished by re-distributing these shards over the physical workers.

Our loaders obey the following type hierarchy: 
torch.data.IterableDataset -> _StatefulDataset -> _WrapperDataset. 
`_StatefulDataset` implements state and checkpointing logic. A `_WrapperDataset` holds a 
single `_StatefulDataset` and iterates via calling its wrapped dataset any number of times, 
then applying some sort of post-processing and yielding the result. Users build data processing 
pipelines by wrapping a base `_StatefulDataset` in any number of `_WrapperDataset` layers, 
which is then passed to the torch DataLoader. 

It is likely that this can be merged into the existing Nodes structure, but we leave this for
future work, for now.
"""


def _shard_partition(itemlist: List[Any], rank: int, worldsize: int) -> List[Any]:
    """
    Partition itemlist into worldsize chunks, grab chunk corresponding to rank and return.
    """
    return itemlist[(rank * len(itemlist)) // worldsize : ((rank + 1) * len(itemlist)) // worldsize]


class _StatefulDataset(data.IterableDataset):
    """
    Stub for stateful datasets, extends data.IterableDataset with state_dict methods.
    All subclasses should specify the params to be considered stateful via self.state_params.
    """

    def __init__(
        self,
        datapath: str,
        rank: int,
        worldsize: int,
    ):
        assert rank >= 0, f"Rank {rank} must be a positive integer"
        assert worldsize > rank, f"Worldsize {worldsize} must be greater than rank {rank}"
        assert datapath is None or (
            os.path.isdir(datapath) and len(os.listdir(datapath)) > 0
        ), f"Data path {datapath} must be a non-empty folder or None"
        self.state_params: List[str] = []

        # Default fields
        self.datapath = datapath
        self.rank = rank
        self.worldsize = worldsize
        self.local_worldsize = -1

        # Setup / loading flags
        self.is_setup = False

    def setup(self):
        """
        This method should contain all setup depending on datapath or rank.
        It is called after init, but immediately before any other operation.
        Certain operations higher up in the pipeline may change rank or datapath
        after init (for example, wrapping in a subdataset sampler layer, or copying
        to worker processes), so all rank- and datapth- dependent ops are deferred to
        this function.
        Currently, this function simply adjusts rank/worldsize to account for
        multiprocess dataloaders.
        """
        if not self.is_setup:
            self.is_setup = True
            # Perform adjustment only if not already adjusted (i.e. via _WrapperDataset)
            if self.local_worldsize == -1:
                info = data.get_worker_info()
                if info is None or info.num_workers == 1:
                    # No multi-worker rank adjustment needed
                    self.local_worldsize = 1
                else:
                    self.local_worldsize = info.num_workers
                    self.worldsize = self.worldsize * self.local_worldsize
                    self.rank = self.local_worldsize * self.rank + info.id

    def statename(self, x: str):
        # Note that this naming convention implicitly disallows repeated layers in the dataset pipeline
        return self.__class__.__name__ + "." + x

    def state_dict(self):
        """
        Retrieve all state_params (each worker/process produces its own state dict shard).
        On the off chance that you're saving a checkpoint with zero steps, run setup first.
        """
        self.setup()
        return {self.statename(flag): getattr(self, flag) for flag in self.state_params}

    def load_state_dict(self, state_dict):
        """
        Run setup if needed, and apply all applicable state_params from the state_dict.
        """
        self.setup()
        [setattr(self, flag, state_dict[self.statename(flag)]) for flag in self.state_params]


class _WrapperDataset(_StatefulDataset):
    """
    Stub for nested wrappers of _StatefulDatasets. Extends state fns with recursion.
    Requires a single instantiated sub-dataset (which may be replicated during setup fn).
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
    ):
        self.dataset = dataset
        # Inherit default flags from sub-dataset
        super().__init__(self.dataset.datapath, self.dataset.rank, self.dataset.worldsize)

    def setup(self):
        """
        Datapath/rank/worldsize percolate upwards recursively during initialization, so
        now we project any desired changes downward, also recursively.
        We also project local_worldsize downward to prevent subsequent layers from
        further inflating the rank/worldsize - we only need to account for multiprocessing once!
        Any code overriding this function should still include this functionality.
        """
        if not self.is_setup:
            super().setup()
            self.dataset.datapath = self.datapath
            self.dataset.rank = self.rank
            self.dataset.worldsize = self.worldsize
            self.dataset.local_worldsize = self.local_worldsize
            self.dataset.setup()

    def load_state_dict(self, state_dict):
        """
        Sets all specified flags at the current level, then recurses into wrapped dataset.
        """
        self.setup()
        super().load_state_dict(state_dict)
        self.dataset.load_state_dict(state_dict)

    def state_dict(self):
        """
        Fetches state dict recursively from wrapped layers, then adds specified flags.
        Overlapping flags are overwritten with a warning.
        """
        self.setup()
        out = self.dataset.state_dict()
        state = super().state_dict()
        for flag in self.state_params:
            if flag in out:
                logging.warning(
                    f"Loader {self.rank}: flag {flag} already present in state_dict with value {out[flag]}. "
                    + f"Overwriting with value {state[flag]}"
                )
        out.update(state)
        return out


#### -------------------------    DATASET LAYERS    ------------------------- ####


class PreprocessDataset(_WrapperDataset):
    """
    Wrapper for a _StatefulDataset that applies a specified preprocessing
    or augmentation function to dataset outputs.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset
    aug_fn : function (any -> any)
        The augmentation function to apply to each dataset item.
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
        aug_fn: Callable,
    ):
        super().__init__(dataset)
        self.aug_fn = aug_fn

    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            out = next(dataset)
            yield self.aug_fn(out)


class SamplingDataset(_WrapperDataset):
    """
    A _WrapperDataset implementing percentage-based sampling: weights can be floats, and the
    number of tokens seen from each subdataset will match those weights as closely as possible.
    This is accomplished by maintaining a _StatefulDataset for each subdataset, and tracking
    the number of tokens emitted by each. Whichever loader is furthest from its target will be
    the next to pass a document.
    Relies on eos token to determine document boundaries, so must sit below BufferDataset.
    ...
    Args
    ----
    datapath : str
        Absolute path to the dataset directory. Expects directory to contain subfolders,
        which in turn contain shard files.
    dataset : _StatefulDataset
        Fully instantiated dataset. Cloned across desired subdatasets during setup.
    delimiter_token : Any
        Token used to indicate sequence/document breaks. Type should match data type.
    datasets : list[str] | None
        A list of subdatasets to draw from. If None, draws from all subfolders of datapath.
    weights : list(float) | None
        Weights describing what percent of emitted tokens should come from each subdataset.
        Need not sum to 1. If None, tokens are drawn evenly.
    verbose : bool
        Track setup progress?
    """

    def __init__(
        self,
        datapath: str,
        dataset: _StatefulDataset,
        delimiter_token: Any,
        datasets=None,
        weights=None,
        verbose=False,
    ):
        super().__init__(dataset)
        self.datapath = datapath
        self.delimiter = delimiter_token
        self.verbose = verbose
        self.datasets = (
            datasets
            if datasets is not None
            else [f for f in os.listdir(datapath) if not os.path.isfile(os.path.join(datapath, f)) and "meta" not in f]
        )
        assert len(self.datasets) > 0, "You must specify at least one dataset"

        if weights is not None:
            assert len(weights) == len(
                self.datasets
            ), f"Number of oversample weights {len(weights)} must match number of datasets {len(self.datasets)}"
            for w in weights:
                assert w > 0, f"Sampling rate {w} must be positive"
        self.weights = [1] * len(self.datasets) if weights is None else weights
        self.weights = [w / sum(self.weights) for w in self.weights]

        self.tokens_seen = [0] * len(self.datasets)

        self.current_iterator = -1
        self.state_params = ["tokens_seen", "current_iterator"]

    def setup(self):
        if not self.is_setup:
            _StatefulDataset.setup(self)
            # Build subdataset iterators
            self.data = []
            for i, d in enumerate(self.datasets):
                self.data.append(deepcopy(self.dataset))
                self.data[-1].datapath = os.path.join(self.datapath, d)
                self.data[-1].rank = self.rank
                self.data[-1].worldsize = self.worldsize
                self.data[-1].local_worldsize = self.local_worldsize
                if self.verbose:
                    logging.info(
                        f"Worker {self.rank} assembled subdataset iterator for {d}, {i+1} of {len(self.datasets)}"
                    )
            [d.setup() for d in self.data]

    def __iter__(self):
        self.setup()
        # Grab one doc at a time in random order
        data = [iter(d) for d in self.data]
        while True:
            if self.current_iterator != -1:
                # Finish current document
                out = next(data[self.current_iterator])
                self.tokens_seen[self.current_iterator] += len(out)
                if out[-1] == self.delimiter:
                    self.current_iterator = -1
                yield out
            else:
                # Choose new subdataset to draw from
                # (whichever is currently most underrepresented compared to target rate)
                offset = [
                    self.weights[i] - self.tokens_seen[i] / (sum(self.tokens_seen) + 1e-9)
                    for i in range(len(self.datasets))
                ]
                offset_argmax = max((diff, i) for i, diff in enumerate(offset))[1]
                self.current_iterator = offset_argmax

    def state_dict(self):
        self.setup()
        # Manually add state of all subloaders to self state
        iterator_states = [d.state_dict() for d in self.data]
        assert len(iterator_states) > 0, f"Worker {self.rank} owns no datasets"
        # Flip list[dict[any]] to dict[list[any]]
        prefix = self.statename("states.")
        out = {prefix + k: [d[k] for d in iterator_states] for k in iterator_states[0].keys()}
        out.update(_StatefulDataset.state_dict(self))
        return out

    def load_state_dict(self, state_dict):
        self.setup()
        # Load stats
        _StatefulDataset.load_state_dict(self, state_dict)
        # Load sub-iterator states
        prefix = self.statename("states.")
        # Flip dict[list[any]] to list[dict[any]]
        iterator_states = [
            {k[k.find(prefix) + len(prefix) :]: v[i] for k, v in state_dict.items() if prefix in k}
            for i in range(len(self.data))
        ]
        # Load individual state sub-dicts
        [self.data[i].load_state_dict(iterator_states[i]) for i in range(len(self.data))]


class DummyDataset(_StatefulDataset):
    """
    A dummy base dataset for demo purposes.

    Normally this dataset would be responsible for using rank, datapath and worldsize arguments
    to perform dataset partitioning, and implement repeating iteration over its particular data shard.

    Spits out random sequences of desired vocab size / seq length as lists.
    Places delimiter token at end of each sequence (used by SamplingDataset).
    """

    def __init__(
        self,
        datapath: str,
        rank: int,
        worldsize: int,
        delimiter_token: Any,
        seed: int = 42,
        vocab: int = 100,
        seqlen: int = 64,
    ):
        super().__init__(datapath, rank, worldsize)
        self.vocab = vocab
        self.seqlen = seqlen
        self.delimiter = delimiter_token
        # Ensure different seeds across ranks and datasets, for demo purposes
        self.seed = seed
        self.generator = None
        self.g_state = None
        self.state_params = ["g_state"]

    def setup(self):
        super().setup()
        if self.generator is None:
            self.generator = torch.Generator().manual_seed(self.seed + self.rank + len(self.datapath) * 100)

    def __iter__(self):
        self.setup()
        while True:
            out = torch.rand(self.seqlen, generator=self.generator)
            out = out.mul(self.vocab).int().tolist()
            out[-1] = self.delimiter
            if self.rank==0:
                print(out)
            yield out

    def state_dict(self):
        self.setup()
        # Write generator state manually
        self.g_state = self.generator.get_state().tolist()
        return super().state_dict()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Manually set generator state
        self.generator.set_state(torch.tensor(self.g_state, dtype=torch.uint8))


class ScalableShardDataset(_WrapperDataset):
    """
    A _WrapperDataset implementing rescalability: loading from checkpoint into a different
    number of gpus will nonetheless keep avoiding all data previously seen in the current epoch.
    This is accomplished by maintaining a large number of smaller StatefulDatasets, cloned from the
    original dataset arg with adjusted ranks, which track state individually and reshard over n_gpus.
    Rescaling only works when this layer wraps all other layers that contribute to state_dict.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset. Cloned into logical workers during setup fn.
    n_logical_shards : int
        Total number of logical shards. Must be a multiple of world size.
    verbose : bool
        Track setup progress?
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
        n_logical_shards: int = 2048,
        verbose=False,
    ):
        super().__init__(dataset)
        assert (
            n_logical_shards % self.worldsize == 0
        ), f"World size {self.worldsize} must divide n_logical_shards {n_logical_shards} evenly"
        assert n_logical_shards > 0, f"n_logical_shards {n_logical_shards} must be a positive integer"

        self.total_shards = n_logical_shards
        self.verbose = verbose

        # Fields to be populated during setup / subdataset setup
        self.data: List[_StatefulDataset] = []
        self.logicals_owned: List[int] = []
        self.n_logicals = 0

        # Position "state", used only for maintaining order when n_workers is unchanged
        # For scaling up or down, logical position is meaningless, and reset
        self.current_reader = 0
        self.load_worldsize = self.worldsize

        self.state_params = ["current_reader"]  # self.data states are handled manually

    def setup(self):
        if not self.is_setup:
            _StatefulDataset.setup(self)
            n_logical_shards = self.total_shards
            logicals = list(range(n_logical_shards))
            self.logicals_owned = _shard_partition(logicals, self.rank, self.worldsize)
            self.n_logicals = n_logical_shards // self.worldsize
            assert (
                len(self.logicals_owned) == self.n_logicals
            ), "(world size * num workers) does not divide logical shards evenly"

            # Build logical shards
            for i in range(self.n_logicals):
                self.data.append(deepcopy(self.dataset))
                self.data[-1].worldsize = n_logical_shards
                self.data[-1].rank = self.logicals_owned[i]
                self.data[-1].local_worldsize = 1
                self.data[-1].datapath = self.datapath
                self.data[-1].verbose = self.rank == 0
                if self.verbose:
                    logging.info(
                        f"Worker {self.rank} assembled logical shard {self.logicals_owned[i]}, {i+1} of {self.n_logicals}"
                    )
            [d.setup() for d in self.data]

    def __iter__(self):
        self.setup()
        # Grab one item at a time, iterating over owned logical shards
        data = [iter(d) for d in self.data]
        while True:
            ind = self.current_reader
            # Read doc
            out = next(data[ind])
            # Update state
            self.current_reader = (self.current_reader + 1) % self.n_logicals
            yield out

    def state_dict(self):
        self.setup()
        # Recursive fetch
        logical_shard_states = [d.state_dict() for d in self.data]
        assert len(logical_shard_states) > 0, f"Worker {self.rank} owns no shards???"
        # Flip list[dict[Any]] to dict[list[Any]]
        state_dict = {k: [d[k] for d in logical_shard_states] for k in logical_shard_states[0].keys()}
        state_dict.update(_StatefulDataset.state_dict(self))

        # Convert to tensor form
        out = {}
        for k, v in state_dict.items():
            v = torch.tensor(v)
            if len(v.shape) == 0:
                k = k + ".scalar"
                v = v.unsqueeze(0)
            out[k] = v

        return out

    def load_state_dict(self, state_dict):
        self.setup()

        # Convert back to lists and scalars
        def detorchify(k, v):
            v = v.tolist()
            if ".scalar" in k:
                k = k[:-7]
                v = v[0]
            return k, v

        plain_dict = {}
        for k, v in state_dict.items():
            k, v = detorchify(k, v)
            plain_dict[k] = v
        state_dict = plain_dict

        # Assemble logical shard states
        # TODO: how is this handling non-resharding state_params when resharding???
        _StatefulDataset.load_state_dict(self, state_dict)
        # Remove all non-resharding state
        [state_dict.pop(self.statename(n)) for n in self.state_params]
        # Flip dict[list[any]] to list[dict[any]]
        logical_shard_states = [{k: v[i] for k, v in state_dict.items()} for i in range(self.n_logicals)]

        # Load values
        for i in range(self.n_logicals):
            self.data[i].load_state_dict(logical_shard_states[i])


#### -------------------------    CHECKPOINT FUNCTIONS    ------------------------- ####


def __pop_dstate(state, device_mesh, placements):
    """
    Removes worker states from the StatefulDataLoader state dict, and assembles them
    into a separate list of dicts for distributed checkpointing.
    """
    dstate = state["_snapshot"]["_worker_snapshots"]
    dstate = [dstate[f"worker_{i}"].pop("dataset_state") for i in range(len(dstate))]
    # Flip list[dict[tensor]] to dict[list[tensor]], and concat
    dstate = {k: torch.cat([d[k] for d in dstate], 0) for k in dstate[0]}
    # Construct dtensors from tensors
    dstate = {
        k: dtensor.DTensor.from_local(
            v,
            device_mesh,
            placements,
        )
        for k, v in dstate.items()
    }
    return dstate


def save_distributed_state_dict(
    loader: StatefulDataLoader,
    path: str,
    device_mesh=None,
):
    """
    Retrieves dataloader state dict, and separates worker states from loader state.
    Loader state is not rescalable, and is saved using normal torch.save.
    It is discarded when rescaling.
    Rescalable worker states are compiled into a dtensor across ranks, and saved 
    using pytorch distributed checkpointing.
    """
    rank = loader.dataset.rank
    state = deepcopy(loader.state_dict())
    dstate = __pop_dstate(state, device_mesh, [dtensor.placement_types.Shard(0)])
    # Write distributed state dict
    writer = checkpoint.FileSystemWriter(path)
    checkpoint.save(
        dstate,
        writer,
    )
    # Write nondistributed state dict
    torch.save(state, os.path.join(path, f"__nondist_cp_{rank}.pth"))


def load_distributed_state_dict(
    loader: StatefulDataLoader,
    path: str,
    device_mesh=None,
):
    """
    Retrieves dataloader state dict, and separates worker states from loader state.
    If not rescaling, load saved dataloader state.
    Rescalable worker states are retrieved using pytorch distributed checkpointing.
    States are distributed over workers, and ScalableShardDataset will handle
    partitioning and re-assignment of available states into logical ranks.
    """
    base = loader.state_dict()
    nworkers = base["_snapshot"]["_main_snapshot"]["_num_workers"]
    rank = loader.dataset.rank
    dstate = __pop_dstate(base, device_mesh, [dtensor.placement_types.Shard(0)])  # placements)
    # Read nondistributed state dict
    ckp_ws = 0 if not os.path.exists(path) else len([x for x in os.listdir(path) if "__nondist_cp_" in x])
    # Check that number of loaders matches
    if ckp_ws == loader.dataset.worldsize:
        state = torch.load(os.path.join(path, f"__nondist_cp_{rank}.pth"))
        # Check that number of workers matches
        if nworkers != state["_snapshot"]["_main_snapshot"]["_num_workers"]:
            state = base
    else:
        # On mismatch, discard saved non-reshardable loader state and start fresh
        state = base
    # Read distributed state dict
    reader = checkpoint.FileSystemReader(path)
    checkpoint.load_state_dict(
        dstate,
        reader,
    )
    # Get local tensors from dtensors, and slice over workers
    dstate = {k: v.to_local().chunk(nworkers) for k, v in dstate.items()}
    # Flip dict[list[tensor]] to list[dict[tensor]]
    dstate = [{k: v[i] for k, v in dstate.items()} for i in range(nworkers)]
    # Re-insert worker states into loader state
    for i in range(nworkers):
        state["_snapshot"]["_worker_snapshots"][f"worker_{i}"]["dataset_state"] = dstate[i]
    # Load into loader
    loader.load_state_dict(state)
