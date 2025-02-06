import logging
import math
import os
import pyarrow as pa
from copy import deepcopy
from typing import Any, Callable, List, Optional, Set

import torch
from torch.distributed import checkpoint
import torch.distributed.tensor as dtensor
import torch.distributed as dist
import torch.utils.data as data

from .stateful_dataloader import StatefulDataLoader

"""
TODO: UPDATE THIS FOR SCALABLEREADER

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


#### -------------------------    FILE READERS    ------------------------- ####


class _ShardFileHandler:
    """
    Stub for shard file readers of different formats.
    Must implement open, length, indexing, and slicing functions.
    """

    def is_legal(self, filepath: str):
        """
        Given a file path, determine if it qualifies for this handler.
        Ideally does not involve opening the file.
        """
        return os.path.isfile(filepath)

    def open(self, path: str):
        """
        Open the file, to be indexed via self.get() method.
        Avoid reading entire multi-Gb files when possible!
        """
        raise NotImplementedError

    def length(self, path: str):
        """
        Calculate the number of documents in the given file.
        Avoid reading entire multi-Gb files when possible!
        """
        raise NotImplementedError

    def get(self, reader, index: int, drop_tokens: Set):
        """
        Given the output of self.open() and an index, return the document at that index.
        Then, remove the first and/or last items if they appear in drop_tokens.
        Try to avoid reading entire documents at a time in case of long documents,
        but this is less important than avoiding reading entire files as above.
        Output must support len() method.
        """
        raise NotImplementedError

    def slice(self, doc, index: int, n_pull: int) -> List:
        """
        Given a long document, retrieve n_pull consecutive items starting from index.
        Again, try to be memory-efficient when doing so, but efficiency in self.get()
        and self.open() is far more important.
        Must return a python list.
        """
        raise NotImplementedError


class ArrowHandler(_ShardFileHandler):
    """
    Reader for indexable, pre-tokenized PyArrow shard files.
    Pyarrow shard files are expected to hold multiple RecordBatches,
    where each RecordBatch has a "tokens" field consisting of
    a single token list (i.e. each document is a single sequence
    under a "token" field, and the file is a list of such sequences).

    A preferred format as we can load document chunks without having to ever pull
    the entire document or shard file, allowing for graceful handling of large documents.
    Non-standard data format, though.
    """

    def __init__(self, col_name: str = "tokens"):
        self.col_name = col_name

    def is_legal(self, filepath: str):
        return "arrow" in os.path.splitext(filepath)[1]

    def open(self, path: str):
        return pa.ipc.open_file(pa.memory_map(path))

    def length(self, path: str):
        return self.open(path).num_record_batches

    def get(self, reader: pa.RecordBatchFileReader, index: int, drop_tokens: Set):
        doc = reader.get_batch(index)[self.col_name]
        if len(doc) > 0 and doc[0].as_py() in drop_tokens:
            doc = doc.slice(1, len(doc) - 1)
        # Recheck len for edge case where doc=[eos]
        if len(doc) > 0 and doc[-1].as_py() in drop_tokens:
            doc = doc.slice(0, len(doc) - 1)
        return doc

    def slice(self, doc: pa.UInt32Array, index: int, n_pull: int) -> List:
        return doc.slice(index, n_pull).to_pylist()
        
        
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


class ScalableReader(_StatefulDataset):
    """
    Maintains shared logical shards but opens them one at a time. Completely repartitions
    unseen shards only when rescaling. 
    """

    def __init__(
        self, 
        datapath: str, 
        rank: int, 
        worldsize: int,
        filehandler: _ShardFileHandler,
        delimiter_token: Any,
        bos_token: Optional[Any] = None,
        strip_tokens: Optional[Set[Any]] = set(),
        seed: int = 42,
        min_length: int = 1,
        max_chunksize: int = 1024,
        n_logical_shards: int = 30720,
        verbose: bool = False,
    ):
        super().__init__(datapath, rank, worldsize)
        self.seed = seed
        self.datapath = datapath
        self.filehandler = filehandler()
        self.min_length = min_length
        assert max_chunksize > 0, f"Max chunksize must be a nonzero positive integer"
        self.chunksize = max_chunksize
        self.eos = delimiter_token
        self.bos = bos_token
        self.drop = strip_tokens
        self.n_logical_shards = n_logical_shards
        self.verbose = verbose
        
        # Position
        self.reader = None
        self.cur_file = None

        # Setup flags
        self.is_setup = False
        self.filesizes = None  # [[filenames], [filesizes]]  CONSTRUCTED PRE ITER IF NOT LOADED
        self.shard_states = None  # shardid, file pos, doc pos, chunk pos, epoch   RESHARD

        # TODO: add handling to prevent zero-length allocations

    def _get_shard_breakdown(self, rank, nshards):
        # Find highest fileid still smaller than start
        sizelist = torch.tensor(self.filesizes[1])
        sizelist = sizelist/sizelist.float().mean()
        cum_sizelist = sizelist.cumsum(0)
        start_frac = rank/nshards*len(sizelist)
        start_id = len(sizelist) - cum_sizelist.gt(start_frac).sum().item()
        # For each doc, assign relevant fractional ownership
        start = start_frac
        end = (rank+1)/nshards*len(sizelist)
        my_files = []  # fileid, start%, end%
        for i, (size, cumsize_incl) in enumerate(
            zip(sizelist[start_id:].tolist(), cum_sizelist[start_id:].tolist())
        ):
            id = start_id + i
            cumsize = cumsize_incl - size
            if cumsize > end:
                break
            elif cumsize <= end and cumsize_incl >= start:
                my_files.append([
                    id,
                    min(max((start - cumsize) / size, 0), 1),
                    min(max((end - cumsize) / size, 0), 1),
                ])
        return my_files

    def setup(self):
        if not self.is_setup:
            # Get your adjusted rank and worldsize
            super().setup()

            # Get logical shard partitions
            my_shards = list(range(
                (self.n_logical_shards * self.rank) // self.worldsize,
                (self.n_logical_shards * (self.rank + 1)) // self.worldsize,
            ))

            # Set up logical shard states (may be overwritten later by ckp load)
            self.shard_states = torch.zeros(math.ceil(self.n_logical_shards / self.worldsize), 5, dtype=torch.int)
            self.shard_states[:len(my_shards), 0] = torch.tensor(my_shards)
            self.shard_states[len(my_shards):, 0] = -1
            self.shard_states[len(my_shards):, 4] = torch.iinfo(torch.int).max

    def _pre_iter(self):
        # Run after loading checkpoint, before iterating

        # Assemble set of available shard files, if nonexistant
        if self.filesizes is None:
            # Find all legal files
            shards = [
                [os.path.join(root,name)[len(self.datapath)+1:], os.path.getsize(os.path.join(root, name))]
                for root, dirs, files in os.walk(self.datapath, topdown=False)
                for name in files
                if self.filehandler.is_legal(os.path.join(root, name))
            ]
            shards.sort()
            # Flip list of (shard,size) tuples into (shardlist,sizelist)
            self.filesizes = list(zip(*shards)) 

    def _get_reader(self, fileid, reader, ndocs):
        """
        If new fileid does not match the current one, open a new reader on
        the corresponding filepath. Also return the number of docs in the file.
        """
        if self.cur_file == fileid:
            return reader, ndocs
        else:
            self.cur_file = fileid
            filepath = os.path.join(self.datapath, self.filesizes[0][fileid])
            return self.filehandler.open(filepath), self.filehandler.length(filepath)

    def _construct_chunk(self, j, doc, n_chunks):
        """
        Grab a chunk of the desired size from the document, with eos/bos handling
        """
        start_index = j * self.chunksize
        n_pull = self.chunksize
        if self.bos is not None:
            if j == 0:
                n_pull -= 1
            else:
                start_index -= 1
        chunk = self.filehandler.slice(doc, start_index, n_pull)
        # Add bos/eos tokens if needed
        if self.bos is not None and j == 0:
            chunk = [self.bos] + chunk
        if j == n_chunks - 1:
            chunk = chunk + [self.eos]
        return chunk
    
    def __iter__(self):
        if not self.is_setup:
            self.setup()
        self._pre_iter()
        reader = None
        ndocs = -1
        while True:
            # Isolate undervisited shards
            epoch_count = self.shard_states[:,4].min().item()
            shardset = self.shard_states[:,4].eq(epoch_count).nonzero().squeeze(-1)
            for i in shardset:
                shardid = self.shard_states[i][0].item()
                files = self._get_shard_breakdown(shardid, self.n_logical_shards)  # list([docid, start%, end%])
                file_offset = self.shard_states[i][1].item()
                for file_pos in range(file_offset, len(files)):
                    # Update position
                    self.shard_states[i][1] = file_pos
                    # Calculate doc range
                    file = files[file_pos]
                    fileid = file[0]
                    reader, ndocs = self._get_reader(fileid, reader, ndocs)
                    doc_start = round(ndocs * file[1])
                    doc_end = round(ndocs * file[2])
                    doc_offset = self.shard_states[i][2].item()
                    for doc_pos in range(doc_offset, doc_end - doc_start):
                        # Update position
                        self.shard_states[i][2] = doc_pos
                        # Fetch doc
                        doc = self.filehandler.get(reader, doc_start + doc_pos, self.drop)
                        doclen = len(doc)
                        nchunks = math.ceil(doclen/self.chunksize)
                        chunk_offset = self.shard_states[i][3].item()
                        for chunk_pos in range(chunk_offset, nchunks):
                            # Update position
                            self.shard_states[i][3] = chunk_pos+1
                            # Yield chunk
                            yield self._construct_chunk(chunk_pos, doc, nchunks)
                        # Reset chunk_pos after finishing doc
                        self.shard_states[i][3] = 0
                    # Reset doc_pos after finishing file
                    self.shard_states[i][2] = 0
                # Reset file_pos after finishing shard
                self.shard_states[i][1] = 0
                # Increase epoch count after finishing shard
                self.shard_states[i][4] += 1
            # Begin new epoch

    def state_dict(self):
        self.setup()
        # Values to save: shard states (shard/repl), filesizes (single/repl)
        # Deepcopy required to prevent in-place modification from later prefetches
        return deepcopy({
            self.statename("shard_states"): self.shard_states.unsqueeze(0),
            self.statename("file_info"): self.filesizes if self.rank == 0 else None
        })
    
    def load_state_dict(self, state_dict):
        self.setup()
        # Load back shard states (global), filesizes (all)
        shard_states = state_dict[self.statename("shard_states")]
        file_info = state_dict[self.statename("file_info")]
        if shard_states.size(0) == self.worldsize:
            self.filesizes = file_info
            self.shard_states = shard_states[self.rank]
        else:
            shard_states = [s[0] for s in shard_states.split(1)]  # [w] n 5
            shard_states = torch.cat(shard_states, dim=0)  # wn 5
            # Sort shards by epoch count
            sorted, indices = torch.sort(shard_states[:,4], descending=True, stable=True)
            shard_states = shard_states[indices]
            # Strip out dummy shards
            n_dummies = sorted.eq(torch.iinfo(torch.int).max).sum()
            shard_states = shard_states[n_dummies:]  # n_logical 5
            assert len(shard_states) == self.n_logical_shards, f"Number of shards {len(shard_states)} does not match specified {self.n_logical_shards}"
            sorted = sorted[n_dummies:]
            # Split into max and non-max epochs
            n_complete = sorted.eq(sorted[0]).sum()
            completed_shards = shard_states[:n_complete]
            incomplete_shards = shard_states[n_complete:]
            # Allocate completed shards
            completed_shards = [
                completed_shards[
                    round(i*len(completed_shards)/self.worldsize):
                    round((i+1)*len(completed_shards)/self.worldsize)
                ] for i in range(self.worldsize)
            ]
            # Sort completed shards by length
            completed_shards.sort(key=len)
            # Allocate incomplete shards
            incomplete_shards = [
                incomplete_shards[
                    round(i*len(incomplete_shards)/self.worldsize):
                    round((i+1)*len(incomplete_shards)/self.worldsize)
                ] for i in range(self.worldsize)
            ]
            # Reverse sort incomplete shards by length
            incomplete_shards.sort(key=len, reverse=True)
            # Pull out shard allocation for this worker
            # (sort/reverse-sort ensures allocations are off by no more than 1)
            shard_states = torch.cat([
                completed_shards[self.rank],
                incomplete_shards[self.rank]
            ])
            # Order shards by global ID (for steady file progression)
            _, indices = shard_states[:,0].sort()
            self.shard_states[:len(shard_states)] = shard_states[indices]
            # Pad out with dummy shards if needed
            self.shard_states[len(shard_states):,0] = -1
            self.shard_states[len(shard_states):,4] = torch.iinfo(torch.int).max
        return None


#### -------------------------    CHECKPOINT FUNCTIONS    ------------------------- ####


def __pop_dstate(state, device_mesh, placements, create_dtensor=False):
    """
    Removes worker states from the StatefulDataLoader state dict, and assembles them
    into a separate list of dicts of dtensors for distributed checkpointing.
    """
    dstate = state["_snapshot"]["_worker_snapshots"]
    dstate = [dstate[f"worker_{i}"].pop("dataset_state") for i in range(len(dstate))]
    # Flip list[dict[tensor]] to dict[list[tensor]], and concat
    shardstate = "ScalableReader.shard_states"
    fileinfo = "ScalableReader.file_info"
    dstate_dict = {
        shardstate: torch.cat([d[shardstate] for d in dstate], 0)
    }
    if create_dtensor == True:
        dstate_dict[shardstate] = dtensor.DTensor.from_local(
            dstate_dict[shardstate],
            device_mesh,
            placements,
        )
    dstate_dict[fileinfo] = dstate[0][fileinfo]
    return dstate_dict


def save_distributed_state_dict(
    loader: StatefulDataLoader,
    path: str,
    device_mesh: dist.DeviceMesh,
):
    """
    Retrieves dataloader state dict, and separates worker states from loader state.
    Loader state is not rescalable, and is discarded when rescaling.
    Rescalable worker states are compiled into a dtensor across ranks, and saved 
    using pytorch distributed checkpointing.
    """
    state = deepcopy(loader.state_dict())
    dstate = __pop_dstate(state, device_mesh, [dtensor.placement_types.Shard(0)], True)
    # Prune empty fileinfos
    if dstate["ScalableReader.file_info"] is None:
        dstate.pop("ScalableReader.file_info")
    out = {"state":state, "dstate":dstate}
    # Write distributed state dict
    writer = checkpoint.FileSystemWriter(path)
    checkpoint.save(
        out,
        writer,
    )


def load_distributed_state_dict(
    loader: StatefulDataLoader,
    path: str,
    device_mesh: dist.DeviceMesh,
):
    """
    Retrieves dataloader state dict, and separates worker states from loader state.
    If not rescaling, load saved dataloader state.
    Rescalable worker states are retrieved using pytorch distributed checkpointing.
    States are replicated over workers, and ScalableReader will handle
    partitioning and re-assignment of available states into logical ranks.
    """
    base = loader.state_dict()
    nworkers = base["_snapshot"]["_main_snapshot"]["_num_workers"]
    dstate = __pop_dstate(base, device_mesh, [dtensor.placement_types.Replicate()], True)
    inp = {"state":deepcopy(base), "dstate":dstate}
    # Read distributed state dict
    reader = checkpoint.FileSystemReader(path)
    checkpoint.load_state_dict(
        inp,
        reader,
    )
    dstate = inp["dstate"]
    # Check that number of workers matches
    ckp_ws = 0 if not os.path.exists(path) else len([x for x in os.lisdtdir(path) if "loader" in x])
    if ckp_ws == loader.dataset.worldsize and nworkers == state["_snapshot"]["_main_snapshot"]["_num_workers"]:
        state = inp["state"]
    else:
        # On mismatch, discard saved non-reshardable loader state and start fresh
        state = base
    # Repeat global tensor over all workers
    dstate = [inp["dstate"],]*nworkers
    # Re-insert worker states into loader state
    for i in range(nworkers):
        state["_snapshot"]["_worker_snapshots"][f"worker_{i}"]["dataset_state"] = dstate[i]
    # Load into loader
    loader.load_state_dict(state)
