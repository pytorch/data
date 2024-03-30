import itertools
import unittest
from typing import Iterator

import torch

from torchdata.stateful_dataloader import Stateful, StatefulDataLoader


class DummyIterator(Iterator, Stateful):
    def __init__(self, samples):
        self.samples = samples
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.samples):
            raise StopIteration
        sample = self.samples[self.i]
        self.i += 1
        return sample

    def state_dict(self):
        return {"i": self.i}

    def load_state_dict(self, state_dict):
        self.i = state_dict["i"]


class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start_row_pairs):
        self.shards = []
        for start, num_rows in start_row_pairs:
            samples = []
            for i in range(start, start + num_rows):
                samples.append(
                    {
                        "id": i,
                        "strcol": f"strcol_{i}",
                        "floatcol": i + 0.1,
                        "listcol": [i, i + 1, i + 2],
                    }
                )

            self.shards.append(samples)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        iter_data = []
        for shard in self.shards[worker_id : len(self.shards) : num_workers]:
            iter_data.extend(shard)
        return DummyIterator(iter_data)


def identity(x):
    return x


class TestStatefulDataLoader(unittest.TestCase):
    # @pytest.mark.parametrize("num_workers", [0, 3])
    # @pytest.mark.parametrize("batch_size", [None, 7])
    # @pytest.mark.parametrize("pw", [False, True])
    # @pytest.mark.parametrize("interrupt", [0, 1, 10, None])
    # @pytest.mark.parametrize("iter_state", [False, True])
    def test_single_shard(self):
        for num_workers, batch_size, pw, interrupt, iter_state in itertools.product(
            [0, 3],  # num_workers
            [None, 7],  # batch_size
            [False, True],  # pw
            [0, 1, 10, None],  # interrupt
            [False, True],  # iter_state
        ):
            with self.subTest(
                num_workers=num_workers, batch_size=batch_size, pw=pw, interrupt=interrupt, iter_state=iter_state
            ):
                if num_workers == 0 and pw:
                    continue
                if not iter_state:
                    continue
                dataset = DummyIterableDataset([(0, 100)])
                dl = StatefulDataLoader(dataset=dataset, num_workers=num_workers, collate_fn=identity)
                exp = list(dl)

                if interrupt == None:
                    interrupt = len(exp)

                batches = []
                it = iter(dl) if iter_state else dl
                for i in range(interrupt):
                    batches.append(next(it))
                state_dict = it.state_dict()
                self.assertEqual(batches, exp[:interrupt])

                # Restore new instance from state
                dl = StatefulDataLoader(dataset=dataset, num_workers=num_workers, collate_fn=identity)
                it = iter(dl) if iter_state else dl
                it.load_state_dict(state_dict)
                for batch in it:
                    batches.append(batch)

                self.assertEqual(batches, exp)

    # @parameterized.expand(
    #     [
    #         20,
    #         10000,
    #     ]
    # )
    # def test_multiworker(self, num_rows):
    #     pairs = []
    #     worker_rows = num_rows // 4
    #     for i in range(4):
    #         pairs.append((i * worker_rows, worker_rows))

    #     ds = DummyDataset(pairs)
    #     # Check dataloader inline
    #     dl = StatefulDataLoader(dataset=ds, batch_size=4)
    #     for i, batch in enumerate(dl):
    #         exp_ids = [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3]
    #         try:
    #             # Handle truncated last batch
    #             exp_ids = exp_ids[: exp_ids.index(num_rows)]
    #         except ValueError:
    #             pass

    #         batch = rows_to_dict([json.loads(line) for line in batch])
    #         self.assertEqual(set(batch.keys()), {"id", "strcol", "floatcol", "listcol"})
    #         self.assertEqual(batch["id"], exp_ids)
    #         self.assertEqual(batch["strcol"], [f"strcol_{j}" for j in exp_ids])
    #         self.assertEqual(batch["floatcol"], [j + 0.1 for j in exp_ids])
    #         self.assertEqual(
    #             batch["listcol"],
    #             [[j, j + 1, j + 2] for j in exp_ids],
    #         )

    #     dl = StatefulDataLoader(dataset=ds, batch_size=4, num_workers=4)
    #     for i, batch in enumerate(dl):
    #         worker = i % 4
    #         base_id = worker * worker_rows + (i // 4) * 4
    #         exp_ids = [
    #             base_id,
    #             base_id + 1,
    #             base_id + 2,
    #             base_id + 3,
    #         ]
    #         try:
    #             # Handle truncated last batch
    #             cutoff = exp_ids.index((worker + 1) * worker_rows)
    #             exp_ids = exp_ids[:cutoff]
    #         except ValueError:
    #             pass

    #         batch = rows_to_dict([json.loads(line) for line in batch])
    #         self.assertEqual(set(batch.keys()), {"id", "strcol", "floatcol", "listcol"})
    #         self.assertEqual(batch["id"], exp_ids, (i, base_id, worker_rows))
    #         self.assertEqual(batch["strcol"], [f"strcol_{j}" for j in exp_ids])
    #         self.assertEqual(batch["floatcol"], [j + 0.1 for j in exp_ids])
    #         self.assertEqual(
    #             batch["listcol"],
    #             [[j, j + 1, j + 2] for j in exp_ids],
    #         )

    # @parameterized.expand(
    #     list(
    #         itertools.product(
    #             [0, 1, 4],  # num_workers
    #             [2, 10],  # interrupt
    #             [False, True],  # iter_state
    #             [False, True],  # persistent_workers
    #         )
    #     )
    # )
    # def test_state(self, num_workers, interrupt, iter_state, persistent_workers):
    #     if persistent_workers and num_workers == 0:
    #         return
    #     num_rows = 1000
    #     pairs = []
    #     worker_rows = num_rows // 4
    #     for i in range(4):
    #         pairs.append((i * worker_rows, worker_rows))

    #     ds = DummyDataset(pairs)

    #     dl = StatefulDataLoader(
    #         dataset=ds,
    #         batch_size=4,
    #         num_workers=num_workers,
    #         persistent_workers=persistent_workers,
    #     )
    #     exp_result = []
    #     for batch in dl:
    #         exp_result.append([json.loads(x) for x in batch])

    #     dl = StatefulDataLoader(
    #         dataset=ds,
    #         batch_size=4,
    #         num_workers=num_workers,
    #         persistent_workers=persistent_workers,
    #     )
    #     it = iter(dl)
    #     result = []
    #     for _ in range(interrupt):
    #         batch = next(it)
    #         result.append([json.loads(x) for x in batch])
    #     if iter_state:
    #         state_dict = it.state_dict()
    #     else:
    #         state_dict = dl.state_dict()
    #     partial_result = copy.deepcopy(result)

    #     # Test with new dataloader instance
    #     dl = StatefulDataLoader(
    #         dataset=ds,
    #         batch_size=4,
    #         num_workers=num_workers,
    #         persistent_workers=persistent_workers,
    #     )
    #     if iter_state:
    #         it = iter(dl)
    #         it.load_state_dict(state_dict)
    #     else:
    #         dl.load_state_dict(state_dict)
    #         it = iter(dl)
    #     for batch in it:
    #         result.append([json.loads(x) for x in batch])
    #     self.assertEqual(result, exp_result)

    #     # Test fresh start with new iterator
    #     result = []
    #     for batch in dl:
    #         result.append([json.loads(x) for x in batch])
    #     self.assertEqual(result, exp_result)

    #     # Try with new instance of dataset
    #     ds = DummyDataset(pairs)
    #     dl = StatefulDataLoader(
    #         dataset=ds,
    #         batch_size=4,
    #         num_workers=num_workers,
    #         persistent_workers=persistent_workers,
    #     )
    #     result = copy.deepcopy(partial_result)
    #     if iter_state:
    #         it = iter(dl)
    #         it.load_state_dict(state_dict)
    #     else:
    #         dl.load_state_dict(state_dict)
    #         it = iter(dl)
    #     for batch in it:
    #         result.append([json.loads(x) for x in batch])
    #     self.assertEqual(result, exp_result)
