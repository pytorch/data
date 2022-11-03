# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import asyncio

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("async_nested_map")
class AsyncNestedMapperIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, async_fn, max_concurrency=32):
        self.source_datapipe = source_datapipe
        self.async_fn = async_fn
        self.max_concurrency = max_concurrency

    def __iter__(self):
        try:
            for i, batch in enumerate(self.source_datapipe):
                new_batch = asyncio.run(self.processbatch(batch))
                yield new_batch
        finally:
            try:
                loop = asyncio.get_running_loop()
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
            except Exception:
                pass

    async def processbatch(self, batch):
        sem = asyncio.Semaphore(self.max_concurrency)

        async def controlled_async_fn(async_fn, data):
            async with sem:
                return await async_fn(data)

        coroutines = []
        for data in batch:
            coroutines.append(controlled_async_fn(self.async_fn, data))
        results = await asyncio.gather(*coroutines)
        return results
