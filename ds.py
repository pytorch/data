# import dataloader.graph as graph

TOTAL_NUMBERS = 100

# class IterDataset:
#     pass
    # def __reduce_ex__(self, *args, **kwargs):
    #     if graph.reduce_ex_hook is not None:
    #         try:
    #             return graph.reduce_ex_hook(self)
    #         except NotImplementedError:
    #             pass
    #     return super().__reduce_ex__(*args, **kwargs)
    

class NumbersDataset(IterDataset):
    def __init__(self, size = TOTAL_NUMBERS):
        self.size = size

    def __iter__(self):
        for i in range(self.size):
            yield i
