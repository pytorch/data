In PyTorch, ATen (https://github.com/pytorch/pytorch/tree/master/aten) is the tensor library stands for "A Tensor",
supports tensor operations that dispatches to different backends (CPU, MKL, CUDA, CUDNN, TPU, etc).

Previously TorchArrow tries to create a similar structure, where ATab is the table/relational library stands for "A Table".
However, given the binding is TorchArrow is a much more slim wrapper, we will simplify it in the future to avoid one more
mysterious name.
