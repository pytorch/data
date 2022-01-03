# Copyright (c) Facebook, Inc. and its affiliates.

# The following utility functions are copied from torchtext
# https://github.com/pytorch/text/blob/main/torchtext/data/datasets_utils.py

import functools
import inspect
import os


def _check_default_set(split, target_select, dataset_name):
    # Check whether given object split is either a tuple of strings or string
    # and represents a valid selection of options given by the tuple of strings
    # target_select.
    if isinstance(split, str):
        split = (split,)
    if isinstance(target_select, str):
        target_select = (target_select,)
    if not isinstance(split, tuple):
        raise ValueError("Internal error: Expected split to be of type tuple.")
    if not set(split).issubset(set(target_select)):
        raise TypeError(
            "Given selection {} of splits is not supported for dataset {}. Please choose from {}.".format(
                split, dataset_name, target_select
            )
        )
    return split


def _wrap_datasets(datasets, split):
    # Wrap return value for _setup_datasets functions to support singular values instead
    # of tuples when split is a string.
    if isinstance(split, str):
        if len(datasets) != 1:
            raise ValueError("Internal error: Expected number of datasets is not 1.")
        return datasets[0]
    return datasets


def _dataset_docstring_header(fn, num_lines=None, num_classes=None):
    """
    Returns docstring for a dataset based on function arguments.
    Assumes function signature of form (root='.data', split=<some tuple of strings>, **kwargs)
    """
    argspec = inspect.getfullargspec(fn)
    if not (argspec.args[0] == "root" and argspec.args[1] == "split"):
        raise ValueError(f"Internal Error: Given function {fn} did not adhere to standard signature.")
    default_split = argspec.defaults[1]

    if not (isinstance(default_split, tuple) or isinstance(default_split, str)):
        raise ValueError(f"default_split type expected to be of string or tuple but got {type(default_split)}")

    header_s = fn.__name__ + " dataset\n"

    if isinstance(default_split, tuple):
        header_s += "\nSeparately returns the {} split".format("/".join(default_split))

    if isinstance(default_split, str):
        header_s += f"\nOnly returns the {default_split} split"

    if num_lines is not None:
        header_s += "\n\nNumber of lines per split:"
        for k, v in num_lines.items():
            header_s += f"\n    {k}: {v}\n"

    if num_classes is not None:
        header_s += "\n\nNumber of classes"
        header_s += f"\n    {num_classes}\n"

    args_s = "\nArgs:"
    args_s += "\n    root: Directory where the datasets are saved."
    args_s += "\n        Default: .data"

    if isinstance(default_split, tuple):
        args_s += "\n    split: split or splits to be returned. Can be a string or tuple of strings."
        args_s += "\n        Default: {}" "".format(str(default_split))

    if isinstance(default_split, str):
        args_s += "\n     split: Only {default_split} is available."
        args_s += "\n         Default: {default_split}.format(default_split=default_split)"

    return "\n".join([header_s, args_s]) + "\n"


def _add_docstring_header(docstring=None, num_lines=None, num_classes=None):
    def docstring_decorator(fn):
        old_doc = fn.__doc__
        fn.__doc__ = _dataset_docstring_header(fn, num_lines, num_classes)
        if docstring is not None:
            fn.__doc__ += docstring
        if old_doc is not None:
            fn.__doc__ += old_doc
        return fn

    return docstring_decorator


def _wrap_split_argument_with_fn(fn, splits):
    """
    Wraps given function of specific signature to extend behavior of split
    to support individual strings. The given function is expected to have a split
    kwarg that accepts tuples of strings, e.g. ('train', 'valid') and the returned
    function will have a split argument that also accepts strings, e.g. 'train', which
    are then turned single entry tuples. Furthermore, the return value of the wrapped
    function is unpacked if split is only a single string to enable behavior such as
    train = AG_NEWS(split='train')
    train, valid = AG_NEWS(split=('train', 'valid'))
    """
    argspec = inspect.getfullargspec(fn)
    if not (
        argspec.args[0] == "root"
        and argspec.args[1] == "split"
        and argspec.varargs is None
        and argspec.varkw is None
        and len(argspec.kwonlyargs) == 0
        and len(argspec.annotations) == 0
    ):
        raise ValueError(f"Internal Error: Given function {fn} did not adhere to standard signature.")

    @functools.wraps(fn)
    def new_fn(root=os.path.expanduser("~/.torchtext/cache"), split=splits, **kwargs):
        result = []
        for item in _check_default_set(split, splits, fn.__name__):
            result.append(fn(root, item, **kwargs))
        return _wrap_datasets(tuple(result), split)

    new_sig = inspect.signature(new_fn)
    new_sig_params = new_sig.parameters
    new_params = []
    new_params.append(new_sig_params["root"].replace(default=".data"))
    new_params.append(new_sig_params["split"].replace(default=splits))
    new_params += [entry[1] for entry in list(new_sig_params.items())[2:]]
    new_sig = new_sig.replace(parameters=tuple(new_params))
    new_fn.__signature__ = new_sig

    return new_fn


def _wrap_split_argument(splits):
    def new_fn(fn):
        return _wrap_split_argument_with_fn(fn, splits)

    return new_fn


def _create_dataset_directory(dataset_name):
    def decorator(func):
        argspec = inspect.getfullargspec(func)
        if not (
            argspec.args[0] == "root"
            and argspec.args[1] == "split"
            and argspec.varargs is None
            and argspec.varkw is None
            and len(argspec.kwonlyargs) == 0
            and len(argspec.annotations) == 0
        ):
            raise ValueError(f"Internal Error: Given function {func} did not adhere to standard signature.")

        @functools.wraps(func)
        def wrapper(root=os.path.expanduser("~/.torchtext/cache"), *args, **kwargs):
            new_root = os.path.join(root, dataset_name)
            if not os.path.exists(new_root):
                os.makedirs(new_root)
            return func(root=new_root, *args, **kwargs)

        return wrapper

    return decorator
