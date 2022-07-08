# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Callable, List, TypeVar

T = TypeVar("T")

# Criteo Data Set Parameters
INT_FEATURE_COUNT = 13
CAT_FEATURE_COUNT = 26
DEFAULT_LABEL_NAME = "label"
DEFAULT_INT_NAMES: List[str] = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
DEFAULT_CAT_NAMES: List[str] = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]
DEFAULT_COLUMN_NAMES: List[str] = [
    DEFAULT_LABEL_NAME,
    *DEFAULT_INT_NAMES,
    *DEFAULT_CAT_NAMES,
]


def safe_cast(val: T, dest_type: Callable[[T], T], default: T) -> T:
    """
    Helper function to safely cast data with default as fallback.
    """
    try:
        return dest_type(val)
    except ValueError:
        return default
