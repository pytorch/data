import types

import _torcharrow as ta

from .column import ColumnFromVelox


class VeloxFunctional(types.ModuleType):
    def __init__(self):
        super().__init__("torcharrow.velox_rt.functional")
        self._populate_udfs()

    @staticmethod
    def create_dispatch_wrapper(op_name: str):
        def wrapper(*args):
            wrapped_args = []

            first_col = next(
                (arg for arg in args if isinstance(arg, ColumnFromVelox)), None
            )
            if first_col is None:
                raise AssertionError("None of the argument is Column")
            length = len(first_col)

            for arg in args:
                if isinstance(arg, ColumnFromVelox):
                    wrapped_args.append(arg._data)
                else:
                    # constant value
                    wrapped_args.append(ta.ConstantColumn(arg, length))

            result_col = ta.generic_udf_dispatch(op_name, *wrapped_args)
            # Generic dispatch always assumes nullable
            result_dtype = result_col.dtype().with_null(True)

            return ColumnFromVelox.from_velox(
                first_col.scope, first_col.device, result_dtype, result_col, True
            )

        return wrapper

    # TODO: automtically populate it
    def __getattr__(self, op_name: str):
        return self.create_dispatch_wrapper(op_name)

    def _populate_udfs(self):
        # TODO: implement this
        pass


functional = VeloxFunctional()
