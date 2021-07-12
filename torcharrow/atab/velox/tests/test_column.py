#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Union, List, Any

# @manual=//pytorch/torchdata/torcharrow/atab/velox:_torcharrow
import _torcharrow as ta


class TestSimpleColumns(unittest.TestCase):
    def test_SimpleColumnInt64(self):
        data = [1, 2, None, 3, 4, None]
        col = infer_column(data)

        self.assertEqual(col[0], 1)
        self.assertEqual(col[1], 2)
        self.assertEqual(col[3], 3)
        self.assertEqual(col[4], 4)

        self.assertEqual(len(col), 6)

        with self.assertRaises(TypeError):
            # TypeError: an integer is required (got type NoneType)
            col.append(None)

        with self.assertRaises(TypeError):
            # TypeError: an integer is required (got type String)
            col.append("hello")

        self.assertEqual(col.is_null_at(0), False)
        self.assertEqual(col.is_null_at(1), False)
        self.assertEqual(col.is_null_at(2), True)
        self.assertEqual(col.is_null_at(3), False)
        self.assertEqual(col.is_null_at(4), False)
        self.assertEqual(col.is_null_at(5), True)
        self.assertEqual(col.get_null_count(), 2)

        sliced_col = col.slice(1, 3)
        self.assertEqual(len(sliced_col), 3)
        self.assertEqual(sliced_col[0], 2)
        self.assertEqual(sliced_col[2], 3)
        self.assertEqual(sliced_col.get_null_count(), 1)

    def assert_SimpleColumn(self, col: ta.BaseColumn, val: List[Any]):
        self.assertEqual(len(col), len(val))
        for i in range(len(val)):
            if val[i] is None:
                self.assertTrue(col.is_null_at(i))
            else:
                self.assertFalse(col.is_null_at(i))
                if isinstance(val[i], float):
                    self.assertAlmostEqual(col[i], val[i], places=6)
                else:
                    self.assertEqual(col[i], val[i])

    def test_SimpleColumnInt64_unary(self):
        data = [1, -2, None, 3, -4, None]
        col = infer_column(data)
        self.assertEqual(col.type().kind_name(), 'BIGINT')

        neg_col = col.neg()
        self.assert_SimpleColumn(neg_col, [-1, 2, None, -3, 4, None])
        self.assertEqual(neg_col.type().kind_name(), 'BIGINT')

        neg_col2 = neg_col.neg()
        self.assert_SimpleColumn(neg_col2, [1, -2, None, 3, -4, None])
        self.assertEqual(neg_col2.type().kind_name(), 'BIGINT')

        neg_col3 = neg_col2.neg()
        self.assert_SimpleColumn(neg_col3, [-1, 2, None, -3, 4, None])
        self.assertEqual(neg_col3.type().kind_name(), 'BIGINT')

        abs_col = col.abs()
        self.assert_SimpleColumn(abs_col, [1, 2, None, 3, 4, None])
        self.assertEqual(abs_col.type().kind_name(), 'BIGINT')

    def test_SimpleColumnInt64_binary(self):
        data1= [1, -2, None, 3, -4, None]
        col1 = infer_column(data1)
        data2= [None, 1, 2, 3, 4, 5]
        col2 = infer_column(data2)

        sum_col = col1.add(col2)
        self.assert_SimpleColumn(sum_col, [None, -1, None, 6, 0, None])
        self.assertEqual(sum_col.type().kind_name(), 'BIGINT')

        # type promotion
        data3= [None, 1., 2., 3., 4., 5.]
        col3 = infer_column(data3)
        self.assertEqual(col3.type().kind_name(), 'REAL')

        sum_col = col1.add(col3)
        self.assertEqual(sum_col.type().kind_name(), 'REAL')
        self.assert_SimpleColumn(sum_col, [None, -1., None, 6., 0., None])

        sum_col2 = col3.add(col1)
        self.assertEqual(sum_col2.type().kind_name(), 'REAL')
        self.assert_SimpleColumn(sum_col2, [None, -1., None, 6., 0., None])

        # add scalar
        add1 = col1.add(1)
        self.assertEqual(add1.type().kind_name(), 'BIGINT')
        self.assert_SimpleColumn(add1, [2, -1, None, 4, -3, None])


    def test_SimpleColumnFloat32_unary(self):
        data = [1.2, -2.3, None, 3.4, -4.6, None]
        col = infer_column(data)
        self.assertEqual(col.type().kind_name(), 'REAL')

        neg_col = col.neg()
        self.assert_SimpleColumn(neg_col, [-1.2, 2.3, None, -3.4, 4.6, None])
        self.assertEqual(neg_col.type().kind_name(), 'REAL')

        abs_col = col.abs()
        self.assert_SimpleColumn(abs_col, [1.2, 2.3, None, 3.4, 4.6, None])
        self.assertEqual(abs_col.type().kind_name(), 'REAL')

        round_col = col.round()
        self.assert_SimpleColumn(round_col, [1., -2., None, 3., -5., None])
        self.assertEqual(round_col.type().kind_name(), 'REAL')

    def test_SimpleColumnBoolean(self):
        data = [True, True, True, True]
        col = infer_column(data)

        for i in range(4):
            self.assertEqual(col[i], True)

        self.assertEqual(len(col), 4)

        with self.assertRaises(TypeError):
            # TypeError: a boolean is required (got type NoneType)
            col.append(None)

        with self.assertRaises(TypeError):
            # TypeError: a boolean is required (got type String)
            col.append("hello")

        col.append_null()
        self.assertEqual(col.is_null_at(0), False)
        self.assertEqual(col.is_null_at(1), False)
        self.assertEqual(col.is_null_at(2), False)
        self.assertEqual(col.is_null_at(3), False)
        self.assertEqual(col.is_null_at(4), True)

    def test_SimpleColumnBoolean_unary(self):
        data = [True, False, None, True, False, None]
        col = infer_column(data)
        self.assertEqual(col.type().kind_name(), 'BOOLEAN')

        inv_col = col.invert()
        self.assertEqual(inv_col.type().kind_name(), 'BOOLEAN')
        self.assert_SimpleColumn(inv_col, [False, True, None, False, True, None])

    def test_SimpleColumnString(self):
        data = ["0", "1", "2", "3"]
        col = infer_column(data)

        for i in range(4):
            self.assertEqual(col[i], str(i))

        self.assertEqual(len(col), 4)

        with self.assertRaises(TypeError):
            # TypeError: a string is required (got type NoneType)
            col.append(None)

        with self.assertRaises(TypeError):
            # TypeError: a string is required (got type int)
            col.append(1)

        col.append_null()
        self.assertEqual(col.is_null_at(0), False)
        self.assertEqual(col.is_null_at(1), False)
        self.assertEqual(col.is_null_at(2), False)
        self.assertEqual(col.is_null_at(3), False)
        self.assertEqual(col.is_null_at(4), True)

    def test_SimpleColumnString_unary(self):
        data = ["abc", "ABC", "XYZ123", None, "xYZ", "123", "äöå", ",.!"]
        col = infer_column(data)

        lcol = col.lower()
        self.assert_SimpleColumn(lcol, ["abc", "abc", "xyz123", None, "xyz", "123", "äöå", ",.!"])

        ucol = col.upper()
        self.assert_SimpleColumn(ucol, ["ABC", "ABC", "XYZ123", None, "XYZ", "123", "ÄÖÅ", ",.!"])

        lcol2 = ucol.lower()
        self.assert_SimpleColumn(lcol2, ["abc", "abc", "xyz123", None, "xyz", "123", "äöå", ",.!"])

        ucol2 = lcol.upper()
        self.assert_SimpleColumn(ucol2, ["ABC", "ABC", "XYZ123", None, "XYZ", "123", "ÄÖÅ", ",.!"])

        alpha = col.isalpha()
        self.assert_SimpleColumn(alpha, [True, True, False, None, True, False, True, False])

        alnum = col.isalnum()
        self.assert_SimpleColumn(alnum, [True, True, True, None, True, True, True, False])


    def test_SimpleColumnUTF(self):
        s = ["hello.this", "is.interesting.", "this.is_24", "paradise"]
        col = infer_column(s)
        for i in range(4):
            self.assertEqual(col[i], s[i])

        self.assertEqual(len(col), 4)

    def test_ConstantColumn(self):
        ###########
        #  BIGINT
        col = ta.ConstantColumn(42, 6)
        self.assertTrue(isinstance(col.type(), ta.VeloxType_BIGINT))
        self.assert_SimpleColumn(col, [42] * 6)

        # Test use constant column for normal add
        data = [1, -2, None, 3, -4, None]
        num_column = infer_column(data)
        add_result = num_column.add(col)
        self.assertTrue(isinstance(add_result.type(), ta.VeloxType_BIGINT))
        self.assert_SimpleColumn(add_result, [43, 40, None, 45, 38, None])

        add_result = col.add(num_column)
        self.assertTrue(isinstance(add_result.type(), ta.VeloxType_BIGINT))
        self.assert_SimpleColumn(add_result, [43, 40, None, 45, 38, None])


        ###########
        #  REAL
        col = ta.ConstantColumn(4.2, 6)
        self.assertTrue(isinstance(col.type(), ta.VeloxType_REAL))
        self.assert_SimpleColumn(col, [4.2] * 6)

        # Test use constant column for normal add
        data = [1.2, -2.3, None, 3.4, -4.6, None]
        num_column = infer_column(data)
        add_result = num_column.add(col)
        self.assertTrue(isinstance(add_result.type(), ta.VeloxType_REAL))
        self.assert_SimpleColumn(add_result, [5.4, 1.9, None, 7.6, -0.4, None])

        add_result = col.add(num_column)
        self.assertTrue(isinstance(add_result.type(), ta.VeloxType_REAL))
        self.assert_SimpleColumn(add_result, [5.4, 1.9, None, 7.6, -0.4, None])


        ###########
        #  VARCHAR
        col = ta.ConstantColumn('abc', 6)
        self.assertTrue(isinstance(col.type(), ta.VeloxType_VARCHAR))
        self.assert_SimpleColumn(col, ['abc'] * 6)

def is_same_type(a, b) -> bool:
    if isinstance(a, ta.VeloxType_BIGINT):
        return isinstance(b, ta.VeloxType_BIGINT)
    if isinstance(a, ta.VeloxType_VARCHAR):
        return isinstance(b, ta.VeloxType_VARCHAR)
    if isinstance(a, ta.VeloxType_BOOLEAN):
        return isinstance(b, ta.VeloxType_BOOLEAN)
    if isinstance(a, ta.VeloxArrayType):
        return isinstance(b, ta.VeloxArrayType) and is_same_type(
            a.element_type(), b.element_type()
        )
    raise NotImplementedError()


# infer result
@dataclass(frozen=True)
class Unresolved:
    def union(self, other: Unresolved) -> Unresolved:
        return other

@dataclass(frozen=True)
class UnresolvedArray(Unresolved):
    element_type: Unresolved

def infer_column(data) -> ta.BaseColumn:
    inferred_column = _infer_column(data)
    if isinstance(inferred_column, Unresolved):
        return resolve_column_with_arbitrary_type(inferred_column)
    else:
        return inferred_column


def resolve_column_with_arbitrary_type(unresolved: Unresolved) -> ta.BaseColumn:
    if isinstance(unresolved, UnresolvedArray):
        element = resolve_column_with_arbitrary_type(unresolved.element_type)
        col = ta.Column(ta.VeloxArrayType(element.type()))
        col.append(element)
        return col
    else:
        return ta.Column(ta.VeloxType_BIGINT())


def get_union_type(inferred_columns: List[Union[ta.BaseColumn, Unresolved, None]]):
    unresolved_item_type = None
    resolved_item_type = None
    for item_col in inferred_columns:
        if item_col is None:
            pass
        elif isinstance(item_col, Unresolved):
            if unresolved_item_type is None:
                unresolved_item_type = item_col
            else:
                unresolved_item_type = unresolved_item_type.union(item_col)

        elif resolved_item_type is None:
            resolved_item_type = item_col.type()
        else:
            assert is_same_type(resolved_item_type, item_col.type())

    if resolved_item_type is None:
        if unresolved_item_type is None:
            return None
        else:
            return unresolved_item_type
    else:
        return resolved_item_type


def _infer_column(data) -> Union[ta.BaseColumn, Unresolved, None]:
    if data is None:
        return None
    assert isinstance(data, list)
    non_null_item = next((item for item in data if item is not None), None)

    if non_null_item is None:
        return Unresolved()
    else:
        if isinstance(non_null_item, list):
            inferred_columns = [_infer_column(item) for item in data]
            union_type = get_union_type(inferred_columns)
            if union_type is None:
                return Unresolved()
            elif isinstance(union_type, Unresolved):
                return UnresolvedArray(union_type)
            else:
                resolved_item_type = union_type
                col = ta.Column(ta.VeloxArrayType(resolved_item_type))
                for item_col, item in zip(inferred_columns, data):
                    if item is None:
                        resolved_item_col = None
                    elif isinstance(item_col, Unresolved):
                        resolved_item_col = resolve_column(item, resolved_item_type)
                    else:
                        resolved_item_col = item_col

                    if resolved_item_col is None:
                        col.append_null()
                    else:
                        col.append(resolved_item_col)
                return col
        elif isinstance(non_null_item, dict):
            keys_array = []
            values_array = []
            for item in data:
                if item is None:
                    keys_array.append(None)
                    values_array.append(None)
                elif isinstance(item, dict):
                    keys_array.append(list(item.keys()))
                    values_array.append(list(item.values()))
                else:
                    raise ValueError("non-dict item in dict list")

            inferred_keys_array_columns = _infer_column(keys_array)
            inferred_values_array_columns = _infer_column(values_array)

            keys_array_type = inferred_keys_array_columns.type()
            values_array_type = inferred_values_array_columns.type()

            if isinstance(keys_array_type, ta.VeloxArrayType) and isinstance(
                values_array_type, ta.VeloxArrayType
            ):
                col = ta.Column(
                    ta.VeloxMapType(
                        keys_array_type.element_type(), values_array_type.element_type()
                    )
                )
                for item in data:
                    if item is None:
                        col.append_null()
                    else:
                        key_col = ta.Column(keys_array_type.element_type())
                        value_col = ta.Column(values_array_type.element_type())
                        for key, value in item.items():
                            key_col.append(key)
                            if value is None:
                                value_col.append_null()
                            else:
                                value_col.append(value)
                        col.append(key_col, value_col)
                return col
            else:
                raise NotImplementedError()

        else:
            type_ = {int: ta.VeloxType_BIGINT(), float: ta.VeloxType_REAL(), str: ta.VeloxType_VARCHAR(), bool: ta.VeloxType_BOOLEAN()}.get(
                type(non_null_item)
            )
            if type_ is None:
                raise NotImplementedError(f"Cannot infer {type(non_null_item)}")
            else:
                col = ta.Column(type_)
                for item in data:
                    if item is None:
                        col.append_null()
                    else:
                        col.append(item)
                return col


def resolve_column(item, type_) -> ta.BaseColumn:
    col = ta.Column(type_)
    for value in item:
        if value is None:
            col.append_null()
        else:
            if type(type_) in (ta.VeloxType_INTEGER, ta.VeloxType_VARCHAR, ta.VeloxType_BOOLEAN):
                col.append(value)
            elif type(type_) == ta.VeloxArrayType:
                col.append(resolve_column(value, type_.element_type()))
            else:
                raise NotImplementedError(f"{type(type_)}")
    return col


class TestInferColumn(unittest.TestCase):
    def test_infer_simple(self):
        data = [1, 2, 3]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.VeloxType_BIGINT()))

    def test_infer_array(self):
        data = [[1], [2], [3]]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.VeloxArrayType(ta.VeloxType_BIGINT())))

    def test_infer_nested_array(self):
        data = [[[1]], [[2], [5]], [[3, 4]]]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.VeloxArrayType(ta.VeloxArrayType(ta.VeloxType_BIGINT()))))

    def test_unresolved(self):
        data = []
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.VeloxType_BIGINT()))

    def test_nested_unresolved1(self):
        data = [[]]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.VeloxArrayType(ta.VeloxType_BIGINT())))

    def test_nested_unresolved2(self):
        data = [None]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.VeloxType_BIGINT()))

    def test_nested_unresolved3(self):
        data = [[None]]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.VeloxArrayType(ta.VeloxType_BIGINT())))

    def test_propagate_unresolved(self):
        data = [None, [], [1], [1, None, 2], None]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.VeloxArrayType(ta.VeloxType_BIGINT())))


class TestArrayColumns(unittest.TestCase):
    def test_ArrayColumnInt64(self):
        data = [None, [], [1], [1, None, 2], None]
        col = infer_column(data)

        for sliced_col, sliced_data in (
            (col, data),
            (col.slice(2, 2), data[2:4]),
            (col.slice(1, 4), data[1:5]),
        ):
            self.assertEqual(len(sliced_col), len(sliced_data))
            for i, item in enumerate(sliced_data):
                if item is None:
                    self.assertTrue(sliced_col.is_null_at(i))
                else:
                    self.assertFalse(sliced_col.is_null_at(i))
                    self.assertEqual(len(sliced_col[i]), len(item))
                    for j, value in enumerate(item):
                        if value is None:
                            self.assertTrue(sliced_col[i].is_null_at(j))
                        else:
                            self.assertFalse(sliced_col[i].is_null_at(j))
                            self.assertEqual(sliced_col[i][j], sliced_data[i][j])

    def test_NestedArrayColumnInt64(self):
        data = [[[1, 2], None, [3, 4]], [[4], [5]]]
        col = infer_column(data)
        self.assertEqual(col[0][0][0], 1)
        self.assertEqual(col[0][0][1], 2)
        self.assertTrue(col[0].is_null_at(1))
        self.assertEqual(col[0][2][0], 3)
        self.assertEqual(col[0][2][1], 4)
        self.assertEqual(col[1][0][0], 4)
        self.assertEqual(col[1][1][0], 5)

    def test_NestedArrayColumnString(self):
        data = [[], [[]], [["a"]], [["b", "c"], ["d", "e", "f"]]]
        col = infer_column(data)
        self.assertEqual(len(col[0]), 0)
        self.assertEqual(len(col[1]), 1)
        self.assertEqual(len(col[1][0]), 0)
        self.assertEqual(col[2][0][0], "a")
        self.assertEqual(col[3][0][0], "b")
        self.assertEqual(col[3][0][1], "c")
        self.assertEqual(col[3][1][0], "d")
        self.assertEqual(col[3][1][1], "e")
        self.assertEqual(col[3][1][2], "f")


class TestMapColumns(unittest.TestCase):
    def test_MapColumnInt64(self):
        data = [{"a": 1, "b": 2}, {"c": 3, "d": 4, "e": 5}]
        col = infer_column(data)
        self.assertEqual(len(col), 2)
        keys = col.keys()
        self.assertEqual(len(keys), 2)
        self.assertEqual(len(keys[0]), 2)
        self.assertEqual(keys[0][0], "a")
        self.assertEqual(keys[0][1], "b")
        self.assertEqual(len(keys[1]), 3)
        self.assertEqual(keys[1][0], "c")
        self.assertEqual(keys[1][1], "d")
        self.assertEqual(keys[1][2], "e")
        values = col.values()
        self.assertEqual(len(values), 2)
        self.assertEqual(len(values[0]), 2)
        self.assertEqual(values[0][0], 1)
        self.assertEqual(values[0][1], 2)
        self.assertEqual(len(values[1]), 3)
        self.assertEqual(values[1][0], 3)
        self.assertEqual(values[1][1], 4)
        self.assertEqual(values[1][2], 5)

        sliced_col = col.slice(1, 1)
        self.assertEqual(len(sliced_col), 1)
        keys = sliced_col.keys()
        self.assertEqual(len(keys), 1)
        self.assertEqual(len(keys[0]), 3)
        self.assertEqual(keys[0][0], "c")
        self.assertEqual(keys[0][1], "d")
        self.assertEqual(keys[0][2], "e")
        values = sliced_col.values()
        self.assertEqual(len(values), 1)
        self.assertEqual(len(values[0]), 3)
        self.assertEqual(values[0][0], 3)
        self.assertEqual(values[0][1], 4)
        self.assertEqual(values[0][2], 5)

    def test_MapColumnInt64_with_none(self):
        data = [None, {"a": 1, "b": 2}, {"c": None, "d": 4, "e": 5}]
        col = infer_column(data)
        self.assertEqual(len(col), 3)
        self.assertTrue(col.is_null_at(0))
        keys = col.keys()
        self.assertEqual(len(keys), 3)
        self.assertEqual(len(keys[1]), 2)
        self.assertEqual(keys[1][0], "a")
        self.assertEqual(keys[1][1], "b")
        self.assertEqual(len(keys[2]), 3)
        self.assertEqual(keys[2][0], "c")
        self.assertEqual(keys[2][1], "d")
        self.assertEqual(keys[2][2], "e")
        values = col.values()
        self.assertEqual(len(values), 3)
        self.assertEqual(len(values[1]), 2)
        self.assertEqual(values[1][0], 1)
        self.assertEqual(values[1][1], 2)
        self.assertEqual(len(values[2]), 3)
        self.assertTrue(values[2].is_null_at(0))
        self.assertEqual(values[2][1], 4)
        self.assertEqual(values[2][2], 5)

        sliced_col = col.slice(1, 1)
        self.assertEqual(len(sliced_col), 1)
        keys = sliced_col.keys()
        self.assertEqual(len(keys), 1)
        self.assertEqual(len(keys[0]), 2)
        self.assertEqual(keys[0][0], "a")
        self.assertEqual(keys[0][1], "b")
        values = sliced_col.values()
        self.assertEqual(len(values), 1)
        self.assertEqual(len(values[0]), 2)
        self.assertEqual(values[0][0], 1)
        self.assertEqual(values[0][1], 2)


class TestRowColumns(unittest.TestCase):
    def test_RowColumn1(self):
        col = ta.Column(ta.VeloxRowType(["a", "b"], [ta.VeloxType_INTEGER(), ta.VeloxType_VARCHAR()]))
        col.child_at(0).append(1)
        col.child_at(1).append("x")
        col.set_length(1)
        col.child_at(0).append(2)
        col.child_at(1).append("y")
        col.set_length(2)

        self.assertEqual(col.type().name_of(0), "a")
        self.assertEqual(col.type().name_of(1), "b")
        self.assertEqual(col.child_at(col.type().get_child_idx("a"))[0], 1)
        self.assertEqual(col.child_at(col.type().get_child_idx("b"))[0], "x")
        self.assertEqual(col.child_at(col.type().get_child_idx("a"))[1], 2)
        self.assertEqual(col.child_at(col.type().get_child_idx("b"))[1], "y")

        sliced_col = col.slice(1, 1)
        self.assertEqual(
            sliced_col.child_at(sliced_col.type().get_child_idx("a"))[0], 2
        )
        self.assertEqual(
            sliced_col.child_at(sliced_col.type().get_child_idx("b"))[0], "y"
        )

    def test_set_child(self):
        col = ta.Column(ta.VeloxRowType(["a", "b"], [ta.VeloxType_INTEGER(), ta.VeloxType_VARCHAR()]))
        col.child_at(0).append(1)
        col.child_at(1).append("x")
        col.set_length(1)
        col.child_at(0).append(2)
        col.child_at(1).append("y")
        col.set_length(2)

        new_child = infer_column([3, 4])
        col.set_child(0, new_child)

        self.assertEqual(col.type().name_of(0), "a")
        self.assertEqual(col.type().name_of(1), "b")
        self.assertEqual(col.child_at(col.type().get_child_idx("a"))[0], 3)
        self.assertEqual(col.child_at(col.type().get_child_idx("b"))[0], "x")
        self.assertEqual(col.child_at(col.type().get_child_idx("a"))[1], 4)
        self.assertEqual(col.child_at(col.type().get_child_idx("b"))[1], "y")

    def test_nested_row(self):
        col = ta.Column(
            ta.VeloxRowType(
                ["a", "b"],
                [ta.VeloxType_INTEGER(), ta.VeloxRowType(["b1", "b2"], [ta.VeloxType_VARCHAR(), ta.VeloxType_INTEGER()])],
            )
        )
        col.child_at(0).append(1)
        col.child_at(1).child_at(0).append("21")
        col.child_at(1).child_at(1).append(22)
        self.assertEqual(col.type().get_child_idx("a"), 0)
        self.assertEqual(col.type().get_child_idx("b"), 1)
        self.assertEqual(col.child_at(1).type().get_child_idx("b1"), 0)
        self.assertEqual(col.child_at(1).type().get_child_idx("b2"), 1)
        self.assertEqual(col.child_at(0)[0], 1)
        self.assertEqual(col.child_at(1).child_at(0)[0], "21")
        self.assertEqual(col.child_at(1).child_at(1)[0], 22)


if __name__ == "__main__":
    unittest.main()
