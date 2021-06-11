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
from typing import Union, List

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

    def test_SimpleColumnInt64_neg(self):
        data = [1, 2, None, 3, 4, None]
        col = infer_column(data)
        neg_col = col.neg()

        self.assertEqual(neg_col[0], -1)
        self.assertEqual(neg_col[1], -2)
        self.assertEqual(neg_col[3], -3)
        self.assertEqual(neg_col[4], -4)

        self.assertEqual(col.is_null_at(0), False)
        self.assertEqual(col.is_null_at(1), False)
        self.assertEqual(col.is_null_at(2), True)
        self.assertEqual(col.is_null_at(3), False)
        self.assertEqual(col.is_null_at(4), False)
        self.assertEqual(col.is_null_at(5), True)
        self.assertEqual(col.get_null_count(), 2)

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

    def test_SimpleColumnUTF(self):
        s = ["hello.this", "is.interesting.", "this.is_24", "paradise"]
        col = infer_column(s)
        for i in range(4):
            self.assertEqual(col[i], s[i])

        self.assertEqual(len(col), 4)


def is_same_type(a, b) -> bool:
    if isinstance(a, ta.BIGINT):
        return isinstance(b, ta.BIGINT)
    if isinstance(a, ta.VARCHAR):
        return isinstance(b, ta.VARCHAR)
    if isinstance(a, ta.BOOLEAN):
        return isinstance(b, ta.BOOLEAN)
    if isinstance(a, ta.ARRAY):
        return isinstance(b, ta.ARRAY) and is_same_type(
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
        col = ta.Column(ta.ARRAY(element.type()))
        col.append(element)
        return col
    else:
        return ta.Column(ta.BIGINT())


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
                col = ta.Column(ta.ARRAY(resolved_item_type))
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

            if isinstance(keys_array_type, ta.ARRAY) and isinstance(
                values_array_type, ta.ARRAY
            ):
                col = ta.Column(
                    ta.MAP(
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
            type_ = {int: ta.BIGINT(), str: ta.VARCHAR(), bool: ta.BOOLEAN()}.get(
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
            if type(type_) in (ta.INTEGER, ta.VARCHAR, ta.BOOLEAN):
                col.append(value)
            elif type(type_) == ta.ARRAY:
                col.append(resolve_column(value, type_.element_type()))
            else:
                raise NotImplementedError(f"{type(type_)}")
    return col


class TestInferColumn(unittest.TestCase):
    def test_infer_simple(self):
        data = [1, 2, 3]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.BIGINT()))

    def test_infer_array(self):
        data = [[1], [2], [3]]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.ARRAY(ta.BIGINT())))

    def test_infer_nested_array(self):
        data = [[[1]], [[2], [5]], [[3, 4]]]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.ARRAY(ta.ARRAY(ta.BIGINT()))))

    def test_unresolved(self):
        data = []
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.BIGINT()))

    def test_nested_unresolved1(self):
        data = [[]]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.ARRAY(ta.BIGINT())))

    def test_nested_unresolved2(self):
        data = [None]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.BIGINT()))

    def test_nested_unresolved3(self):
        data = [[None]]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.ARRAY(ta.BIGINT())))

    def test_propagate_unresolved(self):
        data = [None, [], [1], [1, None, 2], None]
        type_ = infer_column(data).type()
        self.assertTrue(is_same_type(type_, ta.ARRAY(ta.BIGINT())))


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
        col = ta.Column(ta.ROW(["a", "b"], [ta.INTEGER(), ta.VARCHAR()]))
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
        col = ta.Column(ta.ROW(["a", "b"], [ta.INTEGER(), ta.VARCHAR()]))
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
            ta.ROW(
                ["a", "b"],
                [ta.INTEGER(), ta.ROW(["b1", "b2"], [ta.VARCHAR(), ta.INTEGER()])],
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
