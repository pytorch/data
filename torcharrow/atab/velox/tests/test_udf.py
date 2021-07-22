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

import unittest
from typing import List, Any

# @manual=//pytorch/torchdata/torcharrow/atab/velox:_torcharrow
import _torcharrow as ta

class TestSimpleColumns(unittest.TestCase):
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

    @staticmethod
    def construct_simple_column(velox_type, data: List[Any]):
        col = ta.Column(velox_type)
        for item in data:
            if item is None:
                col.append_null()
            else:
                col.append(item)
        return col

    def test_basic(self):
        # test some UDFs together
        data = ["abc", "ABC", "XYZ123", None, "xYZ", "123", "äöå"]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)

        lcol = ta.generic_udf_dispatch("lower", col)
        self.assert_SimpleColumn(lcol, ["abc", "abc", "xyz123", None, "xyz", "123", "äöå"])

        ucol = ta.generic_udf_dispatch("upper", col)
        self.assert_SimpleColumn(ucol, ["ABC", "ABC", "XYZ123", None, "XYZ", "123", "ÄÖÅ"])

        lcol2 = ta.generic_udf_dispatch("lower", ucol)
        self.assert_SimpleColumn(lcol2, ["abc", "abc", "xyz123", None, "xyz", "123", "äöå"])

        ucol2 = ta.generic_udf_dispatch("upper", lcol)
        self.assert_SimpleColumn(ucol2, ["ABC", "ABC", "XYZ123", None, "XYZ", "123", "ÄÖÅ"])

        alpha = ta.generic_udf_dispatch("torcharrow_isalpha", col)
        self.assert_SimpleColumn(alpha, [True, True, False, None, True, False, True])

        data2 = [1, 2, 3, None, 5, None, -7]
        col2 = self.construct_simple_column(ta.VeloxType_BIGINT(), data2)

        neg = ta.generic_udf_dispatch("negate", col2)
        self.assert_SimpleColumn(neg, [-1, -2, -3, None, -5, None, 7])


    def test_lower(self):
        data = ["abc", "ABC", "XYZ123", None, "xYZ", "123", "äöå"]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)
        lcol = ta.generic_udf_dispatch("lower", col)
        self.assert_SimpleColumn(lcol, ["abc", "abc", "xyz123", None, "xyz", "123", "äöå"])
