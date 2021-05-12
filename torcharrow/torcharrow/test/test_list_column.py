import unittest
import operator

import torcharrow.dtypes as dt
from torcharrow import IListColumn, INumericalColumn, Scope


class TestListColumn(unittest.TestCase):
    def setUp(self):
        self.ts = Scope()

    def test_empty(self):
        c = self.ts.Column(dt.List(dt.int64))

        self.assertTrue(isinstance(c, IListColumn))
        self.assertEqual(c.dtype, dt.List(dt.int64))

        self.assertTrue(isinstance(c._data, INumericalColumn))
        self.assertEqual(c._data.dtype, dt.int64)

        self.assertEqual(c.length(), 0)
        self.assertEqual(c.null_count(), 0)

        self.assertEqual(len(c._offsets), 1)
        self.assertEqual(c._offsets[0], 0)
        self.assertEqual(len(c._mask), 0)

    def test_nonempty(self):
        c = self.ts.Column(dt.List(dt.int64))
        for i in range(4):
            c = c.append([list(range(i))])

        verdict = [list(range(i)) for i in range(4)]
        for i, lst in zip(range(4), verdict):
            self.assertEqual(c[i], lst)

    def test_nested_numerical_twice(self):
        c = self.ts.Column(
            dt.List(dt.List(dt.Int64(nullable=False), nullable=True), nullable=False)
        )
        vals = [[[1, 2], None, [3, 4]], [[4], [5]]]
        c = c.append(vals)
        self.assertEqual(vals, list(c))

        d = self.ts.Column(
            dt.List(dt.List(dt.Int64(nullable=False), nullable=True), nullable=False)
        )
        for val in vals:
            d = d.append([val])
        self.assertEqual(vals, list(d))

    def test_nested_string_once(self):
        c = self.ts.Column(dt.List(dt.string))
        c = c.append([[]])
        c = c.append([["a"]])
        c = c.append([["b", "c"]])
        self.assertEqual(list([[], ["a"], ["b", "c"]]), list(c))

    def test_nested_string_twice(self):
        c = self.ts.Column(dt.List(dt.List(dt.string)))
        c = c.append([[]])
        c = c.append([[[]]])
        c = c.append([[["a"]]])
        c = c.append([[["b", "c"], ["d", "e", "f"]]])
        self.assertEqual([[], [[]], [["a"]], [["b", "c"], ["d", "e", "f"]]], list(c))

    def test_get_count_join(self):
        c = self.ts.Column(dt.List(dt.string))
        c = c.append([["The", "fox"], ["jumps"], ["over", "the", "river"]])

        self.assertEqual(list(c.list.get(0)), ["The", "jumps", "over"])
        self.assertEqual(list(c.list.count("The")), [1, 0, 0])
        self.assertEqual(list(c.list.join(" ")), ["The fox", "jumps", "over the river"])

    def test_map_reduce_etc(self):
        c = self.ts.Column(dt.List(dt.string))
        c = c.append([["The", "fox"], ["jumps"], ["over", "the", "river"]])
        self.assertEqual(
            list(c.list.map(str.upper)),
            [["THE", "FOX"], ["JUMPS"], ["OVER", "THE", "RIVER"]],
        )
        self.assertEqual(
            list(c.list.filter(lambda x: x.endswith("fox"))), [["fox"], [], []]
        )

        c = self.ts.Column(dt.List(dt.int64))
        c = c.append([list(range(1, i)) for i in range(1, 6)])
        self.assertEqual(list(c.list.reduce(operator.mul, 1)), [1, 1, 2, 6, 24])

        c = self.ts.Column([["what", "a", "wonderful", "world!"], ["really?"]])
        self.assertEqual(
            list(c.list.map(len, dtype=dt.List(dt.int64))), [[4, 1, 9, 6], [7]]
        )

        # flat map on original columns (not on list)
        fst = ["what", "a", "wonderful", "world!"]
        snd = ["really?"]
        c = self.ts.Column([fst, snd])
        self.assertEqual(list(c.flatmap(lambda xs: [xs, xs])), [fst, fst, snd, snd])

        self.ts.Column([1, 2, 3, 4]).map(str, dtype=dt.string)


if __name__ == "__main__":
    unittest.main()
