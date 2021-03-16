
import operator
import unittest

from torcharrow import (Column, Int64, List_, ListColumn, NumericalColumn,
                        int64, string)

# run python3 -m unittest outside this directory to run all tests


class TestListColumn(unittest.TestCase):

    def test_empty(self):
        c = Column(List_(int64))

        self.assertTrue(isinstance(c, ListColumn))
        self.assertEqual(c._dtype, List_(int64))

        self.assertTrue(isinstance(c._data, NumericalColumn))
        self.assertEqual(c._data._dtype, int64)

        self.assertEqual(c._length, 0)
        self.assertEqual(c._null_count, 0)
        self.assertEqual(c._offset, 0)

        self.assertEqual(len(c._offsets), 1)
        self.assertEqual(c._offsets[0], 0)
        self.assertEqual(len(c._validity), 0)

    def test_nonempty(self):
        c = Column(List_(int64))
        self.assertEqual(c._data._offset, 0)
        for i in range(4):
            self.assertEqual(c._offsets[i], len(c._data._data))
            c.append(list(range(i)))
            self.assertEqual(list(c[i]), list(range(i)))
            self.assertEqual(c._validity[i], True)
            self.assertEqual(c._offsets[i+1], len(c._data._data))

        verdict = [list(range(i)) for i in range(4)]

        for i, lst in zip(range(4), verdict):
            self.assertEqual(c[i], lst)

        c.extend([[-1, -2, -3], [-4, -5]])
        # this is the last validity...
        self.assertEqual(c._data._validity[c._offsets[-1]-1], True)

    def test_nested_numerical_twice(self):
        c = Column(
            List_(List_(Int64(nullable=False), nullable=True), nullable=False))
        vals = [
            [
                [1, 2],
                None,
                [3, 4]
            ],
            [
                [4],
                [5]
            ]
        ]
        c.append(vals[0])
        c.append(vals[1])
        self.assertEqual(vals, list(c))

    def test_nested_string_once(self):
        c = Column(List_(string))
        c.append([])
        c.append(["a"])
        c.append(["b", "c"])
        # s/elf.assertEqual(list([[],["a"],["b","c"]]),list(c))

    def test_nested_string_twice(self):
        c = Column(List_(List_(string)))
        c.append([])
        c.append([[]])
        c.append([["a"]])
        c.append([["b", "c"], ["d", "e", "f"]])
        self.assertEqual(
            [[], [[]], [["a"]], [["b", "c"], ["d", "e", "f"]]], list(c))

    def test_get_count_join(self):
        c = Column(List_(string))
        c.extend([['The', 'fox'], ['jumps'], ['over', 'the', 'river']])

        self.assertEqual(list(c.list.get(0)), ['The', 'jumps', 'over'])
        self.assertEqual(list(c.list.count('The')), [1, 0, 0])
        self.assertEqual(list(c.list.join(' ')), [
                         'The fox', 'jumps', 'over the river'])

    def test_map_reduce_etc(self):
        c = Column(List_(string))
        c.extend([['The', 'fox'], ['jumps'], ['over', 'the', 'river']])
        self.assertEqual(list(c.list.map(str.upper)), [
                         ['THE', 'FOX'], ['JUMPS'], ['OVER', 'THE', 'RIVER']])
        self.assertEqual(list(c.list.filter(lambda x: x.endswith('fox'))), [
                         ['fox'], [], []])

        c = Column(List_(int64))
        c.extend([list(range(1, i)) for i in range(1, 6)])
        self.assertEqual(list(c.list.reduce(operator.mul, 1)),
                         [1, 1, 2, 6, 24])

        c = Column([
            ['what', 'a', 'wonderful', 'world!'],
            ['really?']])
        self.assertEqual(list(c.list.map(len, dtype=List_(int64))), [
                         [4, 1, 9, 6], [7]])

        # flat map on original columns (not on list)
        fst = ['what', 'a', 'wonderful', 'world!']
        snd = ['really?']
        c = Column([fst, snd])
        self.assertEqual(list(c.flatmap(lambda xs: [xs, xs])), [
                         fst, fst, snd, snd])

        Column([1, 2, 3, 4]).map(str, dtype=string)


if __name__ == '__main__':
    unittest.main()
