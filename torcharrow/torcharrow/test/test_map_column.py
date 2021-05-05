import operator
import unittest

from torcharrow import (
    Column,
    Int64,
    List_,
    ListColumn,
    Map,
    MapColumn,
    NumericalColumn,
    StringColumn,
    int64,
    string,
)

# run python3 -m unittest outside this directory to run all tests


class TestMapColumn(unittest.TestCase):
    def test_map(self):
        c = Column(Map(string, int64))
        self.assertIsInstance(c, MapColumn)

        c.append({"abc": 123})
        self.assertDictEqual(c[0], {"abc": 123})

        c.append({"de": 45, "fg": 67})
        self.assertDictEqual(c[1], {"de": 45, "fg": 67})

    def test_infer(self):
        c = Column(
            [
                {"helsinki": [-1.3, 21.5], "moskow": [-4.0, 24.3]},
                {"algiers": [11.2, 25, 2], "kinshasa": [22.2, 26.8]},
            ]
        )
        self.assertIsInstance(c, MapColumn)
        self.assertEqual(len(c), 2)

        self.assertEqual(
            list(c),
            [
                {"helsinki": [-1.3, 21.5], "moskow": [-4.0, 24.3]},
                {"algiers": [11.2, 25, 2], "kinshasa": [22.2, 26.8]},
            ],
        )

    def test_keys_values_get(self):
        c = Column([{"abc": 123}, {"de": 45, "fg": 67}])

        self.assertEqual(list(c.map.keys()), [["abc"], ["de", "fg"]])
        self.assertEqual(list(c.map.values()), [[123], [45, 67]])
        self.assertEqual(c.map.get("de", 0), [0, 45])


if __name__ == "__main__":
    unittest.main()
