import unittest

from torcharrow import Column, StringColumn, string

# run python3 -m unittest outside this directory to run all tests


class TestStringColumn(unittest.TestCase):
    def test_empty(self):
        empty = Column(string)
        self.assertTrue(isinstance(empty, StringColumn))
        self.assertEqual(empty._dtype, string)
        self.assertEqual(empty._length, 0)
        self.assertEqual(empty._null_count, 0)
        self.assertEqual(len(empty._data), 0)
        self.assertEqual(len(empty._validity), 0)
        self.assertEqual(empty._offsets[0], 0)

    def test_append_offsets(self):
        c = Column(string)
        c.extend(["abc", "de", "", "f"])
        self.assertEqual(list(c._offsets), [0, 3, 5, 5, 6])
        self.assertEqual(list(c), ["abc", "de", "", "f"])
        with self.assertRaises(TypeError):
            # TypeError: a string is required (got type NoneType)
            c.append(None)

        c = Column(["abc", "de", "", "f", None])
        self.assertEqual(list(c._offsets), [0, 3, 5, 5, 6, 6])
        self.assertEqual(list(c), ["abc", "de", "", "f", None])

    def test_string_split_methods(self):
        c = Column(string)
        s = ["hello.this", "is.interesting.", "this.is_24", "paradise"]
        c.extend(s)
        self.assertEqual(
            list(c.str.split(".", 2, expand=True)),
            [
                ("hello", "this", None),
                ("is", "interesting", ""),
                ("this", "is_24", None),
                ("paradise", None, None),
            ],
        )

    def test_string_lifted_methods(self):
        c = Column(string)
        s = ["abc", "de", "", "f"]
        c.extend(s)
        self.assertEqual(list(c.str.len()), [len(i) for i in s])
        # cat
        self.assertEqual(list(c.str.slice(0, 2)), [i[0:2] for i in s])
        # slice from

        # self.assertEqual(list(c.str.replace(0,2)), [i[0:2] for i in s])

        c = Column(string)
        s = ["hello.this", "is.interesting.", "this.is_24", "paradise"]
        c.extend(s)

        # TODO needs ListColumn -- add back once List is implemented
        # self.assertEqual(list(c.str.split('.', 2, expand=False)), [])
        # expand = True needs Dataframes
        # -- add back once recursive imports are solved

        self.assertEqual(
            list(Column(["1", "", "+3.0", "-4"]).str.isinteger()),
            [True, False, False, True],
        )
        self.assertEqual(
            list(Column(["1.0", "", "+3.e12", "-4.0"]).str.isfloat()),
            [True, False, True, True],
        )

        self.assertEqual(list(Column(["abc123"]).str.isalnum()), [True])
        self.assertEqual(list(Column(["abc"]).str.isalnum()), [True])
        self.assertEqual(list(Column(["abc"]).str.isascii()), [True])
        self.assertEqual(list(Column(["abc"]).str.isdigit()), [False])
        self.assertEqual(
            list(Column([".abc", "abc.a", "_"]).str.isidentifier()),
            [False, False, True],
        )
        self.assertEqual(list(Column([".abc"]).str.islower()), [True])
        self.assertEqual(
            list(Column(["+3.e12", "abc", "0"]).str.isnumeric()), [False, False, True]
        )
        self.assertEqual(
            list(Column(["+3.e12", "abc"]).str.isprintable()), [True, True]
        )
        self.assertEqual(
            list(Column(["\n", "\t", " ", "", "a"]).str.isspace()),
            [True, True, True, False, False],
        )
        self.assertEqual(
            list(Column(["A B C", "abc", " "]).str.istitle()), [True, False, False]
        )
        self.assertEqual(list(Column(["UPPER", "lower"]).str.isupper()), [True, False])

        self.assertEqual(
            list(Column(["UPPER", "lower"]).str.capitalize()), ["Upper", "Lower"]
        )
        self.assertEqual(
            list(Column(["UPPER", "lower"]).str.swapcase()), ["upper", "LOWER"]
        )
        self.assertEqual(
            list(Column(["UPPER", "lower"]).str.lower()), ["upper", "lower"]
        )
        self.assertEqual(
            list(Column(["UPPER", "lower"]).str.upper()), ["UPPER", "LOWER"]
        )
        self.assertEqual(
            list(Column(["UPPER", "lower", "midWife"]).str.casefold()),
            ["upper", "lower", "midwife"],
        )
        # Todo
        # self.assertEqual(list(Column(['1', '22', '33']).str.repeat(2)), [])
        self.assertEqual(
            list(
                Column(["UPPER", "lower", "midWife"]).str.pad(
                    width=10, side="center", fillchar="_"
                )
            ),
            ["__UPPER___", "__lower___", "_midWife__"],
        )
        # ljust, rjust, center
        self.assertEqual(list(Column(["1", "22"]).str.zfill(3)), ["001", "022"])
        self.assertEqual(
            list(Column(s).str.translate({ord("."): ord("_")})),
            ["hello_this", "is_interesting_", "this_is_24", "paradise"],
        )

        self.assertEqual(list(Column(s).str.count(".")), [1, 2, 1, 0])
        self.assertEqual(
            list(Column(s).str.startswith("h")), [True, False, False, False]
        )
        self.assertEqual(
            list(Column(s).str.endswith("this")), [True, False, False, False]
        )
        self.assertEqual(list(Column(s).str.find("this")), [6, -1, 0, -1])
        self.assertEqual(list(Column(s).str.rfind("this")), [6, -1, 0, -1])
        # TODO Clarify: what happens if pat is not there< will it be null?
        self.assertEqual(list(Column(s).str.index("this")), [6, -1, 0, -1])
        self.assertEqual(list(Column(s).str.rindex("this")), [6, -1, 0, -1])


if __name__ == "__main__":
    unittest.main()
