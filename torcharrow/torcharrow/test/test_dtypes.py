
import unittest

# schemas and dtypes
from torcharrow.dtypes import (
    Field, Schema, int64, float64, Boolean, Int64, string, Float64, boolean, String, List_, Map, Struct, is_numerical)

# run python3 -m unittest outside this directory to run all tests

class TestTypes(unittest.TestCase):

    def test_numericals(self):
        # plain type
        self.assertEqual(str(int64),"int64")
        self.assertEqual(int64.size,8)
        self.assertEqual(int64.name,"int64")
        self.assertEqual(int64.typecode,"l")
        self.assertEqual(int64.arraycode,"l")
        self.assertTrue(is_numerical(int64))

    def test_string(self):
        # plain type
        self.assertEqual(str(string),"string")
        self.assertEqual(string.typecode,'u')
        self.assertEqual(string.nullable,False)
        self.assertEqual(String(nullable=True).nullable,True)
        self.assertEqual(string.size,-1)

    def test_list(self):
        self.assertEqual(str(List_(Int64(nullable=True))),"List_(Int64(nullable=True))")
        self.assertEqual(str(List_(Int64(nullable=True)).item_dtype), "Int64(nullable=True)")
        self.assertEqual(List_(Int64(nullable=True)).typecode,"+l")
        self.assertEqual(List_(int).size,-1)

    def test_map(self):
        self.assertEqual(str(Map(int64, string)),"Map(int64, string)")
        self.assertEqual(Map(int64, string).typecode,"+m")
   
    def test_struct(self):
        self.assertEqual(str(Struct([Field('a',int64), Field('b', string)])),"Struct([Field('a', int64), Field('b', string)])")
        self.assertEqual(Struct([Field('a',int64), Field('b', string)]).typecode,"+s")


if __name__ == '__main__':
    unittest.main()