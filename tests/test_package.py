import unittest

class TestPackage(unittest.TestCase):

    def test_import(self):
        try:
            import jelli
        except ImportError:
            self.fail("Importing jelli failed.")
        else:
            self.assertIsNotNone(jelli)
