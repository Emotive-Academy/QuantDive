# Import the test modules
from . import unittest
from .test_ import Test_


# Create a test suite.
def test_suite() -> None:
    suite = unittest
    suite.addTest(unittest.makeSuite(Test_))
    return suite


# Run the test suite if this package is run as a script.
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(test_suite())
