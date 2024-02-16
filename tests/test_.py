from . import unittest


class Test_(unittest.TestCase):
    """Test class for testing the SiteEcon package."""

    def test_(self) -> None:
        """Test the SiteEcon package."""
        self.assertTrue(True)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(Test_('test_'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
