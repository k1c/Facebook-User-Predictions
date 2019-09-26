import unittest

from data.readers import read_train_data


class TestReaders(unittest.TestCase):
    def test_read(self):
        # TODO: Write a proper test
        features, labels = read_train_data("../../datasets/synthetic")

        self.assertIsNotNone(features)
        self.assertIsNotNone(labels)
