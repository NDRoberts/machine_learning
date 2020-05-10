# Test Dingle
import os
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from NN_Predictor import NN_Predictor


class Test_NN_Predictor(unittest.TestCase):

    def setUp(self):
        phony_csv = np.array([["field1,field2,field3,field4,class"],
                             ["0,1,0,1,A"],
                             ["1,0,1,0,B"],
                             ["1,0,1,0,A"],
                             ["0,1,0,1,B"],
                             ["1,0,1,0,C"]])
        self.ersatz_data = pd.DataFrame(phony_csv)

    def test_init(self):
        with self.assertRaises(Exception, msg="A data set must be supplied to"
                               + "create a Predictor."):
            _ = NN_Predictor()
        with self.assertRaises(FileNotFoundError):
            _ = NN_Predictor("Not A File")
        with patch('pandas.read_csv') as mock_reader:
            _ = NN_Predictor("file.csv")
            mock_reader.assert_called_with(f"{os.getcwd()}/file.csv")

    def test_split_data(self):
        print(self.ersatz_data)
