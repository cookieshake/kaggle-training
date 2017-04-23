import luigi
import pandas as pd
import numpy as np


class SelectColumn(luigi.Task):
    def output(self):
        return luigi.LocalTarget('./output/column_selected.pickle')

    def run(self):
        dtypes ={
            'question1': np.str,
            'question2': np.str,
            'is_duplicate': np.uint8
        }

        df = pd.read_csv('../../dataset/quora_train.csv', dtype=dtypes,
                         usecols=list(dtypes.keys()))
        df.to_pickle(self.output().path)


class LoadTestSet(luigi.Task):
    def output(self):
        return luigi.LocalTarget('./output/test_set.pickle')

    def run(self):
        dtypes ={
            'id': np.uint32,
            'question1': np.str,
            'question2': np.str,
        }

        df = pd.read_csv('../../dataset/quora_test.csv', dtype=dtypes,
                         usecols=list(dtypes.keys()))
        df = df[['id', 'question1', 'question2']]
        df.to_hdf(self.output().path, 'test')