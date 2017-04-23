from feature import AggregateFeatures

import pickle

from sklearn.model_selection import cross_val_score
import luigi
import pandas as pd
import xgboost as xgb

class XGBoostModel(luigi.Task):
    def requires(self):
        return AggregateFeatures()

    def output(self):
        return luigi.LocalTarget('./output/xgboost_model.pickle')

    def run(self):
        df = pd.read_pickle(self.input().path)

        X = df.drop(['is_duplicate'], axis=1)
        y = df['is_duplicate']

        clf = xgb.XGBClassifier()
        clf.fit(X, y)

        score = cross_val_score(clf, X, y, cv=5)
        self.set_status_message(str(score))

        pickle.dump(clf, open(self.output().path, 'wb'))



