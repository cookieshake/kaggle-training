from bootstrap import SelectColumn

import luigi
import pandas as pd
import spacy


class FTCompareNoun(luigi.Task):
    def requires(self):
        return SelectColumn()

    def output(self):
        return luigi.LocalTarget('./output/ft_compare_noun.pickle')

    def run(self):
        input_df = pd.read_pickle(self.input().path)
        nlp = spacy.load('en')

        input_df['compare_noun'] = 0.0
        all = len(input_df)
        for i, row in input_df.iterrows():
            q1 = str(row['question1'])
            q2 = str(row['question2'])

            qt1 = [t for t in nlp(q1) if t.tag_.startswith('N')]
            qt2 = [t for t in nlp(q2) if t.tag_.startswith('N')]

            count = 0
            s_sum = 0
            for t1 in qt1:
                for t2 in qt2:
                    s_sum += t1.similarity(t2)
                    count += 1

            row['compare_noun'] = s_sum / count if count != 0 else 0

            if i % 1000 == 0:
                self.set_status_message('Done: {:.2f}%'.format(i / all * 100))

        input_df['compare_noun'].to_pickle(self.output().path)


class FTLength(luigi.Task):
    def requires(self):
        return SelectColumn()

    def output(self):
        return luigi.LocalTarget('./output/ft_length.pickle')

    def run(self):
        input_df = pd.read_pickle(self.input().path)
        nlp = spacy.load('en')

        input_df['compare_length'] = 0.0
        all = len(input_df)
        for i, row in input_df.iterrows():
            q1 = str(row['question1'])
            q2 = str(row['question2'])

            if len(q2) > len(q1):
                q1, q2 = q2, q1

            row['compare_length'] = len(q2) / len(q1) if len(q1) != 0 else 0

            if i % 1000 == 0:
                self.set_status_message('Done: {:.2f}%'.format(i / all * 100))

        input_df['compare_length'].to_pickle(self.output().path)


class AggregateFeatures(luigi.Task):
    def requires(self):
        return SelectColumn(), FTCompareNoun(), FTLength()

    def output(self):
        return luigi.LocalTarget('./output/aggregated.pickle')

    def run(self):
        input_df = pd.read_pickle(self.input()[0].path)
        compare_n = pd.read_pickle(self.input()[1].path)
        compare_l = pd.read_pickle(self.input()[2].path)

        del input_df['question1']
        del input_df['question2']

        df = pd.concat([input_df, compare_n, compare_l], axis=1)
        df.to_pickle(self.output().path)