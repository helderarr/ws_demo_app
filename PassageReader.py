from os import path
from PassageRetriever import PassageRetriever
import pandas as pd

from UtteranceRetriever import UtteranceRetirever


class PassageReader:

    def __init__(self):
        self.file_name = "data/dataset.csv"
        self.columns = ["conversation_utterance_id", "conversation_id", "utterance_id",
                        "utterance", "rank", "score", "passage","passage_id"]
        self.data = self.load_data()


    def get_conversations(self):
        return pd.unique(self.data["conversation_id"])

    def get_utterances(self,conversation):
        conv = self.data[self.data["conversation_id"] == conversation]
        conv = conv.copy()
        conv["label"] = conv["utterance_id"] #+ " - " + conv["utterance"]
        conv = pd.unique(conv["label"])
        return conv


    def load_data(self):
        if not path.exists(self.file_name):
            self.extract_data()

        return pd.read_csv(self.file_name,
                           names=self.columns)

    def get_utterance_passages(self, conversation_utterance_id: str):
        return self.data[self.data["conversation_utterance_id"] == conversation_utterance_id]

    def extract_data(self):
        with PassageRetriever() as retriever:
            df = pd.read_csv('data/anserini_test_lmd_1000_car_marco_wapo_2019_bert_1000_CORIG.run',
                             names=["conversation_utterance_id", "NA", "passage_id", "gobal_rank", "score", "dataset"],
                             header=None, delimiter=" ")

            marco_df = df[df["passage_id"].str.startswith('MARCO_', na=False)]

            marco_df['rank'] = marco_df.groupby('conversation_utterance_id')['gobal_rank'].rank(method='first')

            top_10_df = marco_df[marco_df["rank"] < 10]

            top_10_df["passage"] = top_10_df.apply(lambda row: retriever.get(row["passage_id"]), axis=1)

            top_10_df = top_10_df[top_10_df["passage"].notnull()]

            top_10_df["conversation_id"] = top_10_df.apply(lambda row: row["conversation_utterance_id"].split("_")[0],
                                                           axis=1)

            top_10_df["utterance_id"] = top_10_df.apply(lambda row: row["conversation_utterance_id"].split("_")[1],
                                                        axis=1)

            utterances = UtteranceRetirever()

            top_10_df["utterance"] = top_10_df \
                .apply(lambda row: utterances.get_utterance(row["conversation_utterance_id"]), axis=1)

            top_10_df["passage"] = top_10_df["passage"].apply(lambda row: [x.strip() for x in row.split("\n") if len(x.strip())>0])

            top_10_df = top_10_df.explode("passage")

            top_10_df.to_csv(self.file_name, header=True, index=False,
                             columns=self.columns)
