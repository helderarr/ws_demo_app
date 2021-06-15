from pandas import DataFrame

from interfaces import PipelineStep


class PrintCenterEntity(PipelineStep):

    def __init__(self):
        super(PrintCenterEntity, self).__init__()
        self.center_entity = None

    def run(self, data: DataFrame) -> DataFrame:

        ents, scores = self.merge_entities(data)

        if len(ents) == 0:
            print("No entities found.")
            self.center_entity = None
            return data

        max_score = 0
        entity_with_max_score = None
        for ent in ents:
            if scores[ent] > max_score:
                entity_with_max_score = ent
                max_score = scores[ent]

        self.center_entity = ents[entity_with_max_score]
        print("Center Entity:",self.center_entity)

        return data


    def merge_entities(self,data:DataFrame):
        d ={}
        scores ={}
        for index, row in data.iterrows():
            row_entities = row['entities']
            for x in row_entities:
                d[x[0]] = x[1]
                scores[x[0]] = x[2]

        return d,scores

