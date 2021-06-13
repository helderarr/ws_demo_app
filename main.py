import streamlit as st
import pandas as pd
import numpy as np

from BlinkEntityExtractor import BlinkEntityExtractor
from DbPediaSpotlightAnnotator import DbPediaSpotlightAnnotator
from DbPediaSpotlightEntityExtractor import DbPediaSpotlightEntityExtractor
from PageRankPassageRanker import PageRankPassageRanker
from PassageReader import PassageReader
from streamlit_agraph import agraph, Node, Edge, Config

from PrintCenterEntity import PrintCenterEntity
from SumarizerStep import SumarizerStep
from TopNExceptPassagesBoosted import TopNExceptPassagesBoosted
from TopNPassages import TopNPassages
from interfaces import Pipeline

st.title('Conversations answers and alternatives')
left, right = st.beta_columns(2)

passage_reader = PassageReader()

conversation = st.sidebar.radio('Conversation',passage_reader.get_conversations()[1:17])

utterances_list = passage_reader.get_utterances(conversation)

if conversation == 69:
    utterances_list = utterances_list[0:5]

utterance = st.sidebar.radio('Conversation',utterances_list)

print(utterance)

conversation_utterance_id = f"{conversation}_{utterance}"

passages = passage_reader.get_utterance_passages(conversation_utterance_id)

main_n_passages = 3
alternalive_n_passages = 2

dbPediaSpotlightEntityExtractor = DbPediaSpotlightEntityExtractor()
sumarizerStep = SumarizerStep(50, 80)

top_n_passages = TopNPassages(main_n_passages)
print_center_entity = PrintCenterEntity()
page_rank_passage_ranker = PageRankPassageRanker()

pipeline_dbpedia = Pipeline()
pipeline_dbpedia.add_step(dbPediaSpotlightEntityExtractor)
pipeline_dbpedia.add_step(page_rank_passage_ranker)
#pipeline_dbpedia.add_step(print_center_entity)
pipeline_dbpedia.add_step(top_n_passages)
pipeline_dbpedia.add_step(sumarizerStep)

df = pipeline_dbpedia.run(passages)

left.header(df[1])
left.write(df[2])
left.dataframe()

def format(data):
    if data["int_rank"] <= main_n_passages:
        return ['text-align: left;background-color: grey' for v in data]
    else :
        return ['text-align: left' for v in data]



passages_list = pipeline_dbpedia.last_run[2][["int_rank","passage"]]
passages_list = passages_list.style.apply(format,axis=1)
right.table(passages_list)

nodes = []
edges = []

for node in page_rank_passage_ranker.order:
    nodes.append(Node(id=node[0],label=node[1]))

cx = page_rank_passage_ranker.graph.tocoo()
for i, j, v in zip(cx.row, cx.col, cx.data):
    if j < i:
        edges.append(Edge(source=page_rank_passage_ranker.order[i][0], target=page_rank_passage_ranker.order[j][0], type="CURVE_SMOOTH"))

config = Config(#width=500,
                #height=500,
                directed=False,
                nodeHighlightBehavior=True,
                highlightColor="blue",
                collapsible=True,
                node={'labelProperty':'label'},
                link={'labelProperty': 'label', 'renderLabel': False},
                staticGraphWithDragAndDrop=True
                # **kwargs e.g. node_size=1000 or node_color="blue"
                )

return_value = agraph(nodes=nodes,
                      edges=edges,
                      config=config)

dbPediaSpotlightAnnotator = DbPediaSpotlightAnnotator()
blinkEntityExtractor = BlinkEntityExtractor()

pipeline_blink_alt_not_boosted_filtered = Pipeline()
pipeline_blink_alt_not_boosted_filtered.add_step(dbPediaSpotlightAnnotator)
pipeline_blink_alt_not_boosted_filtered.add_step(blinkEntityExtractor)
pipeline_blink_alt_not_boosted_filtered.add_step(PageRankPassageRanker())
pipeline_blink_alt_not_boosted_filtered.add_step(TopNExceptPassagesBoosted(main_n_passages,boost_factor=1,keep_filtered_entities=False))
pipeline_blink_alt_not_boosted_filtered.add_step(PageRankPassageRanker())
pipeline_blink_alt_not_boosted_filtered.add_step(PrintCenterEntity())
pipeline_blink_alt_not_boosted_filtered.add_step(TopNPassages(alternalive_n_passages))
pipeline_blink_alt_not_boosted_filtered.add_step(sumarizerStep)

#df_alternative = pipeline_blink_alt_not_boosted_filtered.run(passages)