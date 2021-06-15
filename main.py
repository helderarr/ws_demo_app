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

st.set_page_config(layout="wide",initial_sidebar_state="expanded")

passage_reader = PassageReader()

conversation = st.sidebar.radio('Conversation',passage_reader.get_conversations()[1:17])

utterances_list = passage_reader.get_utterances(conversation)

if conversation == 69:
    utterances_list = utterances_list[0:5]

utterance = st.sidebar.radio('Utterance',utterances_list)

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

dbPediaSpotlightAnnotator = DbPediaSpotlightAnnotator()
blinkEntityExtractor = BlinkEntityExtractor()

pipeline_dbpedia = Pipeline()
pipeline_dbpedia.add_step(dbPediaSpotlightAnnotator)
pipeline_dbpedia.add_step(blinkEntityExtractor)
pipeline_dbpedia.add_step(page_rank_passage_ranker)
pipeline_dbpedia.add_step(print_center_entity)
pipeline_dbpedia.add_step(top_n_passages)
pipeline_dbpedia.add_step(sumarizerStep)

df = pipeline_dbpedia.run(passages)

st.title(df[1])
left, right = st.beta_columns(2)

left.header(f"Center Entity: {print_center_entity.center_entity}")
left.write(df[2])


def format(data,n):
    if data["int_rank"] <= n:
        return ['text-align: left;background-color: grey' for v in data]
    else :
        return ['text-align: left' for v in data]


def format_topn(data):
    return format(data,main_n_passages)

def format_topn_except(data):
    return format(data,alternalive_n_passages)


passages_list = pipeline_dbpedia.last_run[3][["int_rank","passage"]]
passages_list = passages_list.style.apply(format_topn,axis=1)
left.table(passages_list)


dbPediaSpotlightAnnotator = DbPediaSpotlightAnnotator()
blinkEntityExtractor = BlinkEntityExtractor()
print_center_entity_alternative = PrintCenterEntity()
alternative_passage_ranker = PageRankPassageRanker()

pipeline_blink_alt_not_boosted_filtered = Pipeline()
pipeline_blink_alt_not_boosted_filtered.add_step(dbPediaSpotlightAnnotator)
pipeline_blink_alt_not_boosted_filtered.add_step(blinkEntityExtractor)
pipeline_blink_alt_not_boosted_filtered.add_step(PageRankPassageRanker())
pipeline_blink_alt_not_boosted_filtered.add_step(TopNExceptPassagesBoosted(main_n_passages,boost_factor=1,keep_filtered_entities=False))
pipeline_blink_alt_not_boosted_filtered.add_step(alternative_passage_ranker)
pipeline_blink_alt_not_boosted_filtered.add_step(print_center_entity_alternative)
pipeline_blink_alt_not_boosted_filtered.add_step(TopNPassages(alternalive_n_passages))
pipeline_blink_alt_not_boosted_filtered.add_step(sumarizerStep)

df_alternative = pipeline_blink_alt_not_boosted_filtered.run(passages)

right.header(f"Alternative Entity: {print_center_entity_alternative.center_entity}")
right.write(df_alternative[2])

passages_list_alt = pipeline_blink_alt_not_boosted_filtered.last_run[5][["int_rank","passage"]]
passages_list_alt = passages_list_alt.style.apply(format_topn_except,axis=1)
right.table(passages_list_alt)

question_option = "First answer"
alternative_option = "Alternative answer"
graph_type = st.sidebar.selectbox('Graph', [question_option,alternative_option])

nodes = []
edges = []

if graph_type==question_option:
    passage_ranker = page_rank_passage_ranker
else:
    passage_ranker = alternative_passage_ranker

for node in passage_ranker.order:
    nodes.append(Node(id=str(node[0]),label=node[1]))

cx = passage_ranker.graph.tocoo()
for i, j, v in zip(cx.row, cx.col, cx.data):
    if j < i:
        edges.append(Edge(source=str(passage_ranker.order[i][0]), target=str(passage_ranker.order[j][0]), type="CURVE_SMOOTH"))

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
