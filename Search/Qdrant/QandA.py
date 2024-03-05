import datetime
from dotenv import load_dotenv
import os
import random
from pathlib import Path
from typing import AnyStr

import pandas as pd
from datasets import load_dataset
from IPython.display import Markdown, display_markdown
from llama_index.core import (VectorStoreIndex, ServiceContext, SimpleDirectoryReader)
from llama_index.core.postprocessor import FixedRecencyPostprocessor
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
load_dotenv()


def check_environment_keys():
    """
    Utility Function that you have the NECESSARY Keys
    """
    if os.getenv('OPENAI_APIKEY') is None:
        raise ValueError(
            "OPENAI_API_KEY cannot be None. Set the key using os.environ['OPENAI_API_KEY']='sk-xxx'"
        )
    if os.getenv('COHERE_APIKEY') is None:
        raise ValueError(
            "COHERE_API_KEY cannot be None. Set the key using os.environ['COHERE_API_KEY']='xxx'"
        )
    if os.getenv('QDRANT_APIKEY') is None:
        print("[Optional] If you want to use the Qdrant Cloud, please get the Qdrant Cloud API Keys and URL")


check_environment_keys()

dataset = load_dataset('heegyu/news-category-dataset', split='train')
#print(dataset)
def get_single_text(k):
        return f"Under the category:\n{k['category']}:\n{k['headline']}\n{k['short_description']}"

df = pd.DataFrame(dataset)
#print(df.head())

df['year'] = df['date'].dt.year
category_columns_to_keep = ['POLITICS', 'THE WORLDPOST', 'WORLD NEWS', 'WORLDPOST', 'U.S. NEWS']

df_filtered = df[df['category'].isin(category_columns_to_keep)]

def sample_func(x):
    return x.sample(min(len(x), 200), random_state=42)

print(df_filtered.groupby('year', group_keys=True).apply(sample_func).reset_index(drop=True))
#df_sampled = df_filtered.groupby('year').apply(sample_func)#.reset_index(drop=True)
#print(df_sampled['year'].value_counts())
#del df
#df = df_sampled

#print(df['text'][9])