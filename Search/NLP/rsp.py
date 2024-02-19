# code snipppet from https://learn.deeplearning.ai/building-applications-vector-databases

#import warnings
#warnings.filterwarnings('ignore')

#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from openai import OpenAI
#from pinecone import Pinecone, ServerlessSpec
#from tqdm.auto import tqdm, trange
#from dotenv import load_dotenv

#import pandas as pd
#import time
#import os

import nltk
nltk.download("universal_tagset")