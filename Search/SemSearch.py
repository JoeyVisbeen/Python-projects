from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


import os
import time
import torch

from tqdm.auto import tqdm

dataset = load_dataset('quora', split='train[240000:290000]')

dataset[:5]

questions = []
for record in dataset['questions']:
    questions.extend(record['text'])
question = list(set(questions))
print('\n'.join(questions[:10]))
print('-' * 50)
print(f'Number of questions: {len(questions)}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print('Sorry no cuda.')
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

query = 'which city is the most populated in the world?'
xq = model.encode(query)
xq.shape


PINECONE_API_KEY = 'bd94b329-6ebf-4e79-bd13-d0d4a591f966'
