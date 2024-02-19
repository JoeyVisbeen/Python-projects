print("test")
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(sys.path[1])

l = os.environ['PINECONE_APIKEY']

print(l)