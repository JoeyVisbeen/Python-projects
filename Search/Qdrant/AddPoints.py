from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

client = QdrantClient(host='localhost', port='6333')

my_collection = 'first_collection'
first_collection = client.recreate_collection(
    collection_name=my_collection,
    vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE)
)

collection_info = client.get_collection(collection_name=my_collection)

#for i in list(collection_info):
#    print(i)

data = np.random.uniform(low=-1.0 ,high=1.0 ,size=(1_000, 100))
print(data)

type(data[0,0]), data[:2, :20]

#for i in range(len(data[0])):
#    print(i)

index = list(range(len(data)))
#print(index[-10:])

client.upsert(
    collection_name=my_collection,
    points=models.Batch(
        ids=index,
        vectors=data.tolist()
    )
)

retrieved = client.retrieve(
    collection_name=my_collection,
    ids=[100],
    with_vectors=True
)
#print(retrieved[0])
