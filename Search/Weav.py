import weaviate
import json

from dotenv import load_dotenv
import os
load_dotenv()

weavApiKey = os.getenv('WEAV_APIKEY')
cohereApiKey = os.getenv('COHERE_APIKEY')
openAiApiKey = os.getenv('OPENAI_APIKEY')
sandboxJvUrl = os.getenv('WEAV_SANDURL')

eduDemoUrl = 'https://edu-demo.weaviate.network'
eduDemoApiKey = 'readonly-demo'

authConfig = weaviate.auth.AuthApiKey(api_key=eduDemoApiKey) 

client = weaviate.Client(
    url=eduDemoUrl,
    auth_client_secret=authConfig,
    additional_headers={
        "X-Cohere-Api-Key": cohereApiKey,
        "X-OpenAI-Api-Key": openAiApiKey
    }
)

#meta_info = client.get_meta()
#print(json.dumps(meta_info, indent=2))

#ask = {
#    "question": "", 
#    "properties": [""]
#}

res = client.query.get(
    "WikiCity", ["city_name", "country", "lng", "lat"]
).with_near_text({
    "concepts": ["Major European city"]
}).with_limit(5).do()

print(json.dumps(res, indent=2))
