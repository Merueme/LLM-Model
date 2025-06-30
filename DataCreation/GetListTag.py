import requests
import json

base_url = "https://api.mangadex.org"

tags = requests.get(
    f"{base_url}/manga/tag"
).json()

with open("D:/YNOV/M1/NLP/listTag.json", "w") as file:
    json.dump(tags, file, indent=4) 
    print('done')
