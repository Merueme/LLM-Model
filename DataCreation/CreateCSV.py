import json
import pandas as pd

with open("D:/YNOV/M1/NLP/listTag.json", "r") as file:
    tags = json.load(file)

columns = []

for tag in tags['data'] :
    columns.append(tag['attributes']['name']['en'])

columns.sort()

columns = ['Description'] + columns

df = pd.DataFrame(data=[], columns=columns)

df.to_csv("D:/YNOV/M1/NLP/CSVMangaTag.csv", index=False)