import requests
import pandas as pd
import time
from sklearn.preprocessing import MultiLabelBinarizer

# Constants
BASE_URL = "https://api.mangadex.org"
LIMIT = 100
LANGUAGES = ["en"]
CONTENT_RATINGS = ["safe", "suggestive"]
DEMOGRAPHICS = ["shounen", "shoujo", "josei", "seinen"]
STATUSES = ["ongoing", "completed", "hiatus", "cancelled"]

def get_genre_tags():
    tag_map = {}
    response = requests.get(f"{BASE_URL}/manga/tag")
    response.raise_for_status()

    for tag in response.json()['data']:
        if tag['attributes']['group'] != "genre":
            continue
        tag_id = tag['id']
        name = tag['attributes']['name'].get('en') or list(tag['attributes']['name'].values())[0]
        tag_map[tag_id] = name
    return tag_map


def get_manga_data():
    genre_tag_map = get_genre_tags()
    manga_info_list = []

    for rating in CONTENT_RATINGS:
        for demo in DEMOGRAPHICS:
            for status in STATUSES:
                offset = 0
                while True:
                    print(f"Fetching rating={rating}, demographic={demo}, status={status}, offset={offset}")
                    try:
                        response = requests.get(
                            f"{BASE_URL}/manga",
                            params={
                                "limit": LIMIT,
                                "offset": offset,
                                "publicationDemographic[]": demo,
                                "status[]": status,
                                "availableTranslatedLanguage[]": "en",
                                "contentRating[]": rating,
                                "includes[]": "tag",
                                "order[createdAt]": "asc"
                            },
                            timeout=10
                        )
                        response.raise_for_status()
                        data = response.json()
                        entries = data.get('data', [])
                        if not entries:
                            break

                        for manga in entries:
                            attributes = manga['attributes']
                            title = attributes['title'].get('en') or list(attributes['title'].values())[0]

                            desc_dict = attributes.get('description', {})
                            description = desc_dict.get('en') or (list(desc_dict.values())[0] if desc_dict else "")

                            genres = []
                            for tag in attributes.get('tags', []):
                                if tag['id'] in genre_tag_map:
                                    genres.append(genre_tag_map[tag['id']])

                            manga_info_list.append({
                                "title": title,
                                "description": description,
                                "status": status,
                                "demographic": demo,
                                "content_rating": rating,
                                "genres": genres
                            })

                        offset += LIMIT
                        time.sleep(0.5) 

                    except requests.exceptions.RequestException as e:
                        print(f"Request failed: {e} — Skipping combination.")
                        break 

    return manga_info_list


if __name__ == "__main__":
    mangas = get_manga_data()
    print(f"Total manga fetched before filtering: {len(mangas)}")

    print(mangas[:10])

    df = pd.DataFrame(mangas)

    print(df.head())

    
    df = df[df['description'].str.strip().astype(bool)]
    df = df[df['genres'].map(lambda g: len(g) > 0)]
    print(f"Total manga after filtering: {len(df)}")

    if df.empty:
        print("Warning: Final dataset is empty. No file saved.")
    else:
        
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(df['genres'])
        genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

        df = df.reset_index(drop=True)
        genre_df = genre_df.reset_index(drop=True)
        df = pd.concat([df[['title', 'description', 'status', 'demographic', 'content_rating']], genre_df], axis=1)


        df.to_csv("D:/YNOV/M1/NLP/Projet/DataSetMangaGenre.csv", index=False)
        print("✅ Dataset saved as DataSetMangaGenre.csv")
