import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# path = os.path.join("../../..", "data/laboration_1/ml-latest/")

def load_data(data_path):
    movies_path = os.path.join(data_path, "movies.csv")
    tag_path = os.path.join(data_path, "tags.csv")
    ratings_path = os.path.join(data_path, "ratings.csv")
    links_path = os.path.join(data_path, "links.csv")

    df_movies = pd.read_csv(movies_path)
    df_ratings = pd.read_csv(ratings_path)
    df_links = pd.read_csv(links_path)
    df_tags = pd.read_csv(tag_path)

    return df_movies, df_tags

def data_prep(df_movies, df_tags):
    df_tags = df_tags.dropna(subset=["tag"]).copy()

    df_tags_grouped = (df_tags.groupby("movieId")["tag"]
                                            .apply(lambda x: " ".join(x.astype(str)))
                                            .reset_index()
                                            )

    df_movies["genres"] = df_movies["genres"].str.replace("|", " ", regex=False)

    df_movies = df_movies.merge(df_tags_grouped, on="movieId", how="left")
    df_movies["tag"] = df_movies["tag"].fillna("")

    df_movies["features"] = df_movies["genres"] + " " + df_movies["tag"]
    df_movies = df_movies[df_movies["features"] != "(no genres listed) "].copy()

    df_movies["genres"] = df_movies["genres"].replace(
        "(no genres listed)",
        ""
    )

    df_movies["features"] = (df_movies["genres"] + " " + df_movies["tag"]).str.strip()

    df_movies = df_movies.reset_index(drop=True)

    return df_movies


def build_model(df_movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_movies["features"])
    
    indices = pd.Series(
        df_movies.index,
        index = df_movies["title"]
    ).drop_duplicates()
    

    return tfidf_matrix, indices   


def recommend(movie_title, df_movies, tfidf_matrix, indices, n=5):
    if movie_title not in indices:
        matches = df_movies[
            df_movies["title"].str.contains(movie_title, case=False, na=False)
        ][["title", "genres"]].head(10)

        return f"Movie not found. Did you mean one of these?\n{matches}"
    
    idx = indices[movie_title]

    similarities = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = similarities.argsort()[::-1]
    similar_indices = similar_indices[similar_indices != idx]

    top_indices = similar_indices[:n]

    return df_movies.iloc[top_indices][["title", "genres"]]

def main():
    parser = argparse.ArgumentParser(
        description="Movie recommender using TF-IDF and cosine similarity"
    )

    parser.add_argument(
        "movie",
        type=str,
        help="Movie title, for exemple: 'Toy Story (1995)'"
    )

    parser.add_argument(
        "-n",
        type=int,
        default=5,
        help="Number of recommendations"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="../../../data/laboration_1/ml-latest/",
        help="Path to the dataset folder."
    )

    args = parser.parse_args()

    movies, tags = load_data(args.data_path)
    movies = data_prep(movies, tags)
    tfidf_matrix, indices = build_model(movies)

    result = recommend(
        args.movie,
        movies,
        tfidf_matrix,
        indices,
        n=args.n
    )

    print(result)


if __name__ == "__main__":
    main()
