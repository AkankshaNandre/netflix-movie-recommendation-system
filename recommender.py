from pathlib import Path

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate


ROOT = Path(__file__).resolve().parents[1]
RATINGS_PATH = ROOT / "data" / "combined_data_1.txt"
TITLES_PATH = ROOT / "data" / "movie_titles.csv"


def parse_netflix_ratings(path: Path) -> pd.DataFrame:
    """Parse Netflix Prize-style ratings into a tidy DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Ratings file not found at {path}. "
            "See data/README.md for the expected files."
        )

    rows = []
    current_movie_id = None

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.endswith(":"):
                current_movie_id = int(line[:-1])
                continue

            parts = line.split(",")
            if len(parts) < 2 or current_movie_id is None:
                continue

            customer_id = int(parts[0])
            rating = float(parts[1])

            rows.append((customer_id, current_movie_id, rating))

    return pd.DataFrame(rows, columns=["Cust_Id", "Movie_Id", "Rating"])


def load_titles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Movie-title file not found at {path}. "
            "See data/README.md for the expected files."
        )

    titles = pd.read_csv(
        path,
        encoding="ISO-8859-1",
        header=None,
        names=["Movie_Id", "Year", "Name"],
        on_bad_lines="skip",
    )

    titles["Movie_Id"] = pd.to_numeric(titles["Movie_Id"], errors="coerce")
    titles = titles.dropna(subset=["Movie_Id"]).copy()
    titles["Movie_Id"] = titles["Movie_Id"].astype(int)

    return titles


def filter_sparse_activity(
    ratings: pd.DataFrame,
    quantile: float = 0.70,
) -> tuple[pd.DataFrame, int, int]:
    """
    Remove users and movies below activity thresholds derived from
    the selected count quantile.
    """
    movie_counts = ratings.groupby("Movie_Id")["Rating"].count()
    user_counts = ratings.groupby("Cust_Id")["Rating"].count()

    movie_threshold = int(round(movie_counts.quantile(quantile)))
    user_threshold = int(round(user_counts.quantile(quantile)))

    active_movies = movie_counts[movie_counts >= movie_threshold].index
    active_users = user_counts[user_counts >= user_threshold].index

    filtered = ratings[
        ratings["Movie_Id"].isin(active_movies)
        & ratings["Cust_Id"].isin(active_users)
    ].copy()

    return filtered, movie_threshold, user_threshold


def build_surprise_dataset(
    ratings: pd.DataFrame,
    max_rows: int | None = None,
):
    sample = ratings
    if max_rows is not None and len(ratings) > max_rows:
        sample = ratings.sample(max_rows, random_state=42)

    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(
        sample[["Cust_Id", "Movie_Id", "Rating"]],
        reader,
    )


def evaluate_svd(ratings: pd.DataFrame, max_rows: int = 100_000) -> pd.DataFrame:
    data = build_surprise_dataset(ratings, max_rows=max_rows)

    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
    )

    cv = cross_validate(
        model,
        data,
        measures=["RMSE", "MAE"],
        cv=3,
        verbose=False,
    )

    return pd.DataFrame(
        {
            "Fold": np.arange(1, len(cv["test_rmse"]) + 1),
            "RMSE": cv["test_rmse"],
            "MAE": cv["test_mae"],
        }
    )


def train_final_model(ratings: pd.DataFrame) -> SVD:
    data = build_surprise_dataset(ratings)

    trainset = data.build_full_trainset()

    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
    )
    model.fit(trainset)

    return model


def recommend_for_user(
    model: SVD,
    ratings: pd.DataFrame,
    titles: pd.DataFrame,
    customer_id: int,
    top_n: int = 10,
) -> pd.DataFrame:
    rated_movies = set(
        ratings.loc[
            ratings["Cust_Id"] == customer_id,
            "Movie_Id",
        ]
    )

    candidates = titles[
        ~titles["Movie_Id"].isin(rated_movies)
    ].copy()

    candidates["Estimated_Rating"] = candidates["Movie_Id"].apply(
        lambda movie_id: model.predict(customer_id, movie_id).est
    )

    return (
        candidates[
            ["Movie_Id", "Name", "Year", "Estimated_Rating"]
        ]
        .sort_values("Estimated_Rating", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def main():
    ratings = parse_netflix_ratings(RATINGS_PATH)
    titles = load_titles(TITLES_PATH)

    print("Raw ratings shape:", ratings.shape)
    print("Unique customers:", ratings["Cust_Id"].nunique())
    print("Unique movies:", ratings["Movie_Id"].nunique())
    print("\nRatings distribution:")
    print(ratings["Rating"].value_counts().sort_index())

    filtered, movie_threshold, user_threshold = filter_sparse_activity(ratings)

    print("\nActivity thresholds")
    print("Movie rating-count threshold:", movie_threshold)
    print("Customer rating-count threshold:", user_threshold)
    print("Filtered ratings shape:", filtered.shape)

    cv_results = evaluate_svd(filtered)
    print("\nCross-validation results")
    print(cv_results.to_string(index=False))
    print("\nMean metrics")
    print(cv_results[["RMSE", "MAE"]].mean().round(4))

    model = train_final_model(filtered)

    sample_customer = int(filtered["Cust_Id"].value_counts().index[0])
    recommendations = recommend_for_user(
        model,
        filtered,
        titles,
        customer_id=sample_customer,
        top_n=10,
    )

    print(f"\nTop recommendations for customer {sample_customer}")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
    
