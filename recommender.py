from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
RATINGS_PATH = ROOT / "data" / "combined_data_1.txt"
TITLES_PATH = ROOT / "data" / "movie_titles.csv"


def sample_netflix_ratings(path: Path, every_n: int = 100) -> pd.DataFrame:
    """Create a deterministic sample from Netflix Prize-style ratings data."""
    if not path.exists():
        raise FileNotFoundError(f"Missing ratings file: {path}")

    rows = []
    movie_id = None
    rating_number = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.endswith(":"):
                movie_id = int(line[:-1])
                continue

            parts = line.split(",", 2)
            rating_number += 1

            if rating_number % every_n == 0:
                rows.append((int(parts[0]), movie_id, float(parts[1])))

    return pd.DataFrame(rows, columns=["Cust_Id", "Movie_Id", "Rating"])


def parse_titles(path: Path) -> pd.DataFrame:
    """Parse movie_titles.csv safely when titles themselves contain commas."""
    if not path.exists():
        raise FileNotFoundError(f"Missing title file: {path}")

    rows = []
    with path.open("r", encoding="latin-1") as file:
        for line in file:
            parts = line.rstrip("\n\r").split(",", 2)
            if len(parts) != 3:
                continue
            try:
                movie_id = int(parts[0])
            except ValueError:
                continue
            rows.append((movie_id, parts[1] or None, parts[2]))

    return pd.DataFrame(rows, columns=["Movie_Id", "Year", "Name"])


def prepare_model_data(ratings: pd.DataFrame) -> pd.DataFrame:
    user_counts = ratings["Cust_Id"].value_counts()
    active_users = user_counts[user_counts >= 5].index

    filtered = ratings[ratings["Cust_Id"].isin(active_users)].copy()

    movie_counts = filtered["Movie_Id"].value_counts()
    active_movies = movie_counts[movie_counts >= 3].index

    return filtered[filtered["Movie_Id"].isin(active_movies)].reset_index(drop=True)


class MatrixFactorization:
    """Bias-aware latent-factor collaborative-filtering model."""

    def __init__(
        self,
        n_factors=20,
        n_epochs=20,
        learning_rate=0.01,
        regularization=0.05,
        random_state=42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state

    def fit(self, df: pd.DataFrame):
        self.user_values = df["Cust_Id"].unique()
        self.item_values = df["Movie_Id"].unique()

        self.user_map = {v: i for i, v in enumerate(self.user_values)}
        self.item_map = {v: i for i, v in enumerate(self.item_values)}

        u = df["Cust_Id"].map(self.user_map).to_numpy(np.int32)
        i = df["Movie_Id"].map(self.item_map).to_numpy(np.int32)
        r = df["Rating"].to_numpy(np.float32)

        indices = np.arange(len(df))
        train_idx, valid_idx = train_test_split(
            indices, test_size=0.20, random_state=self.random_state
        )

        rng = np.random.default_rng(self.random_state)
        self.global_mean = float(r[train_idx].mean())

        self.P = rng.normal(
            0, 0.05, (len(self.user_values), self.n_factors)
        ).astype(np.float32)
        self.Q = rng.normal(
            0, 0.05, (len(self.item_values), self.n_factors)
        ).astype(np.float32)

        self.user_bias = np.zeros(len(self.user_values), dtype=np.float32)
        self.item_bias = np.zeros(len(self.item_values), dtype=np.float32)

        lr = self.learning_rate

        for _ in range(self.n_epochs):
            rng.shuffle(train_idx)

            for row in train_idx:
                user = u[row]
                item = i[row]

                pred = (
                    self.global_mean
                    + self.user_bias[user]
                    + self.item_bias[item]
                    + float(self.P[user] @ self.Q[item])
                )
                error = float(r[row] - pred)

                p_old = self.P[user].copy()
                q_old = self.Q[item].copy()

                self.user_bias[user] += lr * (
                    error - self.regularization * self.user_bias[user]
                )
                self.item_bias[item] += lr * (
                    error - self.regularization * self.item_bias[item]
                )

                self.P[user] += lr * (
                    error * q_old - self.regularization * p_old
                )
                self.Q[item] += lr * (
                    error * p_old - self.regularization * q_old
                )

            lr *= 0.95

        valid_predictions = (
            self.global_mean
            + self.user_bias[u[valid_idx]]
            + self.item_bias[i[valid_idx]]
            + np.sum(self.P[u[valid_idx]] * self.Q[i[valid_idx]], axis=1)
        )
        valid_predictions = np.clip(valid_predictions, 1, 5)

        self.rmse = mean_squared_error(
            r[valid_idx], valid_predictions
        ) ** 0.5
        self.mae = mean_absolute_error(r[valid_idx], valid_predictions)

        return self

    def recommend(self, customer_id, ratings, titles, top_n=10):
        if customer_id not in self.user_map:
            raise ValueError("Customer is not present in the modeling sample.")

        user_idx = self.user_map[customer_id]
        seen = set(
            ratings.loc[
                ratings["Cust_Id"] == customer_id, "Movie_Id"
            ]
        )

        estimated = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias
            + self.Q @ self.P[user_idx]
        )
        estimated = np.clip(estimated, 1, 5)

        candidates = pd.DataFrame(
            {
                "Movie_Id": self.item_values,
                "Estimated_Rating": estimated,
            }
        )
        candidates = candidates[~candidates["Movie_Id"].isin(seen)]

        return (
            candidates.merge(titles, on="Movie_Id", how="left")
            .sort_values("Estimated_Rating", ascending=False)
            .head(top_n)
            [["Movie_Id", "Name", "Year", "Estimated_Rating"]]
            .reset_index(drop=True)
        )


def main():
    ratings = sample_netflix_ratings(RATINGS_PATH, every_n=100)
    titles = parse_titles(TITLES_PATH)
    model_data = prepare_model_data(ratings)

    model = MatrixFactorization().fit(model_data)

    print(f"Modeling rows: {len(model_data):,}")
    print(f"Customers: {model_data['Cust_Id'].nunique():,}")
    print(f"Movies: {model_data['Movie_Id'].nunique():,}")
    print(f"RMSE: {model.rmse:.4f}")
    print(f"MAE: {model.mae:.4f}")

    customer_id = int(model_data["Cust_Id"].value_counts().idxmax())
    print(f"\nRecommendations for customer {customer_id}:")
    print(
        model.recommend(
            customer_id, model_data, titles, top_n=10
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
