# Netflix Movie Recommendation System

A portfolio-ready collaborative-filtering project built from Netflix Prize-style ratings data.  
The project analyzes rating behavior, creates a reproducible modeling subset, trains a latent-factor recommendation model, evaluates prediction quality, and generates personalized movie recommendations.

## Project Highlights

- **24,053,764 ratings**
- **470,758 customers**
- **4,499 rated movies** in `combined_data_1.txt`
- Average rating: **3.60 / 5**
- Latent-factor collaborative filtering using a **FunkSVD-style matrix-factorization model**
- Validation **RMSE: 0.991**
- Validation **MAE: 0.795**
- Personalized Top-N movie recommendation workflow

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Repository Structure

```text
netflix-movie-recommendation-system/
├── data/
│   └── README.md
├── notebooks/
│   └── netflix_recommendation_system.ipynb
├── src/
│   └── recommender.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

The analysis uses Netflix Prize-style data:

- `combined_data_1.txt`
- `movie_titles.csv`

The raw ratings file is intentionally not stored in GitHub because it is large.  
Place the files inside `data/` before rerunning the complete workflow.

## Analysis Workflow

1. Validate the Netflix ratings source
2. Inspect rating distribution and overall dataset scale
3. Create a deterministic modeling sample from the full ratings file
4. Retain sufficiently active users and movies
5. Split ratings into training and validation sets
6. Train a bias-aware latent-factor model
7. Evaluate the model using RMSE and MAE
8. Rank unseen movies for a selected customer
9. Produce personalized Top-N recommendations

## Model Experiment

To keep the portfolio notebook reproducible on a standard laptop, the modeling stage uses a deterministic sample of the full ratings file. After activity filtering, the experiment contains:

| Metric | Value |
|---|---:|
| Modeling ratings | 28,596 |
| Customers | 4,946 |
| Movies | 1,659 |
| Training ratings | 22,876 |
| Validation ratings | 5,720 |
| RMSE | 0.991 |
| MAE | 0.795 |

The notebook clearly separates **full-dataset descriptive statistics** from the **sampled modeling experiment**.

## Example Recommendations

For one active customer in the modeling sample, the system produced recommendations including:

- Lost: Season 1
- Lord of the Rings: The Fellowship of the Ring
- The Silence of the Lambs
- Finding Nemo (Widescreen)
- Six Feet Under: Season 4

Recommendations are based on predicted preference scores for movies the selected customer had not rated in the modeling subset.

## Skills Demonstrated

- data wrangling
- large-file processing
- exploratory data analysis
- recommender systems
- collaborative filtering
- matrix factorization
- train/validation evaluation
- RMSE and MAE
- ranking unseen items
- reproducible Python project structure

## Limitations

- The modeling experiment uses a sample of the complete ratings file for practical runtime.
- Explicit rating prediction does not use movie metadata such as genre, cast, or plot.
- RMSE and MAE measure rating prediction error, not ranking quality.
- Production recommendation systems should additionally evaluate metrics such as Precision@K, Recall@K and coverage.

## Future Improvements

- Precision@K / Recall@K evaluation
- hybrid collaborative + content-based recommendation
- hyperparameter tuning
- implicit-feedback modeling
- interactive recommendation application

## Author

**Akanksha Nandre**  
Data Analyst | Business Intelligence | Power BI | SQL | Python
