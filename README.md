# Netflix Movie Recommendation System

A collaborative filtering recommendation system built with Python and Singular Value Decomposition (SVD) using Netflix ratings data.

## Project Overview

The goal of this project is to analyze user-movie rating behavior and build a personalized movie recommendation system. The project uses collaborative filtering with SVD to estimate user preferences and recommend movies that a user has not previously rated.

## Objectives

- Explore Netflix user and movie rating behavior
- Analyze rating distributions and user/movie activity
- Prepare and filter sparse rating data
- Build a collaborative filtering recommendation model
- Train an SVD-based recommendation system
- Evaluate model performance using RMSE and MAE
- Generate personalized Top-N movie recommendations

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Surprise
- Jupyter Notebook

## Project Workflow

1. Load and parse Netflix ratings data
2. Perform exploratory data analysis
3. Analyze user and movie activity
4. Filter low-activity users and movies
5. Prepare data for collaborative filtering
6. Train an SVD recommendation model
7. Evaluate the model using cross-validation
8. Measure RMSE and MAE
9. Generate personalized movie recommendations

## Machine Learning Approach

The recommendation engine uses **Singular Value Decomposition (SVD)**, a matrix-factorization technique commonly used in collaborative filtering.

Instead of relying on movie genres or descriptions, the model learns patterns from user ratings to estimate how a user may rate movies they have not yet seen.

## Model Evaluation

The SVD model is evaluated using cross-validation with:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

Actual evaluation results are generated when the notebook is executed and are not hard-coded in this repository.

## Repository Structure

```text
netflix-movie-recommendation-system/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── netflix_recommendation_system.ipynb
│
├── src/
│   └── recommender.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Key Skills Demonstrated

- Data Cleaning and Preparation
- Exploratory Data Analysis
- Recommendation Systems
- Collaborative Filtering
- Matrix Factorization
- SVD
- Model Evaluation
- Cross-Validation
- Personalized Recommendation Ranking
- Python Data Analysis

## Future Improvements

Potential extensions include:

- Precision@K and Recall@K evaluation
- movie metadata and genre features
- hybrid collaborative/content-based recommendations
- recommendation-system deployment
- interactive recommendation dashboard

## Author

**Akanksha Nandre**

Data Analyst | Business Intelligence | Power BI | SQL | Python
