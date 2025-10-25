Data Clustering and Exploratory Analysis App

A web-based Data Science tool for performing Exploratory Data Analysis (EDA) and density-based clustering.
Built with Dash, Flask, PostgreSQL, and Scikit-learn, this app provides an interactive environment to upload, explore, and analyze datasets — all in one place.

Features

Upload, create, and manage datasets through a clean web interface.

Connect to PostgreSQL databases:

db – stores user-created, uploaded, and modified datasets.

db2 – stores public, cleaned datasets available for analysis.

Perform clustering using a density-based algorithm (DBSCAN-style) on up to three numerical columns.

Export results as downloadable .csv files.

Merge data sources easily by changing the db2 variable to db in main.py.

Demo

A short demo of the application in action:

https://user-images.githubusercontent.com/105871709/211107465-35405023-6ab1-4899-bc65-9822f5d8b841.mp4

How It Works

Data Handling

Upload your own CSV dataset or load a random dataset from the public database.

Store datasets securely in PostgreSQL.

Exploratory Data Analysis

Preview your data, inspect summary statistics, and select columns for clustering.

Clustering

Apply a density-based clustering algorithm (inspired by DBSCAN).

Visualize the resulting clusters within the app.

Export clustered data to CSV for further analysis.

Tech Stack

Frontend: Dash (Plotly)

Backend: Flask

Database: PostgreSQL

Machine Learning: Scikit-learn

Language: Python

Configuration

By default, the app connects to two PostgreSQL databases:

Variable	Purpose
db	User-created, uploaded, and modified datasets
db2	Public pre-cleaned datasets for analysis

If you prefer to work with a single database, simply update the following line in main.py:

db2 = db


This will merge both data sources into one.

Planned Updates

Support higher-dimensional analysis using dimensionality reduction techniques (e.g., PCA, t-SNE).

Integrate evaluation metrics for cluster quality assessment.

Add a categorical encoder to allow mixed-type data clustering.

Project Goals

This project was built to demonstrate how a full-stack data science tool can integrate:

Interactive web frameworks (Dash/Flask)

Scalable database management (PostgreSQL)

Machine learning functionality (Scikit-learn)

It serves as a practical example of how to bridge data engineering, analytics, and visualization in a cohesive workflow.

Getting Started

Clone the repository

git clone https://github.com/yourusername/data-clustering-app.git
cd data-clustering-app


Install dependencies

pip install -r requirements.txt


Set up PostgreSQL databases

Create db and db2 according to your environment.

Update connection URIs in the configuration file or environment variables.

Run the app

python main.py


Access in browser

http://localhost:8050
