# Spotify Anomaly Detection: Identifying Mislabeled Genres

## Project Overview
This project addresses the problem of identifying musical tracks that do not belong to the "Classical" genre within a mislabeled dataset. Using unsupervised learning techniques, I developed a pipeline to detect Global Outliers—tracks (such as Metal or Rock) that possess audio characteristics fundamentally different from traditional classical music.

## Dataset
- Source: Spotify Tracks Dataset from Kaggle.
- Content: Audio features including acousticness, energy, loudness, and instrumentalness.
- Scope: Focused on Western Classical music after applying a noise removal filter for non-Western artists.

## Methodology
The core of the project relies on strategic Feature Engineering. Based on exploratory data analysis, I applied custom weights to the feature vectors to maximize the separation between genres:
- Acousticness (5.0) and Instrumentalness (4.0) were prioritized as primary discriminators.
- Energy (3.0) and Loudness (4.0) were used to penalize modern compressed audio.
- Tempo (0.05) was minimized to prevent fast-paced classical pieces from being misclassified as anomalies.

## Models and Results
I implemented and compared four algorithms from the PyOD library:
1. KNN (K-Nearest Neighbors): Best performance with a Silhouette Score of 0.6544.
2. Isolation Forest: Highly robust for isolating global outliers like Power Metal tracks.
3. ECOD: Probabilistic approach focusing on distribution tails.
4. One-Class SVM: Boundary-based detection using an RBF kernel.

The analysis confirms that geometric and isolation-based methods are superior for this specific type of audio outlier detection.

## Installation
1. Clone the repository.
2. Install dependencies: pip install pandas numpy matplotlib seaborn scikit-learn pyod.
3. Run the Ivan_Vlad_Daniel.ipynb notebook.

## License
MIT License
