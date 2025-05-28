# Movie-Recommendation-ML-Project
This Jupyter Notebook implements a content-based movie recommendation system using TF-IDF vectorization and cosine similarity, leveraging movie metadata (genres, keywords, overview) to suggest similar films. Built with Python, pandas, scikit-learn, and NLTK.


## Project Overview

The notebook contains:

1. Import libraries for data processing (numpy, pandas), text processing (nltk, sklearn.feature_extraction.text), visualization (matplotlib, wordcloud), and similarity computation (sklearn.metrics.pairwise).

2. Loading and Understanding Data
Load the movie dataset (movies.csv) into a pandas DataFrame from a specified path.

   Inspect the data using df.head() to display the first five rows and df.shape to check the number of rows and columns, ensuring an understanding of the dataset's structure (e.g., columns like title, genres, keywords, overview).

3. Data Preprocessing
Combine relevant text features (e.g., genres, keywords, overview) into a single column (combined) for each movie.

   a. Define a preprocessing function (preprocess_text) using NLTK to:
   Convert text to lowercase.

   b. Remove special characters and numbers using regex.

   c. Tokenize text and remove stopwords to create clean text data.

   d. Apply the preprocessing function to the combined column, creating a new cleaned_text column.

4. Text Vectorization
Use TfidfVectorizer from scikit-learn to convert the cleaned_text into a TF-IDF matrix, limiting to the top 5000 features to capture significant words while managing dimensionality.

5. Computing Cosine Similarity
Calculate cosine similarity between all movie pairs using the TF-IDF matrix to create a similarity matrix (cosine_sim), which quantifies how similar movies are based on their text features.

6. Building the Recommendation Function
Define a function (recommend_movies) that:
Takes a movie title and returns the top N similar movies (default N=5).
Finds the index of the input movie in the DataFrame.

   Retrieves similarity scores from the cosine similarity matrix, sorts them in descending order, and selects the top N indices.

   Returns the titles of the top N similar movies.

7. Testing the Recommendation System
Test the recommendation function with an example movie (e.g., "Avengers: Age of Ultron") to verify it returns relevant movie recommendations based on the similarity scores.

## Dataset

This project uses a movie dataset from TMDB (metadata) and GroupLens (ratings/links). 

Download the dataset from:
- TMDB API: https://www.themoviedb.org/documentation/api
- GroupLens: https://grouplens.org/datasets/movielens/

Also included in Data/movies.csv

## Dataset Attribution
- Movie metadata is sourced from the TMDB Open API (https://www.themoviedb.org/documentation/api). This product uses the TMDB API but is not endorsed or certified by TMDB.
- Movie links and ratings are from GroupLens (https://grouplens.org/datasets/movielens/).


## Technologies Used

- **Python**: Core programming language (Python 3).
- **Jupyter Notebook**: Interactive environment for development and visualization.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: TF-IDF vectorization and cosine similarity computation.
- **NLTK**: Text preprocessing (stopwords removal, tokenization).
- **Matplotlib**: Data visualization (optional plots).
- **WordCloud**: Word frequency visualization (not used in current code).
- **Regular Expressions (re)**: Text cleaning.

## ML Models Used

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Converts movie metadata (genres, keywords, overview) into numerical vectors, emphasizing significant terms (max 5000 features).
- **Cosine Similarity**: Measures similarity between movies based on TF-IDF vectors to identify similar movies.
- **Content-Based Recommendation Algorithm**: Recommends top N movies (default N=5) with highest cosine similarity to the input movie.

## Results

### Example Results
Recommendations for **Avengers: Age of Ultron**:

| Rank | Recommended Movie                          |
|------|--------------------------------------------|
| 1    | Pirates of the Caribbean: At World's End |
| 2    | Spectre                                   |
| 3    | The Dark Knight Rises                    |
| 4    | John Carter                              |
| 5    | Spider-Man 3                             |


The recommendations are generated using a content-based filtering approach. Movie metadata (genres, keywords, overview) from TMDB is preprocessed, converted to TF-IDF vectors, and compared using cosine similarity to find the most similar movies.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/MOSIS-CREATOR/Movie-Recommendation-ML-Project/tree/main
   ```
2. Navigate to the project directory:
   ```bash
   cd Movie-Recommendation-ML-Project
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the notebook:
   ```bash
   jupyter notebook Movie_recommendation_system.ipynb
   ```

## License

This project is licensed under the GNU General Public License v3.0.
