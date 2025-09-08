

# 🎬 Personalized Movie Recommender

A **Streamlit-based web application** that provides personalized movie recommendations based on a movie you already like. The app uses **TMDB dataset** and **content-based filtering** with TF-IDF and cosine similarity to suggest similar movies, along with their posters and trailers.

---

## 🚀 Features

* **Movie Recommendations:** Enter a movie you like, and get the top 5 similar movies.
* **Movie Posters:** Displays high-quality posters fetched from TMDB API.
* **Trailers:** Watch the trailer of recommended movies directly on YouTube.
* **Premium UI:** Gradient headers, card-style recommendations, hover effects, and emoji ratings.
* **Fast & Responsive:** Built using Streamlit with caching for optimized performance.

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Streamlit** – Interactive web interface
* **Pandas** – Data manipulation
* **Scikit-learn** – TF-IDF Vectorizer & cosine similarity
* **Requests** – TMDB API integration

---

## 📁 Dataset

* **Movies:** `tmdb_5000_movies.csv`
* **Credits:** `tmdb_5000_credits.csv`

The datasets are publicly available via [Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run main.py
```

---

## 🔑 TMDB API Key

* The app uses TMDB API to fetch movie posters and trailers.
* Get your free API key from [TMDB API](https://www.themoviedb.org/documentation/api).
* Replace the `TMDB_API_KEY` in `main.py` with your key:

```python
TMDB_API_KEY = 'YOUR_API_KEY_HERE'
```

---

## 🧠 How It Works

1. **Data Processing:**

   * Merges `movies` and `credits` datasets.
   * Extracts genres, keywords, cast (top 3), director, and overview to create a `tags` column.
   * Converts all text to lowercase for uniformity.

2. **Content-based Filtering:**

   * Uses **TF-IDF Vectorizer** to convert `tags` into numerical vectors.
   * Computes **cosine similarity** between movies.

3. **Recommendation:**

   * Finds the 5 most similar movies to the selected one.
   * Fetches poster and trailer URL using TMDB API.

---

## 📈 Future Enhancements

* Add **genre filters** for more precise recommendations.
* Include **ratings and reviews** from TMDB.
* Implement **user login** to save favorite movies.
* Add **dark mode toggle** for UI personalization.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
