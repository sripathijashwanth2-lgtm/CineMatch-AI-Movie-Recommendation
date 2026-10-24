# 🎬 CineMatch — AI Movie Recommendation System

CineMatch is a content-based movie recommendation system built with Python and Streamlit. It recommends movies based on their genres and descriptions using machine learning techniques.

## ✨ Features

* 🎥 Select a movie and discover similar movies.
* 🤖 Content-based recommendations using TF-IDF and cosine similarity.
* 🖼️ Movie posters from TMDB.
* ⭐ Movie ratings from TMDB.
* 🎨 Dark-themed, interactive user interface.
* 📚 Dataset containing 80+ movies.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Requests
* The Movie Database (TMDB) API

## 📂 Project Structure

```text
Movie_Recomendation_system/
├── app.py
├── movies.csv
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml
```

## ⚙️ Installation and Setup

1. Install Python.

2. Open a terminal in the project folder.

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure your TMDB API key in `.streamlit/secrets.toml`:

   ```toml
   TMDB_API_KEY = "YOUR_TMDB_API_KEY"
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

## 🧠 How It Works

1. Movie genres and descriptions are combined into text features.
2. TF-IDF converts the text into numerical vectors.
3. Cosine similarity measures how similar movies are.
4. The application recommends movies with the highest similarity scores, excluding the selected movie.

## 🎯 Project Objective

To demonstrate how machine learning and content-based filtering can help users discover movies they may enjoy.

## 👨‍💻 Author

Developed as an AI/ML learning and internship project.
