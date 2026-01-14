# Movie Recommendation System 🎬

This project is a Machine Learning based Movie Recommendation System 
that suggests similar movies using content-based filtering.

## Features
- Recommends movies based on similarity
- Uses cosine similarity for ML logic
- Fetches real-time movie posters using TMDB API
- Simple web interface using Streamlit

## Tech Stack
- Python
- Pandas, Scikit-learn
- Streamlit

- NOTE:
The trained similarity model file (similarity.pkl) is not included in this repository
because it exceeds GitHub's file size limit.
To run the project locally, generate the model by running train_model.py first.

- TMDB API

## How to Run
1. Clone the repository  
2. Install requirements  
3. Add your TMDB API key in .env file  
4. Run: streamlit run app.py
