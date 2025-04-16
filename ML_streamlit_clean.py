
import streamlit as st
import numpy as np
import pandas as pd
import ast
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import FeatureHasher

nltk.download('stopwords')
try:
    stop_words_EF = stopwords.words('english') + stopwords.words('french')
except:
    stop_words_EF = []
    st.warning("⚠️ Les stopwords n'ont pas pu être chargés correctement. Le modèle peut être moins performant.")



# --- Interface Streamlit ---
st.title("🎬 Duch_Recommandation de films")
st.image(r"https://images.app.goo.gl/S1xiFmJA1dJu42JG7")

# --- Chargement du dataset ---
df = pd.read_parquet(r"df_machine_learning.parquet")
df = df.reset_index(drop=True)

# --- Prétraitement identique au code ML initial ---

# Colonnes numériques
df_numeric = df.select_dtypes(include=['number'])
scaler = StandardScaler()
df_numeric_scale = scaler.fit_transform(df_numeric)

# NLP sur overview avec TF-IDF
df['overview'] = df['overview'].astype(str).fillna("")
tfidf = TfidfVectorizer(stop_words=stop_words_EF, max_features=300)
overview_enc = tfidf.fit_transform(df['overview'].fillna(""))

# Encodage des colonnes de type liste
df['genres'] = df['genres'].apply(lambda x: [x] if isinstance(x, str) else x)
df['actors'] = df['actors'].apply(lambda x: [x] if isinstance(x, str) else x).fillna('')
df['directors'] = df['directors'].apply(lambda x: [x] if isinstance(x, str) else x).fillna('')
df['production_countries'] = df['production_countries'].apply(ast.literal_eval)


mlb = MultiLabelBinarizer()
genres_enc = mlb.fit_transform(df['genres'])
countries_enc = mlb.fit_transform(df['production_countries'])
hasher_directeurs = FeatureHasher(n_features=128, input_type='string')
director_enc = hasher_directeurs.fit_transform(df['directors'])
hasher_acteurs = FeatureHasher(n_features=256, input_type='string')
actor_enc = hasher_acteurs.fit_transform(df['actors'])


# Fusion des features
data = csr_matrix(hstack([
    overview_enc,
    genres_enc,
    countries_enc,
    director_enc,
    actor_enc,
    df_numeric_scale
]))

# --- Entraînement du modèle Nearest Neighbors ---
NNmodel = NearestNeighbors(n_neighbors=6, metric='minkowski')
NNmodel.fit(data)

# --- Interface utilisateur pour choix de film ---
film_titre = st.selectbox("📽️ Choisissez un film :", df["title"].dropna().sort_values().unique())
film_index = df[df["title"] == film_titre].index[0]

# --- Affichage des infos du film ---
film = df.iloc[film_index]
st.subheader("🎞️ Film sélectionné")
st.write(f"**Titre :** {film['title']}")
st.write(f"**Genres :** {film['genres']}")
st.write(f"**Année :** {film['startYear']}")
st.write(f"**Pays :** {film['production_countries']}")
st.write(f"**Acteurs :** {film['actorsName']}")
st.write("**Synopsis :**", film["overview"])

# --- Recommandation ---
recommandation = NNmodel.kneighbors(data[film_index], return_distance=False)
st.subheader("🎯 Films similaires recommandés")
for i in recommandation[0][1:]:
    reco = df.iloc[i]
    st.markdown(f"**🎬 {reco['title']}** - *{reco['genres']}*")

st.markdown("---")





