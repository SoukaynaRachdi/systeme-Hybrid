import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- Chargement des données ---
@st.cache_data
def load_data():
    lastfm_df = pd.read_csv("Last.fm_data.csv")
    tracks_df = pd.read_csv("tracks.csv")

    lastfm_df.dropna(inplace=True)
    tracks_df.dropna(inplace=True)
    lastfm_df['Date'] = pd.to_datetime(lastfm_df['Date'])

    merged_df = pd.merge(lastfm_df, tracks_df, left_on='Track', right_on='name', how='inner')

    # On garde les colonnes utiles
    merged_df = merged_df[['Username', 'Track', 'Artist', 'popularity', 'danceability', 'energy', 'loudness', 'tempo']]

    # Normalisation des features
    scaler = StandardScaler()
    merged_df[['popularity', 'danceability', 'energy', 'loudness', 'tempo']] = scaler.fit_transform(
        merged_df[['popularity', 'danceability', 'energy', 'loudness', 'tempo']]
    )

    # Création de la matrice user-item
    user_item_matrix = lastfm_df.pivot_table(index='Username', columns='Track', aggfunc='size', fill_value=0)

    # Matrice de similarité entre utilisateurs
    similarity_matrix = pd.DataFrame(
        cosine_similarity(user_item_matrix),
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    return lastfm_df, tracks_df, merged_df, user_item_matrix, similarity_matrix

# Chargement des données
lastfm_df, tracks_df, merged_df, user_item_matrix, similarity_matrix = load_data()

# --- Header personnalisé ---
st.markdown("""
    <h1 style="text-align:center; color:#6a0dad;">🎶 Application de Recommandation Musicale Hybride</h1>
    <p style="text-align:center; font-size:18px;">Basée sur les préférences des utilisateurs et les caractéristiques audio des chansons</p>
""", unsafe_allow_html=True)


# --- Recommandation basée sur le contenu ---
def recommend_content_for_user(user_id, merged_df, top_n=10):
    user_tracks = merged_df[merged_df['Username'] == user_id]
    user_profile = user_tracks[['popularity', 'danceability', 'energy', 'loudness', 'tempo']].mean().values.reshape(1, -1)
    all_tracks = merged_df.drop_duplicates(subset='Track')
    non_listened_tracks = all_tracks[~all_tracks['Track'].isin(user_tracks['Track'])]
    features = non_listened_tracks[['popularity', 'danceability', 'energy', 'loudness', 'tempo']]
    similarities = cosine_similarity(user_profile, features)
    non_listened_tracks = non_listened_tracks.copy()
    non_listened_tracks['similarity'] = similarities[0]
    return non_listened_tracks[['Track', 'Artist', 'similarity']].drop_duplicates(subset=['Track'])

# --- Recommandation collaborative ---
def recommend_collaborative_for_user(user_id, user_similarity_df, user_item_matrix, top_n=10):
    user_similarities = user_similarity_df[user_id]
    similar_users = user_similarities.sort_values(ascending=False).index[1:]
    similar_users_tracks = user_item_matrix.loc[similar_users].sum(axis=0)
    return similar_users_tracks.sort_values(ascending=False).head(top_n)

# --- Recommandation hybride ---
def hybrid_recommendation(user_id, merged_df, similarity_matrix, user_item_matrix, alpha=0.5, top_n=10):
    user_ratings = user_item_matrix.loc[user_id]
    user_rated_items = user_ratings[user_ratings > 0].index.tolist()

    collaborative_scores = similarity_matrix.loc[user_id].dot(user_item_matrix).div(similarity_matrix.loc[user_id].sum())
    content_scores = user_item_matrix.mean(axis=0)
    final_scores = alpha * collaborative_scores + (1 - alpha) * content_scores

    final_scores = final_scores.drop(user_rated_items, errors='ignore')
    top_recs = final_scores.sort_values(ascending=False).head(top_n)
    hybrid_df = pd.DataFrame({'Track': top_recs.index, 'final_score': top_recs.values})

    # Récupérer les artistes
    if 'Track_Artist' in merged_df.columns:
        track_artist_df = merged_df[['Track', 'Track_Artist']].drop_duplicates(subset=['Track'])
    elif 'Artist' in merged_df.columns:
        track_artist_df = merged_df[['Track', 'Artist']].drop_duplicates(subset=['Track'])
        track_artist_df = track_artist_df.rename(columns={'Artist': 'Track_Artist'})
    else:
        raise ValueError("Aucune colonne artiste ('Artist' ou 'Track_Artist') trouvée dans merged_df")

    # Fusion avec les artistes
    hybrid_df = pd.merge(hybrid_df, track_artist_df, on='Track', how='left')
    hybrid_df['Track_Artist'] = hybrid_df['Track_Artist'].fillna('Artiste inconnu')

    return hybrid_df

# --- Barre latérale avec informations ---
with st.sidebar:
    st.markdown("## 👤 Sélectionnez un utilisateur")
    user_list = merged_df['Username'].unique()
    selected_user = st.selectbox("", user_list)

    st.markdown("## ⚖️ Réglage alpha (pondération)")
    alpha = st.slider("0 = contenu | 1 = collaboratif", 0.0, 1.0, 0.5, 0.05)

    st.markdown("## 📝 À propos")
    st.markdown("""
        Ce système hybride utilise :
        - 🔁 Filtrage collaboratif
        - 🎯 Recommandation basée sur le contenu

        Vous pouvez :
        - Sélectionner un utilisateur
        - Ajuster la pondération alpha
        - Parcourir les recommandations une à une
    """)

    if st.button("🎧 Générer recommandations"):
        recommendations = hybrid_recommendation(selected_user, merged_df, similarity_matrix, user_item_matrix, alpha, top_n=10)
        st.session_state['rec'] = recommendations
        st.session_state['index'] = 0

# --- Affichage des recommandations ---
if 'rec' in st.session_state:
    idx = st.session_state['index']
    rec = st.session_state['rec']

    st.markdown(f"""
    <div style="border-radius: 15px; padding: 20px; background: linear-gradient(135deg, #6a0dad, #9b59b6); color: white;">
        <h2 style="font-family: 'Arial Black', Gadget, sans-serif;">🎵 {rec.loc[idx, 'Track']}</h2>
        <h4 style="font-style: italic; font-family: 'Arial';">Artiste : {rec.loc[idx, 'Track_Artist']}</h4>
        <p style="font-weight: bold;">Score de recommandation : {rec.loc[idx, 'final_score']:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Précédent") and idx > 0:
            st.session_state['index'] = idx - 1

    with col3:
        if st.button("Suivant ➡️") and idx < len(rec) - 1:
            st.session_state['index'] = idx + 1

# --- Footer personnalisé ---
st.markdown("""
    <hr>
    <div style="text-align:center; font-size: 14px; color: gray;">
        📊 Développé par Lamyae BENNIS | 📅 2025 | 🎓 Projet académique - Recommandation Hybride
    </div>
""", unsafe_allow_html=True)
