import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Memuat dataset
def load_data(desc_path, rating_path):
    # Load data deskripsi dan rating
    desc_df = pd.read_csv(desc_path)
    rating_df = pd.read_csv(rating_path)
    return desc_df, rating_df

# Membersihkan teks pada kolom FASILITAS --> TF-IDF
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def clean_data(desc_df):
    desc_df['FASILITAS'] = desc_df['FASILITAS'].fillna('').apply(preprocess_text)
    desc_df['Id wisata'] = desc_df['Id wisata'].astype(int) #tipe integer
    return desc_df

# Menghitung Cosine Similarity antar item (wisata) berdasarkan data rating.
def calculate_item_similarity(rating_df):
    ratings = rating_df.iloc[:, 2:].astype(float)
    similarity_matrix = cosine_similarity(ratings.T)
    return pd.DataFrame(similarity_matrix, index=ratings.columns.astype(int), columns=ratings.columns.astype(int))

def get_tfidf_similarity(desc_df):
    desc_df = clean_data(desc_df)
    # Menghitung TF-IDF pada kolom FASILITAS
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = tfidf.fit_transform(desc_df['FASILITAS'])
    # Hitung cosine similarity antar wisata berdasarkan fasilitas
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # Konversi ke DataFrame dengan index dan kolom Id wisata
    return pd.DataFrame(similarity_matrix, index=desc_df['Id wisata'], columns=desc_df['Id wisata'])

def validate_wisata(selected_wisata, desc_df):
    """
    Validasi apakah wisata yang dipilih tersedia dalam dataset.
    """
    if selected_wisata not in desc_df['Nama Wisata'].values:
        raise ValueError("Wisata tidak ditemukan dalam dataset.")

def recommend_wisata(item_similarity, tfidf_similarity, selected_wisata, desc_df, top_n=3, alpha=0.6):
    # Merekomendasikan wisata berdasarkan kombinasi item-based similarity dan TF-IDF similarity.
  
    # Validasi input
    validate_wisata(selected_wisata, desc_df)

    # Gabungkan similarity dengan bobot alpha
    combined_similarity = (alpha * item_similarity) + ((1 - alpha) * tfidf_similarity)

    # Ambil ID wisata yang dipilih
    wisata_idx = desc_df.loc[desc_df['Nama Wisata'] == selected_wisata, 'Id wisata'].iloc[0]

    # Urutkan wisata berdasarkan similarity
    sim_scores = combined_similarity.loc[wisata_idx].sort_values(ascending=False)
    sim_scores = sim_scores.drop(wisata_idx)  # Hapus wisata itu sendiri

    # Ambil top N rekomendasi
    top_indices = sim_scores.head(top_n).index
    recommended_wisata = []
    for idx in top_indices:
        nama_wisata = desc_df.loc[desc_df['Id wisata'] == idx, 'Nama Wisata'].values[0]
        score = sim_scores.loc[idx]
        recommended_wisata.append((nama_wisata, score))
    return recommended_wisata

def visualize_combined_similarity(item_similarity, tfidf_similarity, desc_df, alpha=0.6):
    combined_similarity = (alpha * item_similarity) + ((1 - alpha) * tfidf_similarity)
    plt.figure(figsize=(12, 10))
    sns.heatmap(combined_similarity, cmap="Blues", xticklabels=desc_df['Nama Wisata'], yticklabels=desc_df['Nama Wisata'])
    plt.title("Combined Similarity Heatmap")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(plt)


def recommend_by_profile(rating_df, desc_df, umur, jenis_kelamin, top_n=5):
    filtered_users = rating_df[rating_df['Jenis Kelamin'] == jenis_kelamin]
    
    if filtered_users.empty:
        raise ValueError("Tidak ada pengguna dengan jenis kelamin yang dipilih.")

    avg_ratings = filtered_users.iloc[:, 2:].mean().sort_values(ascending=False)

    # Ambil top N wisata dengan rating tertinggi
    recommendations = []
    for wisata_id, score in avg_ratings.head(top_n).items():
        nama_wisata = desc_df.loc[desc_df['Id wisata'] == int(wisata_id), 'Nama Wisata'].values[0]
        recommendations.append((nama_wisata, score))

    return recommendations

