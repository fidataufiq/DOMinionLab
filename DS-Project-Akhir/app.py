import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.recommendation import (
    load_data,
    calculate_item_similarity,
    get_tfidf_similarity,
    recommend_wisata,
    validate_wisata,
    recommend_by_profile,
    visualize_combined_similarity
)

# Path ke file data
DESC_PATH = 'data/descwisata.csv'
RATING_PATH = 'data/ratingwisata.csv'

# Load data
desc_df, rating_df = load_data(DESC_PATH, RATING_PATH)
item_similarity = calculate_item_similarity(rating_df)
tfidf_similarity = get_tfidf_similarity(desc_df)

# Sidebar untuk navigasi
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio("Pilih Menu", ["Rekomendasi Wisata", "Daftar Wisata", "Proses Metode", "Rekomendasi Berdasarkan Profil"])

# Halaman Rekomendasi Wisata
if menu == "Rekomendasi Wisata":
    st.title("Sistem Rekomendasi Wisata Banyuwangi")
    st.write("Rekomendasi ini dibuat menggunakan metode Item-Based Collaborative Filtering dan TF-IDF.")

    # Input form: Pilih hingga 3 tempat wisata
    st.header("Tempat wisata yang pernah Anda kunjungi")
    col1, col2, col3 = st.columns(3)

    # Dropdown input untuk memilih wisata
    selected_wisata_1 = col1.selectbox("Wisata 1", ["Pilih wisata"] + list(desc_df['Nama Wisata'].unique()), key="wisata1")
    selected_wisata_2 = col2.selectbox("Wisata 2", ["Pilih wisata"] + list(desc_df['Nama Wisata'].unique()), key="wisata2")
    selected_wisata_3 = col3.selectbox("Wisata 3", ["Pilih wisata"] + list(desc_df['Nama Wisata'].unique()), key="wisata3")

    # Filter input yang valid
    selected_wisata = [wisata for wisata in [selected_wisata_1, selected_wisata_2, selected_wisata_3] if wisata != "Pilih wisata"]

    # Button untuk mendapatkan rekomendasi
    if st.button("Dapatkan Rekomendasi"):
        if not selected_wisata:
            st.warning("Silakan pilih minimal satu tempat wisata yang pernah Anda kunjungi!")
        else:
            try:
                # Menggabungkan rekomendasi dari wisata yang dipilih
                combined_recommendations = {}
                for wisata in selected_wisata:
                    validate_wisata(wisata, desc_df)
                    recommendations = recommend_wisata(
                        item_similarity,
                        tfidf_similarity,
                        wisata,
                        desc_df,
                        top_n=5,
                        alpha=0.6  # Bobot alpha bisa disesuaikan
                    )
                    for rec_wisata, score in recommendations:
                        if rec_wisata in combined_recommendations:
                            combined_recommendations[rec_wisata] += score
                        else:
                            combined_recommendations[rec_wisata] = score

                # Urutkan hasil rekomendasi
                sorted_recommendations = sorted(combined_recommendations.items(), key=lambda x: x[1], reverse=True)

                # Tampilkan rekomendasi
                st.subheader("Rekomendasi wisata untuk Anda:")
                for wisata, score in sorted_recommendations[:5]:
                    st.write(f"- {wisata} (Similarity Score: {score:.2f})")
            except ValueError as e:
                st.error(str(e))

# Halaman Daftar Wisata
elif menu == "Daftar Wisata":
    st.title("Daftar Wisata Banyuwangi")
    st.write("Berikut adalah daftar lengkap tempat wisata beserta atributnya:")

    item_style = """
    <style>
    .item-container {
        border: 1px solid #444;
        border-radius: 8px;
        padding: 10px;
        margin: 10px;
        background-color: #1e1e1e;
        color: #f5f5f5;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        text-align: center;
        height: 100%; /* Tinggi maksimal sama */
        min-height: 500px; /* Tinggi minimum untuk semua kartu */
    }
    .item-container h4 {
        color: #ffffff;
        font-size: 18px;
        margin: 5px 0;
    }
    .item-container p {
        font-size: 14px;
        color: #cccccc;
        margin: 5px 0;
    }
    img {
        border-radius: 8px;
        margin-top: 10px;
        max-height: 150px;
        object-fit: cover;
    }
    </style>
    """
    st.markdown(item_style, unsafe_allow_html=True)

    # Membuat grid 3 kolom menampilkan wisata
    num_cols = 3
    cols = st.columns(num_cols)

    # Iterasi untuk setiap item wisata
    for idx, row in desc_df.iterrows():
        col = cols[idx % num_cols]
        with col:
            st.markdown(
                f"""
                <div class="item-container">
                    <h4>{row['Nama Wisata']}</h4>
                    <img src="{row['Link Gambar']}" alt="Gambar {row['Nama Wisata']}" width="100%">
                    <p><b>Jenis Wisata:</b> {row['JENIS WISATA']}</p>
                    <p><b>HTM:</b> {row['HTM']}</p>
                    <p><b>Akses Jalan:</b> {row['AKSES JALAN']}</p>
                    <p><b>Fasilitas:</b> {row['FASILITAS']}</p>
                </div>
                """, unsafe_allow_html=True
            )

        # Reset kolom setiap 3 item
        if (idx + 1) % num_cols == 0 and idx + 1 < len(desc_df):
            cols = st.columns(num_cols)



# Halaman Proses Metode
elif menu == "Proses Metode":
    st.title("Proses Metode yang Digunakan")
    st.write("Visualisasi metode **Item-Based Collaborative Filtering** dan **TF-IDF**:")

    # Visualisasi Heatmap Cosine Similarity (IBCF)
    st.subheader("1. Cosine Similarity Antar Wisata (Item-Based Collaborative Filtering)")
    plt.figure(figsize=(10, 8))
    sns.heatmap(item_similarity, annot=False, cmap="Blues", xticklabels=desc_df['Nama Wisata'], yticklabels=desc_df['Nama Wisata'])
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    st.pyplot(plt)

    # Menampilkan Kata Penting dari TF-IDF
    st.subheader("2. Kata-Kata Penting Berdasarkan Deskripsi Fasilitas (TF-IDF)")
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Hitung TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=10)
    tfidf_matrix = tfidf.fit_transform(desc_df['FASILITAS'].fillna(''))
    feature_names = tfidf.get_feature_names_out()

    st.write("Berikut adalah 10 kata paling penting dari kolom **FASILITAS** berdasarkan TF-IDF:")
    for i, word in enumerate(feature_names, 1):
        st.write(f"{i}. {word}")

    # Visualisasi Combined Similarity Heatmap
    st.subheader("3. Combined Similarity Heatmap")
    alpha = 0.6  # Bobot untuk IBCF
    combined_similarity = (alpha * item_similarity) + ((1 - alpha) * tfidf_similarity)
    plt.figure(figsize=(12, 10))
    sns.heatmap(combined_similarity, cmap="Blues", xticklabels=desc_df['Nama Wisata'], yticklabels=desc_df['Nama Wisata'])
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Combined Similarity Heatmap")
    plt.tight_layout()
    st.pyplot(plt)

    # Informasi Singkat tentang Metode
    st.subheader("4. Penjelasan Singkat Metode")
    st.write("""
    - **Item-Based Collaborative Filtering**: Menghitung kemiripan antar wisata berdasarkan rating pengguna menggunakan *Cosine Similarity*.
    - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Menganalisis kata-kata penting pada deskripsi fasilitas wisata untuk mengetahui wisata yang relevan berdasarkan teks.
    - **Combined Similarity**: Menggabungkan kedua metode di atas dengan bobot tertentu untuk mendapatkan rekomendasi yang lebih akurat.
    """)

# Halaman Analisis Per Wisata
elif menu == "Analisis Per Wisata":
    st.title("Analisis Per Wisata")
    st.write("Pilih satu wisata untuk melihat detail dan analisis lebih lanjut.")

    # Dropdown untuk memilih wisata
    selected_wisata = st.selectbox("Pilih Tempat Wisata", desc_df['Nama Wisata'].unique())

    if selected_wisata:
        # Menampilkan detail atribut wisata
        st.subheader("Detail Tempat Wisata")
        wisata_detail = desc_df[desc_df['Nama Wisata'] == selected_wisata]

        if wisata_detail.empty:
            st.warning("Data wisata tidak ditemukan.")
        else:
            st.table(wisata_detail)

            # Analisis TF-IDF untuk kolom FASILITAS
            st.subheader("Analisis Fasilitas (TF-IDF)")
            from sklearn.feature_extraction.text import TfidfVectorizer

            tfidf = TfidfVectorizer(stop_words='english', max_features=10)
            tfidf_matrix = tfidf.fit_transform(desc_df['FASILITAS'].fillna(''))
            feature_names = tfidf.get_feature_names_out()

            idx = wisata_detail.index[0]
            tfidf_scores = tfidf_matrix[idx].toarray().flatten()
            tfidf_keywords = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}

            st.write("Kata-Kata Penting dari Fasilitas:")
            for word, score in sorted(tfidf_keywords.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {word}: {score:.4f}")

            # Tampilkan wisata mirip berdasarkan cosine similarity
            st.subheader("Wisata Mirip Berdasarkan Cosine Similarity")
            alpha = 0.6  # Bobot untuk IBCF
            combined_similarity = (alpha * item_similarity) + ((1 - alpha) * tfidf_similarity)
            combined_similarity.index = combined_similarity.index.astype(int)
            combined_similarity.columns = combined_similarity.columns.astype(int)

            wisata_id = int(wisata_detail['Id wisata'].values[0])
            sim_scores = combined_similarity.loc[wisata_id].sort_values(ascending=False)
            sim_scores = sim_scores.drop(wisata_id)  # Hapus wisata itu sendiri
            top_similar = sim_scores.head(5)

            for idx, score in top_similar.items():
                nama_wisata = desc_df.loc[desc_df['Id wisata'] == idx, 'Nama Wisata'].values[0]
                st.write(f"- {nama_wisata} (Similarity Score: {score:.2f})")

# Section baru untuk rekomendasi berdasarkan profil pengguna
elif menu == "Rekomendasi Berdasarkan Profil":
    st.title("Rekomendasi Wisata Berdasarkan Profil Anda")
    st.write("Masukkan informasi pribadi Anda untuk mendapatkan rekomendasi wisata.")

    # Form input
    nama_user = st.text_input("Nama Anda (Opsional)", "")
    umur = st.number_input("Umur Anda", min_value=1, max_value=120, step=1)
    jenis_kelamin = st.radio("Jenis Kelamin", ["Laki - Laki", "Perempuan"])

    # Button untuk mendapatkan rekomendasi
    if st.button("Dapatkan Rekomendasi"):
        if umur == 0:
            st.warning("Silakan masukkan umur Anda untuk mendapatkan rekomendasi!")
        else:
            try:
                # Panggil fungsi rekomendasi berdasarkan profil pengguna
                recommendations = recommend_by_profile(rating_df, desc_df, umur, jenis_kelamin)
                
                # Tampilkan hasil dengan menyapa user jika nama diisi
                if nama_user:
                    st.subheader(f"Halo, {nama_user}! Rekomendasi wisata untuk Anda:")
                else:
                    st.subheader("Rekomendasi wisata untuk Anda:")

                # Tampilkan rekomendasi
                for wisata, score in recommendations:
                    st.write(f"- {wisata} (Similarity Score: {score:.2f})")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

