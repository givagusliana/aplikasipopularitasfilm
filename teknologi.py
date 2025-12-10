import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px

# --- Styling CSS ---
st.markdown("""
<style>
.stApp { background: #ffffff; }
[data-testid="stSidebar"] { background-color: #3f51b5; color: white; border-right: 2px solid #5c6bc0; }
.header-box { background-color: #e8eaf6; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 3px 6px rgba(0,0,0,0.1); margin-bottom: 25px; }
.header-box h2 { color: #3f51b5; font-weight: 700; margin-bottom: 5px; }
div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] ul, div[data-testid="stMarkdownContainer"] h3 { text-align: left !important; color: #222; }
.dataframe-container { background-color: #fafafa; border-radius: 8px; padding: 10px; margin-top: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header biru
# -----------------------------
st.markdown("""
<div class="header-box">
    <h2>Aplikasi Analisis Pola Hubungan Popularitas Film</h2>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("<h3 style='color: white; text-align: center;'>MENU NAVIGASI</h3>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["Home", "Input Data", "Preprocessing", "Analysis", "Visualizations", "About Us"],
        icons=["house", "upload", "gear", "bar-chart", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={"container": {"background-color": "#3f51b5"},
                "icon": {"color": "#e8eaf6", "font-size": "20px"},
                "nav-link": {"font-size": "18px", "color": "white"},
                "nav-link-selected": {"background-color": "#e8eaf6", "color": "#3f51b5"}}
    )

# -----------------------------
# Session state
# -----------------------------
if "data" not in st.session_state: st.session_state["data"] = None
if "clean_data" not in st.session_state: st.session_state["clean_data"] = None

# -----------------------------
# HOME
# -----------------------------
def home():
    st.markdown("""
    <div style="text-align:left; padding: 5px 10px;">
        <h3 style="font-size:24px; font-weight:700;">Selamat Datang 👋</h3>
        <p>
        Aplikasi ini membantu Anda menganalisis pola hubungan antara <b>rating film</b>, <b>popularitas</b>,
        <b>genre</b>, dan <b>tahun rilis</b> menggunakan metode <b>K-Means Clustering</b>.
        </p>
        <p>Langkah-langkah:</p>
        <ul>
            <li>Unggah dataset film (contoh dari TMDB)</li>
            <li>Bersihkan data di menu Preprocessing</li>
            <li>Lihat hasil Analisis dan Visualisasi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.image("chart_icon.jpg", use_container_width=False, width=350)

# -----------------------------
# INPUT DATA
# -----------------------------
def input_data():
    st.subheader("Input Data Film 🎬")
    uploaded_file = st.file_uploader("Unggah dataset CSV", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            rename_map = {"vote_average": "Rating", "popularity": "Popularitas", "release_date": "Tahun_Rilis", "genres": "Genre"}
            data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns}, inplace=True)
            if "Tahun_Rilis" in data.columns: data["Tahun_Rilis"] = pd.to_datetime(data["Tahun_Rilis"], errors="coerce").dt.year
            if "Genre" in data.columns:
                def parse_genres(g):
                    try:
                        genres = ast.literal_eval(g)
                        if isinstance(genres, list):
                            return ", ".join([x.get("name", "") for x in genres if isinstance(x, dict)])
                    except: return g
                    return g
                data["Genre"] = data["Genre"].apply(parse_genres)
            st.session_state["data"] = data
            st.session_state["clean_data"] = None
            st.success("✅ Dataset berhasil diunggah!")
            st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
            st.dataframe(data.head(10))
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
    else:
        st.info("Silakan unggah file CSV dengan kolom `vote_average`, `popularity`, `release_date`, dan `genres`.")

# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocessing():
    st.subheader("Preprocessing Data 🎯")
    
    if st.session_state["data"] is None:
        st.warning("❗ Harap unggah dataset terlebih dahulu di menu Input Data.")
        return
    
    data = st.session_state["data"].copy()
    
    st.write("**📋 Data Mentah:**")
    st.dataframe(data)
    
    missing_total = data.isnull().sum().sum()
    duplicates_total = data.duplicated().sum()
    
    st.info(
        f"Ditemukan **{missing_total}** nilai kosong "
        f"dan **{duplicates_total}** baris duplikat sebelum pembersihan."
    )

    if st.button("Bersihkan Data (Hapus Baris Kosong & Duplikat)"):

        before_rows = len(data)
        rows_with_na = data.isnull().any(axis=1).sum()
        data_cleaned_na = data.dropna(how='any')
        before_dedup = len(data_cleaned_na)
        data_final = data_cleaned_na.drop_duplicates(keep='first')
        after_rows = len(data_final)

        duplicates_deleted = before_dedup - after_rows

        st.session_state["clean_data"] = data_final
        
        st.success("✅ Data berhasil dibersihkan!")

        st.markdown("---")
        st.write("### Ringkasan Proses Pembersihan")
        st.write(f"- Baris Awal: **{before_rows}**")
        st.write(f"- Baris Dihapus (nilai kosong): **{rows_with_na}**")
        st.write(f"- Baris Dihapus (duplikat): **{duplicates_deleted}**")
        st.write(f"- Baris Akhir: **{after_rows}**")
        
        st.write("**📊 Data Setelah Dibersihkan:**")
        st.dataframe(data_final)

# -----------------------------
# ANALYSIS (DENGAN K-MEANS MANUAL INSIGHT)
# -----------------------------
def analysis():
    st.subheader("Analisis Data 📊")
    if st.session_state['clean_data'] is None or st.session_state['clean_data'].empty:
        st.warning("⚠️ Jalankan preprocessing terlebih dahulu sebelum melakukan analisis.")
        return
    data = st.session_state['clean_data'].copy()
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    if not numeric_cols:
        st.error("Dataset tidak memiliki kolom numerik untuk dianalisis.")
        return

    analysis_type = st.selectbox(
        "Pilih Jenis Analisis:", 
        [
            "Statistik Deskriptif", 
            "Korelasi Antar Fitur Numerik", 
            "Rata-rata Nilai Berdasarkan Kategori", 
            "Analisis Korelasi Dua Fitur", 
            "K-Means Clustering"
        ]
    )

    if analysis_type == "Statistik Deskriptif":
        st.markdown("### 📈 Statistik Deskriptif")
        st.dataframe(data[numeric_cols].describe().T.style.background_gradient(cmap="Purples"))

    elif analysis_type == "Korelasi Antar Fitur Numerik":
        st.markdown("### 🔍 Korelasi Antar Fitur Numerik")
        corr = data[numeric_cols].corr()
        st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None))

    elif analysis_type == "Rata-rata Nilai Berdasarkan Kategori":
        if not categorical_cols:
            st.warning("Dataset ini tidak memiliki kolom kategorikal untuk analisis ini."); return
        cat_col = st.selectbox("Pilih kolom kategorikal:", categorical_cols)
        num_col = st.selectbox("Pilih kolom numerik:", numeric_cols)
        avg_values = data.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(20)
        st.dataframe(avg_values.to_frame().style.background_gradient(cmap="Purples"))

    elif analysis_type == "Analisis Korelasi Dua Fitur":
        if len(numeric_cols) < 2: st.warning("Dataset tidak memiliki cukup kolom numerik untuk analisis ini."); return
        col1 = st.selectbox("Pilih fitur pertama:", numeric_cols)
        col2 = st.selectbox("Pilih fitur kedua:", [c for c in numeric_cols if c != col1])
        corr_value = data[col1].corr(data[col2])
        st.markdown(f"### 🔗 Korelasi antara **{col1}** dan **{col2}** = **{corr_value:.3f}**")
        arah = "positif" if corr_value > 0 else "negatif"
        tingkat = "kuat" if abs(corr_value) >= 0.7 else "sedang" if abs(corr_value) >= 0.4 else "lemah"
        st.info(f"Tingkat korelasi {tingkat} ({arah})")

    # -------------------------
    # K-MEANS + ELBOW METHOD
    # -------------------------
    elif analysis_type == "K-Means Clustering":
        st.markdown("### 🤖 Analisis K-Means Clustering")

        # Set default fitur agar konsisten dengan analisis sebelumnya (Rating, Popularitas, Tahun_Rilis)
        default_features = ["Rating", "Popularitas", "Tahun_Rilis"]
        available_defaults = [f for f in default_features if f in numeric_cols]
        if len(available_defaults) < 2:
             available_defaults = numeric_cols[:2]

        selected_features = st.multiselect("Pilih fitur numerik untuk clustering:", numeric_cols, default=available_defaults)
        
        if len(selected_features) < 2:
            st.warning("Pilih minimal dua fitur numerik.")
            return

        X = data[selected_features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- ELBOW METHOD ---
        st.markdown("### 📌 Menentukan Jumlah Cluster Terbaik (Elbow Method)")

        max_k = st.slider("Pilih Maksimum K:", 4, 15, 10)
        sse = []

        for k in range(1, max_k+1):
            km = KMeans(n_clusters=k, random_state=42, n_init='auto') # n_init='auto' untuk menghindari warning
            km.fit(X_scaled)
            sse.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(range(1, max_k+1), sse, marker='o')
        ax.set_xlabel("Jumlah Cluster (k)")
        ax.set_ylabel("SSE / Inertia")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

        st.info("Pilih titik siku (elbow) untuk menentukan cluster optimal.")

        # --- CLUSTERING FINAL ---
        num_clusters = st.slider("Pilih jumlah cluster:", 2, 10, 3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto') # n_init='auto'
        
        # Pastikan data yang digunakan untuk prediksi sesuai dengan X yang sudah difilter/dibersihkan
        data_to_cluster = data[selected_features].dropna()
        data.loc[data_to_cluster.index, "Cluster"] = kmeans.fit_predict(X_scaled)
        
        # Konversi Cluster ke tipe integer
        if "Cluster" in data.columns:
            data["Cluster"] = data["Cluster"].fillna(-1).astype(int) 

        st.session_state["clean_data"] = data

        st.success("✅ Clustering selesai dan hasilnya telah ditambahkan ke dataset.")

        summary_df = data.groupby("Cluster")[selected_features].mean().round(2)
        st.write("Rata-rata fitur tiap cluster:")
        st.dataframe(summary_df.style.background_gradient(cmap="Purples"))

        # --- RINGKASAN MANUAL YANG DIMINTA PENGGUNA ---
        st.markdown("### 🧩 Ringkasan Pola Cluster Manual (Contoh Interpretasi Hasil K=3)")
        
        # Interpretasi Manual Berdasarkan data yang telah dibahas sebelumnya:
        st.markdown("""
        * **Cluster 0**: Film Klasik (sekitar 1978) dengan **Rating Tertinggi** (sekitar 7.01) dan Popularitas Tinggi/Menengah.
        * **Cluster 1**: Film Modern (sekitar 2009) dengan **Popularitas Terendah** (sekitar 19.81) dan **Rating Terendah** (sekitar 5.53).
        * **Cluster 2**: Film Modern (sekitar 2008) dengan **Popularitas Tertinggi** (sekitar 50.44) dan Rating Tinggi (sekitar 6.92).
        
        **Catatan:** Interpretasi ini adalah contoh, dan angka rata-rata aktual dapat berbeda tergantung dataset yang diunggah.
        """)
        # ---------------------------------------------


# -----------------------------
# VISUALIZATION
# -----------------------------
def visualizations():
    st.subheader("📊 Visualisasi Data Film (Dataset Asli)")

    if st.session_state['clean_data'] is None or st.session_state['clean_data'].empty:
        st.warning("⚠️ Jalankan preprocessing dan clustering terlebih dahulu sebelum melakukan visualisasi.")
        return

    data = st.session_state['clean_data'].copy()
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()

    st.markdown("### 🔹 Filter Data")
    col1, col2 = st.columns(2)
    with col1:
        if "Tahun_Rilis" in data.columns:
            tahun_min = int(data['Tahun_Rilis'].min()) if not data['Tahun_Rilis'].empty else 1900
            tahun_max = int(data['Tahun_Rilis'].max()) if not data['Tahun_Rilis'].empty else 2025
            tahun_range = st.slider("Pilih rentang Tahun Rilis:", tahun_min, tahun_max, (tahun_min, tahun_max))
        else:
            tahun_range = (0, 9999)
    with col2:
        if "Genre" in data.columns:
            # Mengumpulkan semua genre unik
            all_genres_list = []
            for item in data["Genre"].dropna():
                all_genres_list.extend([g.strip() for g in item.split(',') if g.strip()])
            genre_list = sorted(list(set(all_genres_list)))
            
            selected_genre = st.multiselect("Pilih Genre:", genre_list, default=genre_list[:5] if genre_list else [])
        else:
            selected_genre = []

    filtered = data.copy()
    if "Tahun_Rilis" in filtered.columns:
        filtered = filtered[(filtered['Tahun_Rilis'] >= tahun_range[0]) & (filtered['Tahun_Rilis'] <= tahun_range[1])]
    if selected_genre and "Genre" in filtered.columns:
        filtered = filtered[filtered['Genre'].apply(lambda x: any(g in x for g in selected_genre))]

    st.write(f"Menampilkan **{len(filtered)}** film setelah filter.")
    st.dataframe(filtered.head(10))

    if numeric_cols:
        st.markdown("### 🔥 Heatmap Korelasi")
        corr = filtered[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.markdown("### 🔗 Pilih Fitur Numerik untuk Pairplot / Scatter 3D")
    selected_features_viz = st.multiselect("Fitur numerik:", numeric_cols, default=numeric_cols[:3])

    if selected_features_viz:
        num_data = filtered[selected_features_viz].copy().dropna()

        if len(selected_features_viz) >= 2:
            st.markdown("### 🔹 Pairplot Antar Fitur Numerik")
            try:
                # Tambahkan 'Cluster' sebagai hue jika sudah ada
                hue_col = "Cluster" if "Cluster" in filtered.columns and filtered["Cluster"].nunique() > 1 else None
                pairplot_fig = sns.pairplot(filtered.dropna(subset=selected_features_viz), vars=selected_features_viz, hue=hue_col, palette="viridis")
                st.pyplot(pairplot_fig)
            except Exception as e:
                st.warning(f"Gagal membuat pairplot: Pastikan data mencukupi. ({e})")

        if len(selected_features_viz) >= 3:
            st.markdown("### 🔹 Scatter 3D Antar Fitur Numerik")
            fig = px.scatter_3d(
                filtered.dropna(subset=selected_features_viz),
                x=selected_features_viz[0],
                y=selected_features_viz[1],
                z=selected_features_viz[2],
                color="Cluster" if "Cluster" in filtered.columns else None,
                hover_data=["title", "Genre", "Tahun_Rilis", "Rating", "Popularitas"] if "Genre" in filtered.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)

    if "Genre" in filtered.columns:
        st.markdown("### 🎭 Distribusi Genre Teratas")
        all_genres = []
        for g_list in filtered["Genre"].dropna():
            all_genres.extend([g.strip() for g in g_list.split(',') if g.strip()])
            
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        
        if not genre_counts.empty:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="viridis", ax=ax)
            ax.set_xlabel("Jumlah Film")
            ax.set_ylabel("Genre")
            st.pyplot(fig)
        else:
            st.info("Tidak ada data genre setelah filter.")

        st.markdown("### ☁️ Wordcloud Genre")
        text = " ".join(all_genres)
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Tidak ada data genre untuk Wordcloud.")

    if "Tahun_Rilis" in filtered.columns and not filtered["Tahun_Rilis"].empty:
        st.markdown("### 📈 Trend Tahunan Rating & Popularitas")
        trend = filtered.groupby("Tahun_Rilis")[["Rating","Popularitas"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(trend["Tahun_Rilis"], trend["Rating"], label="Rating", marker='o')
        ax.plot(trend["Tahun_Rilis"], trend["Popularitas"], label="Popularitas", marker='o')
        ax.set_xlabel("Tahun Rilis")
        ax.set_ylabel("Nilai Rata-rata")
        ax.set_title("Trend Tahunan Rating & Popularitas")
        ax.legend()
        st.pyplot(fig)

# -----------------------------
# ABOUT US
# -----------------------------
def about_us():
    st.subheader("Tentang Kami 💡")
    st.markdown("""
    <div style="background-color: #f0f4ff; padding: 15px; border-radius: 8px; border-left: 5px solid #3f51b5;">
        <p><strong>Pengembang:</strong> Kelompok 5 Mata Kuliah Akuisisi Data</p>
        <ul style="list-style-type: none; padding-left: 20px;">
            <li>🎓 Maghfira Islami (2311521010)</li>
            <li>🎓 Sherly Ayuma Putri (2311521018)</li>
            <li>🎓 Giva Gusliana (2311523022)</li>
        </ul>
        <p><strong>Tujuan:</strong> Mengembangkan aplikasi interaktif untuk analisis popularitas film berbasis data.</p>
        <p><strong>Teknologi:</strong> Streamlit, Pandas, Seaborn, Matplotlib, dan Scikit-learn.</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# ROUTING
# -----------------------------
if selected == "Home": home()
elif selected == "Input Data": input_data()
elif selected == "Preprocessing": preprocessing()
elif selected == "Analysis": analysis()
elif selected == "Visualizations": visualizations()
elif selected == "About Us": about_us()