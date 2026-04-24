# 🎮 Steam Game Recommender System

**Sistem Rekomendasi Game Berbasis Content-Based Filtering**  
Kelompok 12 - Proyek Sains Data ITS

---

## 👥 Tim Pengembang

- **Ghalib Ibrahim Zardy** (5052231028)
- **M Shah Aquilla Febryano** (5052231043)

---

## 📋 Deskripsi Proyek

Aplikasi ini adalah **sistem rekomendasi game Steam** yang menggunakan metode **Content-Based Filtering** dengan teknik **TF-IDF** dan **Cosine Similarity**. Sistem dapat merekomendasikan game berdasarkan:

1. **Preferensi Genre** (Action, RPG, Strategy, Horror, dll)
2. **Spesifikasi Perangkat** (RAM requirement)
3. **Batas Anggaran** (USD/IDR)

Dataset yang digunakan berasal dari [Kaggle Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) dengan lebih dari **50,000 game**.

---

## ✨ Fitur Utama

### 🎯 Rekomendasi Game
- Filter berdasarkan genre favorit (multi-select)
- Filter berdasarkan budget (USD/IDR)
- Filter berdasarkan spesifikasi RAM
- Tampilan card yang elegant dan interaktif
- Top-N recommendations dengan similarity score

### 📊 Dashboard Monitoring
- **Precision@10** (Target: ≥70%)
- **Recall@20** (Target: ≥50%)
- **NDCG@10** (Target: ≥75%)
- **Coverage** (Target: ≥40%)
- **Diversity** (Target: ≥60%)
- **MAP Score** (Target: ≥65%)
- **User Satisfaction** (Target: ≥4.0/5.0)

### 📈 Visualisasi Data
- Distribusi genre game
- Distribusi harga game
- Distribusi rating
- Distribusi RAM requirements
- Korelasi harga vs rating
- Similarity score analysis
- Dan banyak lagi!

### 🔍 Exploratory Data Analysis
- Dataset overview dengan statistik lengkap
- Missing values analysis
- Data quality insights
- Sample data preview

### 📚 Dokumentasi Lengkap
- Step 1-7 metodologi data science
- Penelitian terdahulu
- Pembagian kerja tim
- Referensi akademis

---

## 🚀 Cara Instalasi

### 1. Clone Repository atau Download Files

```bash
# Clone repository (jika menggunakan git)
git clone <repository-url>
cd steam-recommender

# Atau download dan extract files
```

### 2. Download Dataset

**PENTING:** Dataset tidak disertakan karena ukurannya besar (300+ MB)

1. Download dataset dari: https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
2. Extract file `games.csv`
3. Letakkan `games.csv` di folder yang sama dengan `steam_recommender_app.py`

### 3. Install Dependencies

```bash
# Menggunakan pip
pip install -r requirements.txt

# Atau install satu per satu
pip install streamlit pandas numpy plotly scikit-learn
```

### 4. Jalankan Aplikasi

```bash
streamlit run steam_recommender_app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

---

## 📁 Struktur File

```
steam-recommender/
│
├── steam_recommender_app.py    # Main application file
├── requirements.txt            # Python dependencies
├── README.md                   # Dokumentasi ini
└── games.csv                   # Dataset (download terpisah)
```

---

## 💻 System Requirements

- **Python:** 3.8 atau lebih baru
- **RAM:** Minimal 4GB (8GB recommended)
- **Storage:** ~500MB untuk dataset dan dependencies
- **Browser:** Chrome, Firefox, Safari, atau Edge versi terbaru

---

## 🎨 Tampilan UI

Aplikasi menggunakan **Dark Theme Modern** dengan:
- Background gradient yang elegan (navy, charcoal)
- Aksen warna neon blue (#00d4ff)
- Typography Poppins yang modern
- Game cards dengan hover effects
- Interactive charts dengan Plotly
- Responsive layout

---

## 📊 Metodologi

### Content-Based Filtering

**TF-IDF Vectorization:**
```python
TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)
```

**Cosine Similarity:**
- Menghitung kemiripan antara user preference dan game features
- Formula: `similarity = (A · B) / (||A|| × ||B||)`

**Filtering Logic:**
1. Filter by budget (Price ≤ Max Budget)
2. Filter by RAM (RAM Required ≤ User RAM)
3. Filter by genre preferences
4. Rank by similarity score
5. Return Top-N recommendations

---

## 📈 Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Precision@10** | ≥70% | Proporsi game relevan dalam Top-10 |
| **Recall@20** | ≥50% | Proporsi game relevan yang ditemukan |
| **NDCG@10** | ≥75% | Kualitas ranking dengan bobot posisi |
| **Coverage** | ≥40% | Persentase game yang dapat direkomendasikan |
| **Diversity** | ≥60% | Variasi genre dalam rekomendasi |
| **MAP** | ≥65% | Mean Average Precision |
| **User Satisfaction** | ≥4.0/5.0 | Skor kepuasan pengguna |

---

## 🔧 Troubleshooting

### Error: "File 'games.csv' tidak ditemukan"

**Solusi:**
1. Download dataset dari Kaggle
2. Extract `games.csv`
3. Letakkan di folder yang sama dengan aplikasi
4. Refresh browser

### Error: "Module not found"

**Solusi:**
```bash
pip install --upgrade -r requirements.txt
```

### Aplikasi lambat saat loading

**Solusi:**
- Dataset besar memerlukan waktu preprocessing
- Tunggu hingga cache selesai (hanya sekali di awal)
- Selanjutnya akan lebih cepat karena caching

### RAM tidak cukup

**Solusi:**
- Tutup aplikasi lain yang berat
- Minimal RAM 4GB diperlukan
- 8GB recommended untuk performa optimal

---

## 📚 Referensi

1. **Aggarwal, C. C. (2016).** "Recommender Systems: The Textbook". Springer.
2. **Ricci, F., Rokach, L., & Shapira, B. (2015).** "Recommender Systems Handbook". Springer.
3. **Pazzani, M. J., & Billsus, D. (2007).** "Content-Based Recommendation Systems". Springer.
4. **Lu, J., et al. (2015).** "Recommender System Application Developments: A Survey". Decision Support Systems.
5. **Shani, G., & Gunawardana, A. (2011).** "Evaluating Recommendation Systems". Springer.

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademis - Proyek Sains Data ITS.

---

## 📞 Kontak

Untuk pertanyaan atau feedback:
- **Ghalib Ibrahim Zardy:** 5052231028
- **M Shah Aquilla Febryano:** 5052231043

---

## 🌟 Features Checklist

- ✅ Content-Based Filtering Implementation
- ✅ TF-IDF Vectorization
- ✅ Cosine Similarity Calculation
- ✅ Multi-parameter Filtering (Genre, Budget, RAM)
- ✅ Interactive UI with Dark Theme
- ✅ Real-time Recommendations
- ✅ Comprehensive Metrics Dashboard
- ✅ Multiple Data Visualizations
- ✅ EDA with Statistical Analysis
- ✅ Complete Documentation (Step 1-7)
- ✅ User-friendly Interface
- ✅ Performance Monitoring
- ✅ Responsive Design

---

**Developed with ❤️ by Kelompok 12**  
**Institut Teknologi Sepuluh Nopember**
