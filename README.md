# RasaKu 🍳

**RasaKu** adalah aplikasi web canggih yang memberikan **rekomendasi resep masakan** berdasarkan preferensi dan kebutuhan pengguna. Dirancang untuk mempermudah pencarian resep yang *sehat*, *lezat*, dan sesuai dengan gaya hidup atau kondisi kesehatan masing-masing pengguna.

![RasaKu](https://img.shields.io/badge/version-1.0-blue.svg) ![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)

## 📌 Fitur Utama

- **Rekomendasi Resep Personalisasi**: Dapatkan resep yang disesuaikan dengan preferensi, bahan yang tersedia, atau kebutuhan diet Anda.
- **Dataset Resep**: Menyediakan kumpulan resep masakan yang dapat diproses dan diubah sesuai kebutuhan.
- **Aplikasi Web Interaktif**: Pengguna dapat berinteraksi langsung melalui antarmuka berbasis web untuk menemukan resep.

## 🌟 Fitur Spesial

- 🥗 **Diet Sehat**: Pilih resep berdasarkan jenis diet yang Anda jalani (misalnya, keto, vegan, dll).
- 🍲 **Rekomendasi Bahan**: Temukan resep berdasarkan bahan makanan yang ada di rumah Anda.
- 🌍 **Global Recipes**: Temukan resep masakan dari seluruh dunia, mulai dari masakan Indonesia hingga masakan Barat.

## 🗂️ Struktur Repository

/
├── app.py # Main application untuk menjalankan web app
├── generate_recipes_dataset.py # Script untuk membangun dataset resep
├── recipes.csv # Dataset resep masakan
├── instance/ # Konfigurasi aplikasi (mis. database)
├── templates/ # Folder HTML templates untuk web
└── venv/ # Virtual environment Python


## 🚀 Cara Instalasi & Run

### 1. **Clone repository**
   Mulai dengan mengkloning repository ini ke lokal Anda:
   ```bash
   git clone https://github.com/wizdanilyumnaa/RasaKu.git
   cd RasaKu
    python -m venv venv
    source venv/bin/activate      # macOS/Linux
    venv\Scripts\activate         # Windows
    pip install -r requirements.txt
    python app.py

