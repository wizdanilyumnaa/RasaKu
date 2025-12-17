from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =======================================
# KONFIGURASI DASAR FLASK
# =======================================
app = Flask(__name__)
app.secret_key = 'rahasia-super-kuat'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

OPENWEATHER_API_KEY = '2c3712cc809ae5608109d60c3cf7aba9'

db = SQLAlchemy(app)

# =======================================
# MODEL DATABASE
# =======================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)

class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    recipe_name = db.Column(db.String(200))

class HealthProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True)
    
    # Status Gizi & Tujuan Diet
    diet_goal = db.Column(db.String(50))  # menurunkan_berat, menaikkan_berat, menjaga_berat, sehat_umum
    activity_level = db.Column(db.String(50))  # rendah, sedang, tinggi, sangat_tinggi
    
    # Preferensi Diet
    diet_preference = db.Column(db.String(100))  # normal, vegetarian, vegan, pescatarian, halal
    
    # Kondisi Kesehatan
    health_conditions = db.Column(db.Text)  # diabetes, hipertensi, kolesterol, asam_urat, maag (comma separated)
    
    # Alergi & Intoleransi
    allergies = db.Column(db.Text)  # kacang, seafood, telur, susu, gluten (comma separated)
    
    # Preferensi Nutrisi
    low_sugar = db.Column(db.Boolean, default=False)
    low_salt = db.Column(db.Boolean, default=False)
    low_fat = db.Column(db.Boolean, default=False)
    high_protein = db.Column(db.Boolean, default=False)
    high_fiber = db.Column(db.Boolean, default=False)
    
    # Timestamp
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())


# =======================================
# FUNGSI HELPER UNTUK CUACA
# =======================================
def get_weather_data(city="Jakarta"):
    """Mengambil data cuaca dari OpenWeatherMap API"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},ID&appid={OPENWEATHER_API_KEY}&units=metric&lang=id"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None

def get_season_indonesia():
    """Menentukan musim di Indonesia (kemarau/hujan)"""
    from datetime import datetime
    month = datetime.now().month
    # Musim hujan: Oktober - Maret
    # Musim kemarau: April - September
    if month >= 10 or month <= 3:
        return "hujan"
    else:
        return "kemarau"

def get_weather_category(weather_data):
    """Mengkategorikan cuaca berdasarkan data API"""
    if not weather_data:
        return "normal", 25
    
    temp = weather_data.get('main', {}).get('temp', 25)
    weather_main = weather_data.get('weather', [{}])[0].get('main', '').lower()
    weather_desc = weather_data.get('weather', [{}])[0].get('description', '')
    
    # Kategorisasi berdasarkan suhu dan kondisi cuaca
    if temp >= 32:
        category = "panas"
    elif temp <= 22:
        category = "dingin"
    elif 'rain' in weather_main or 'drizzle' in weather_main:
        category = "hujan"
    elif 'cloud' in weather_main:
        category = "mendung"
    else:
        category = "cerah"
    
    return category, temp

def get_recipe_tags_for_weather(weather_category, season):
    """Menentukan tag resep yang cocok berdasarkan cuaca dan musim"""
    tags = {
        "panas": {
            "keywords": ["segar", "dingin", "es", "salad", "mentah", "buah", "jus", "smoothie", "asinan", "rujak"],
            "categories": ["Minuman", "Cemilan", "Salad"],
            "description": "Cuaca panas! Cocok untuk makanan dan minuman yang menyegarkan."
        },
        "dingin": {
            "keywords": ["hangat", "kuah", "sup", "soto", "bakso", "mie", "gorengan", "pedas", "rempah"],
            "categories": ["Makanan Utama", "Sup", "Kuah"],
            "description": "Cuaca dingin! Pas untuk makanan berkuah dan hangat."
        },
        "hujan": {
            "keywords": ["hangat", "kuah", "gorengan", "pedas", "sup", "soto", "bakso", "mie", "wedang", "jahe"],
            "categories": ["Makanan Utama", "Cemilan", "Minuman"],
            "description": "Sedang hujan! Enaknya makan yang hangat-hangat."
        },
        "mendung": {
            "keywords": ["comfort food", "gurih", "gorengan", "snack", "cemilan", "hangat"],
            "categories": ["Cemilan", "Makanan Utama"],
            "description": "Cuaca mendung, cocok untuk comfort food!"
        },
        "cerah": {
            "keywords": ["segar", "ringan", "sehat", "sayur", "grill", "panggang"],
            "categories": ["Makanan Utama", "Salad"],
            "description": "Cuaca cerah! Cocok untuk masakan ringan dan sehat."
        },
        "normal": {
            "keywords": [],
            "categories": [],
            "description": "Semua makanan cocok untuk cuaca saat ini!"
        }
    }
    
    # Tambahan tag berdasarkan musim
    if season == "hujan":
        tags[weather_category]["keywords"].extend(["jahe", "rempah", "hangat"])
    else:  # kemarau
        tags[weather_category]["keywords"].extend(["segar", "dingin"])
    
    return tags.get(weather_category, tags["normal"])

def get_weather_recommendations(df, weather_category, season, top_n=6):
    """Mendapatkan rekomendasi resep berdasarkan cuaca menggunakan CBF"""
    tags = get_recipe_tags_for_weather(weather_category, season)
    
    # Membuat query dari keywords cuaca
    weather_query = ' '.join(tags["keywords"])
    
    if not weather_query:
        # Jika tidak ada query spesifik, return random recipes
        return df.sample(n=min(top_n, len(df))).to_dict(orient='records')
    
    stopwords_indonesia = [
        'dan', 'atau', 'dengan', 'yang', 'untuk', 'dari', 'di', 'ke', 'pada', 'sebagai', 'ini',
        'itu', 'dalam', 'oleh', 'karena', 'adalah', 'juga', 'akan', 'sudah', 'agar', 'lebih'
    ]
    
    # Gabungkan fitur resep
    df['fitur'] = df['kategori'].fillna('') + ' ' + df['deskripsi'].fillna('') + ' ' + df['bahan'].fillna('') + ' ' + df['nama_resep'].fillna('')
    
    # Tambahkan weather query sebagai dokumen
    all_docs = df['fitur'].tolist() + [weather_query]
    
    vectorizer = TfidfVectorizer(stop_words=stopwords_indonesia)
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    # Hitung similarity antara weather query dan semua resep
    weather_vector = tfidf_matrix[-1]  # Last document is our weather query
    recipe_vectors = tfidf_matrix[:-1]  # All recipes
    
    cosine_sim = cosine_similarity(weather_vector, recipe_vectors).flatten()
    
    df['weather_similarity'] = cosine_sim
    
    # Filter by preferred categories if available
    if tags["categories"]:
        preferred_df = df[df['kategori'].isin(tags["categories"])]
        if len(preferred_df) >= top_n:
            df = preferred_df
    
    # Sort by similarity and return top N
    recommendations = df.sort_values(by='weather_similarity', ascending=False).head(top_n)
    
    return recommendations.to_dict(orient='records')


# =======================================
# FUNGSI HELPER UNTUK KESEHATAN
# =======================================
def get_health_keywords(health_profile):
    """Menghasilkan keywords berdasarkan profil kesehatan pengguna"""
    keywords = {
        'preferred': [],  # Kata kunci yang disukai/direkomendasikan
        'avoided': []     # Kata kunci yang harus dihindari
    }
    
    if not health_profile:
        return keywords
    
    # Berdasarkan tujuan diet
    diet_goal_keywords = {
        'menurunkan_berat': {
            'preferred': ['rendah kalori', 'sayur', 'salad', 'rebus', 'kukus', 'panggang', 'segar', 'ringan', 'diet'],
            'avoided': ['goreng', 'santan', 'berlemak', 'manis', 'gorengan']
        },
        'menaikkan_berat': {
            'preferred': ['protein', 'daging', 'ayam', 'telur', 'karbohidrat', 'nasi', 'roti', 'susu'],
            'avoided': []
        },
        'menjaga_berat': {
            'preferred': ['seimbang', 'sayur', 'protein', 'sehat'],
            'avoided': []
        },
        'sehat_umum': {
            'preferred': ['sehat', 'segar', 'sayur', 'buah', 'protein'],
            'avoided': []
        }
    }
    
    if health_profile.diet_goal in diet_goal_keywords:
        keywords['preferred'].extend(diet_goal_keywords[health_profile.diet_goal]['preferred'])
        keywords['avoided'].extend(diet_goal_keywords[health_profile.diet_goal]['avoided'])
    
    # Berdasarkan preferensi diet
    diet_pref_keywords = {
        'vegetarian': {
            'preferred': ['sayur', 'tahu', 'tempe', 'telur', 'susu', 'vegetarian'],
            'avoided': ['daging', 'ayam', 'sapi', 'kambing', 'bebek', 'ikan', 'seafood', 'udang']
        },
        'vegan': {
            'preferred': ['sayur', 'tahu', 'tempe', 'buah', 'kacang', 'vegan'],
            'avoided': ['daging', 'ayam', 'sapi', 'telur', 'susu', 'keju', 'mentega', 'ikan', 'seafood']
        },
        'pescatarian': {
            'preferred': ['ikan', 'seafood', 'udang', 'sayur', 'tahu', 'tempe'],
            'avoided': ['daging', 'ayam', 'sapi', 'kambing', 'bebek']
        },
        'halal': {
            'preferred': ['halal', 'ayam', 'sapi', 'kambing'],
            'avoided': ['babi', 'pork', 'bacon', 'ham']
        }
    }
    
    if health_profile.diet_preference in diet_pref_keywords:
        keywords['preferred'].extend(diet_pref_keywords[health_profile.diet_preference]['preferred'])
        keywords['avoided'].extend(diet_pref_keywords[health_profile.diet_preference]['avoided'])
    
    # Berdasarkan kondisi kesehatan
    if health_profile.health_conditions:
        conditions = [c.strip() for c in health_profile.health_conditions.split(',')]
        
        condition_keywords = {
            'diabetes': {
                'preferred': ['rendah gula', 'sayur', 'protein', 'serat'],
                'avoided': ['gula', 'manis', 'kecap manis', 'madu', 'sirup', 'kue']
            },
            'hipertensi': {
                'preferred': ['rendah garam', 'sayur', 'buah', 'rebus', 'kukus'],
                'avoided': ['garam', 'asin', 'kecap asin', 'msg', 'gorengan', 'santan']
            },
            'kolesterol': {
                'preferred': ['rendah lemak', 'sayur', 'ikan', 'rebus', 'kukus', 'panggang'],
                'avoided': ['goreng', 'santan', 'kulit ayam', 'jeroan', 'gorengan', 'lemak']
            },
            'asam_urat': {
                'preferred': ['sayur', 'buah', 'air', 'rendah purin'],
                'avoided': ['jeroan', 'seafood', 'kacang', 'melinjo', 'emping', 'daging merah']
            },
            'maag': {
                'preferred': ['lembut', 'rebus', 'kukus', 'bubur', 'sup'],
                'avoided': ['pedas', 'asam', 'goreng', 'cabai', 'cuka', 'jeruk']
            }
        }
        
        for condition in conditions:
            if condition in condition_keywords:
                keywords['preferred'].extend(condition_keywords[condition]['preferred'])
                keywords['avoided'].extend(condition_keywords[condition]['avoided'])
    
    # Berdasarkan alergi
    if health_profile.allergies:
        allergies = [a.strip() for a in health_profile.allergies.split(',')]
        
        allergy_keywords = {
            'kacang': ['kacang', 'peanut', 'almond', 'mete', 'kenari'],
            'seafood': ['udang', 'kepiting', 'cumi', 'kerang', 'ikan', 'seafood'],
            'telur': ['telur', 'egg', 'telor'],
            'susu': ['susu', 'keju', 'mentega', 'cream', 'yogurt', 'dairy'],
            'gluten': ['tepung terigu', 'roti', 'mie', 'pasta', 'gandum']
        }
        
        for allergy in allergies:
            if allergy in allergy_keywords:
                keywords['avoided'].extend(allergy_keywords[allergy])
    
    # Berdasarkan preferensi nutrisi
    if health_profile.low_sugar:
        keywords['preferred'].append('rendah gula')
        keywords['avoided'].extend(['gula', 'manis', 'sirup', 'madu'])
    
    if health_profile.low_salt:
        keywords['preferred'].append('rendah garam')
        keywords['avoided'].extend(['garam', 'asin', 'msg'])
    
    if health_profile.low_fat:
        keywords['preferred'].extend(['rendah lemak', 'rebus', 'kukus', 'panggang'])
        keywords['avoided'].extend(['goreng', 'santan', 'minyak', 'lemak'])
    
    if health_profile.high_protein:
        keywords['preferred'].extend(['protein', 'daging', 'ayam', 'ikan', 'telur', 'tahu', 'tempe'])
    
    if health_profile.high_fiber:
        keywords['preferred'].extend(['serat', 'sayur', 'buah', 'kacang', 'whole grain'])
    
    # Hapus duplikat
    keywords['preferred'] = list(set(keywords['preferred']))
    keywords['avoided'] = list(set(keywords['avoided']))
    
    return keywords

def get_health_recommendations(df, health_profile, top_n=6):
    """Mendapatkan rekomendasi resep berdasarkan profil kesehatan menggunakan CBF"""
    keywords = get_health_keywords(health_profile)
    
    stopwords_indonesia = [
        'dan', 'atau', 'dengan', 'yang', 'untuk', 'dari', 'di', 'ke', 'pada', 'sebagai', 'ini',
        'itu', 'dalam', 'oleh', 'karena', 'adalah', 'juga', 'akan', 'sudah', 'agar', 'lebih'
    ]
    
    # Gabungkan fitur resep
    df['fitur'] = (
        df['kategori'].fillna('') + ' ' + 
        df['deskripsi'].fillna('') + ' ' + 
        df['bahan'].fillna('') + ' ' + 
        df['nama_resep'].fillna('') + ' ' +
        df['cara_memasak'].fillna('')
    )
    
    # Buat query dari preferred keywords
    health_query = ' '.join(keywords['preferred'])
    
    if not health_query:
        # Jika tidak ada preferensi spesifik, return semua resep yang tidak mengandung avoided keywords
        filtered_df = df.copy()
        for avoided in keywords['avoided']:
            filtered_df = filtered_df[~filtered_df['fitur'].str.lower().str.contains(avoided.lower(), na=False)]
        return filtered_df.head(top_n).to_dict(orient='records')
    
    # Vectorize dengan TF-IDF
    all_docs = df['fitur'].tolist() + [health_query]
    
    vectorizer = TfidfVectorizer(stop_words=stopwords_indonesia)
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    # Hitung similarity
    health_vector = tfidf_matrix[-1]
    recipe_vectors = tfidf_matrix[:-1]
    
    cosine_sim = cosine_similarity(health_vector, recipe_vectors).flatten()
    df['health_similarity'] = cosine_sim
    
    # Filter resep yang mengandung bahan yang harus dihindari
    filtered_df = df.copy()
    for avoided in keywords['avoided']:
        # Turunkan skor similarity untuk resep yang mengandung bahan yang harus dihindari
        mask = filtered_df['fitur'].str.lower().str.contains(avoided.lower(), na=False)
        filtered_df.loc[mask, 'health_similarity'] *= 0.1  # Kurangi 90% skor
    
    # Sort by similarity dan return top N
    recommendations = filtered_df.sort_values(by='health_similarity', ascending=False).head(top_n)
    
    return recommendations.to_dict(orient='records')

def get_health_summary(health_profile):
    """Menghasilkan ringkasan profil kesehatan untuk ditampilkan"""
    summary = {
        'diet_goal': {
            'menurunkan_berat': 'Menurunkan Berat Badan',
            'menaikkan_berat': 'Menaikkan Berat Badan',
            'menjaga_berat': 'Menjaga Berat Badan',
            'sehat_umum': 'Pola Makan Sehat Umum'
        }.get(health_profile.diet_goal, 'Belum diatur'),
        
        'activity_level': {
            'rendah': 'Rendah (Jarang olahraga)',
            'sedang': 'Sedang (1-3x/minggu)',
            'tinggi': 'Tinggi (4-5x/minggu)',
            'sangat_tinggi': 'Sangat Tinggi (Setiap hari)'
        }.get(health_profile.activity_level, 'Belum diatur'),
        
        'diet_preference': {
            'normal': 'Normal (Semua jenis makanan)',
            'vegetarian': 'Vegetarian',
            'vegan': 'Vegan',
            'pescatarian': 'Pescatarian',
            'halal': 'Halal'
        }.get(health_profile.diet_preference, 'Belum diatur'),
        
        'health_conditions': [],
        'allergies': [],
        'nutrient_preferences': []
    }
    
    # Parse kondisi kesehatan
    condition_labels = {
        'diabetes': 'Diabetes',
        'hipertensi': 'Hipertensi/Darah Tinggi',
        'kolesterol': 'Kolesterol Tinggi',
        'asam_urat': 'Asam Urat',
        'maag': 'Maag/GERD'
    }
    
    if health_profile.health_conditions:
        for condition in health_profile.health_conditions.split(','):
            condition = condition.strip()
            if condition in condition_labels:
                summary['health_conditions'].append(condition_labels[condition])
    
    # Parse alergi
    allergy_labels = {
        'kacang': 'Kacang-kacangan',
        'seafood': 'Seafood',
        'telur': 'Telur',
        'susu': 'Susu & Produk Dairy',
        'gluten': 'Gluten'
    }
    
    if health_profile.allergies:
        for allergy in health_profile.allergies.split(','):
            allergy = allergy.strip()
            if allergy in allergy_labels:
                summary['allergies'].append(allergy_labels[allergy])
    
    # Preferensi nutrisi
    if health_profile.low_sugar:
        summary['nutrient_preferences'].append('Rendah Gula')
    if health_profile.low_salt:
        summary['nutrient_preferences'].append('Rendah Garam')
    if health_profile.low_fat:
        summary['nutrient_preferences'].append('Rendah Lemak')
    if health_profile.high_protein:
        summary['nutrient_preferences'].append('Tinggi Protein')
    if health_profile.high_fiber:
        summary['nutrient_preferences'].append('Tinggi Serat')
    
    return summary


# =======================================
# ROUTE REGISTER
# =======================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username sudah digunakan!', 'danger')
            return redirect(url_for('register'))

        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email sudah digunakan!', 'danger')
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        new_user = User(username=username, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        flash('Akun berhasil dibuat! Silakan login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# =======================================
# ROUTE LOGIN
# =======================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Berhasil login!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Email atau password salah!', 'danger')

    return render_template('login.html')

# =======================================
# ROUTE LOGOUT
# =======================================
@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah logout.', 'info')
    return redirect(url_for('home'))

# =======================================
# ROUTE PROFIL USER
# =======================================
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Silakan login untuk melihat profil.', 'warning')
        return redirect(url_for('login', next=url_for('profile')))

    user = User.query.get(session['user_id'])
    if not user:
        flash('User tidak ditemukan.', 'danger')
        return redirect(url_for('logout'))

    favorite_count = Favorite.query.filter_by(user_id=user.id).count()
    recommendation_count = min(6, favorite_count) if favorite_count > 0 else 0
    
    health_profile = HealthProfile.query.filter_by(user_id=user.id).first()
    has_health_profile = health_profile is not None

    return render_template('profile.html', 
                           user=user, 
                           favorite_count=favorite_count,
                           recommendation_count=recommendation_count,
                           has_health_profile=has_health_profile,
                           is_logged_in=True,
                           username=session.get('username'))

# =======================================
# UPDATE PROFIL
# =======================================
@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user:
        flash('User tidak ditemukan.', 'danger')
        return redirect(url_for('logout'))

    new_username = request.form.get('username', '').strip()
    
    if not new_username or len(new_username) < 3:
        flash('Username harus minimal 3 karakter.', 'danger')
        return redirect(url_for('profile'))

    existing_user = User.query.filter(User.username == new_username, User.id != user.id).first()
    if existing_user:
        flash('Username sudah digunakan oleh user lain.', 'danger')
        return redirect(url_for('profile'))

    user.username = new_username
    session['username'] = new_username
    db.session.commit()
    
    flash('Profil berhasil diperbarui!', 'success')
    return redirect(url_for('profile'))

# =======================================
# GANTI PASSWORD
# =======================================
@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user:
        flash('User tidak ditemukan.', 'danger')
        return redirect(url_for('logout'))

    current_password = request.form.get('current_password', '')
    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')

    if not check_password_hash(user.password, current_password):
        flash('Password saat ini salah.', 'danger')
        return redirect(url_for('profile'))

    if len(new_password) < 6:
        flash('Password baru harus minimal 6 karakter.', 'danger')
        return redirect(url_for('profile'))

    if new_password != confirm_password:
        flash('Konfirmasi password tidak cocok.', 'danger')
        return redirect(url_for('profile'))

    user.password = generate_password_hash(new_password, method='pbkdf2:sha256', salt_length=16)
    db.session.commit()

    flash('Password berhasil diubah!', 'success')
    return redirect(url_for('profile'))

# =======================================
# HAPUS AKUN
# =======================================
@app.route('/delete_account')
def delete_account():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user:
        flash('User tidak ditemukan.', 'danger')
        return redirect(url_for('logout'))

    HealthProfile.query.filter_by(user_id=user.id).delete()
    Favorite.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    session.clear()
    
    flash('Akun Anda telah dihapus.', 'info')
    return redirect(url_for('home'))

# =======================================
# ROUTE HOME - Dengan Pagination
# =======================================
@app.route('/')
def home():
    if not os.path.exists('recipes.csv'):
        flash('Dataset recipes.csv tidak ditemukan!', 'warning')
        return render_template('home.html', 
                               username=session.get('username'), 
                               recipes=[], 
                               is_logged_in='user_id' in session,
                               page=1, total_pages=0, total_recipes=0, per_page=6, page_range=[])

    df = pd.read_csv('recipes.csv')
    df['gambar'] = df['gambar_url']
    
    per_page = 6
    total_recipes = len(df)
    total_pages = (total_recipes + per_page - 1) // per_page
    
    page = request.args.get('page', 1, type=int)
    
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    recipes = df.iloc[start_idx:end_idx].to_dict(orient='records')
    
    page_range_start = max(1, page - 2)
    page_range_end = min(total_pages, page + 2)
    page_range = list(range(page_range_start, page_range_end + 1))

    return render_template('home.html', 
                           username=session.get('username'), 
                           recipes=recipes, 
                           is_logged_in='user_id' in session,
                           page=page,
                           total_pages=total_pages,
                           total_recipes=total_recipes,
                           per_page=per_page,
                           page_range=page_range)

# =======================================
# DETAIL RESEP
# =======================================
@app.route('/resep/<nama_resep>')
def detail_resep(nama_resep):
    df = pd.read_csv('recipes.csv')
    df['gambar'] = df['gambar_url']

    if nama_resep not in df['nama_resep'].values:
        flash('Resep tidak ditemukan.', 'danger')
        return redirect(url_for('home'))

    resep = df[df['nama_resep'] == nama_resep].iloc[0].to_dict()

    is_favorited = False
    if 'user_id' in session:
        user_id = session['user_id']
        is_favorited = Favorite.query.filter_by(user_id=user_id, recipe_name=nama_resep).first() is not None

    return render_template('detail_resep.html', 
                           resep=resep, 
                           is_favorited=is_favorited,
                           is_logged_in='user_id' in session,
                           username=session.get('username'))

# =======================================
# TAMBAH FAVORIT
# =======================================
@app.route('/add_favorite/<nama_resep>', methods=['POST'])
def add_favorite(nama_resep):
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu untuk menambahkan favorit.', 'warning')
        return redirect(url_for('login', next=url_for('detail_resep', nama_resep=nama_resep)))

    user_id = session['user_id']
    existing = Favorite.query.filter_by(user_id=user_id, recipe_name=nama_resep).first()
    if existing:
        flash('Resep sudah ada di daftar favorit kamu!', 'info')
    else:
        new_fav = Favorite(user_id=user_id, recipe_name=nama_resep)
        db.session.add(new_fav)
        db.session.commit()
        flash('Resep berhasil ditambahkan ke favorit!', 'success')

    return redirect(url_for('detail_resep', nama_resep=nama_resep))

# =======================================
# HAPUS FAVORIT
# =======================================
@app.route('/remove_favorite/<nama_resep>', methods=['POST'])
def remove_favorite(nama_resep):
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))

    user_id = session['user_id']
    favorite = Favorite.query.filter_by(user_id=user_id, recipe_name=nama_resep).first()
    if favorite:
        db.session.delete(favorite)
        db.session.commit()
        flash('Resep berhasil dihapus dari favorit.', 'success')
    else:
        flash('Resep tidak ditemukan di daftar favorit.', 'info')

    referrer = request.referrer
    if referrer and 'favorit' in referrer:
        return redirect(url_for('favorit'))
    else:
        return redirect(url_for('detail_resep', nama_resep=nama_resep))

# =======================================
# HALAMAN FAVORIT
# =======================================
@app.route('/favorit')
def favorit():
    if 'user_id' not in session:
        flash('Silakan login untuk melihat resep favorit kamu.', 'warning')
        return redirect(url_for('login', next=url_for('favorit')))

    user_id = session['user_id']
    favorites = Favorite.query.filter_by(user_id=user_id).all()
    favorite_names = [f.recipe_name for f in favorites]

    df = pd.read_csv('recipes.csv')
    df['gambar'] = df['gambar_url']

    if not favorite_names:
        flash('Belum ada resep favorit.', 'info')
        return render_template('favorit.html', favorites=[], username=session['username'], is_logged_in=True)

    fav_recipes = df[df['nama_resep'].isin(favorite_names)].to_dict(orient='records')

    return render_template('favorit.html', favorites=fav_recipes, username=session['username'], is_logged_in=True)

# =======================================
# REKOMENDASI PERSONAL
# =======================================
@app.route('/rekomendasi_personal')
def rekomendasi_personal():
    if 'user_id' not in session:
        flash('Silakan login untuk mendapatkan rekomendasi personal.', 'warning')
        return redirect(url_for('login', next=url_for('rekomendasi_personal')))

    user_id = session['user_id']
    favorites = Favorite.query.filter_by(user_id=user_id).all()
    favorite_names = [f.recipe_name for f in favorites]

    if not favorite_names:
        flash('Tambahkan beberapa resep ke favorit terlebih dahulu.', 'info')
        return redirect(url_for('home'))

    df = pd.read_csv('recipes.csv')
    df['gambar'] = df['gambar_url']

    stopwords_indonesia = [
        'dan', 'atau', 'dengan', 'yang', 'untuk', 'dari', 'di', 'ke', 'pada', 'sebagai', 'ini',
        'itu', 'dalam', 'oleh', 'karena', 'adalah', 'juga', 'akan', 'sudah', 'agar', 'lebih'
    ]

    df['fitur'] = df['kategori'] + ' ' + df['deskripsi'] + ' ' + df['bahan']
    vectorizer = TfidfVectorizer(stop_words=stopwords_indonesia)
    tfidf_matrix = vectorizer.fit_transform(df['fitur'])

    fav_indices = [df[df['nama_resep'] == name].index[0] for name in favorite_names if name in df['nama_resep'].values]
    if not fav_indices:
        flash('Resep favorit tidak ditemukan di dataset.', 'warning')
        return redirect(url_for('home'))

    user_profile = tfidf_matrix[fav_indices].mean(axis=0)
    user_profile = np.asarray(user_profile)
    cosine_sim = cosine_similarity(user_profile, tfidf_matrix).flatten()

    df['similarity'] = cosine_sim
    rekomendasi = (
        df[~df['nama_resep'].isin(favorite_names)]
        .sort_values(by='similarity', ascending=False)
        .head(6)
        .to_dict(orient='records')
    )

    return render_template('rekomendasi.html', rekomendasi=rekomendasi, username=session['username'], is_logged_in=True)


# =======================================
# REKOMENDASI CUACA
# =======================================
@app.route('/rekomendasi_cuaca')
def rekomendasi_cuaca():
    # Get city from query parameter or default to Jakarta
    city = request.args.get('city', 'Jakarta')
    
    # Get weather data
    weather_data = get_weather_data(city)
    weather_category, temperature = get_weather_category(weather_data)
    season = get_season_indonesia()
    
    # Prepare weather info for template
    if weather_data:
        weather_info = {
            'city': weather_data.get('name', city),
            'country': weather_data.get('sys', {}).get('country', 'ID'),
            'temperature': round(temperature),
            'description': weather_data.get('weather', [{}])[0].get('description', 'Tidak tersedia'),
            'icon': weather_data.get('weather', [{}])[0].get('icon', '01d'),
            'humidity': weather_data.get('main', {}).get('humidity', 0),
            'wind_speed': weather_data.get('wind', {}).get('speed', 0),
            'category': weather_category,
            'season': season
        }
    else:
        weather_info = {
            'city': city,
            'country': 'ID',
            'temperature': 25,
            'description': 'Data tidak tersedia',
            'icon': '01d',
            'humidity': 0,
            'wind_speed': 0,
            'category': 'normal',
            'season': season
        }
    
    # Get recipe tags for display
    weather_tags = get_recipe_tags_for_weather(weather_category, season)
    
    # Load recipes
    if not os.path.exists('recipes.csv'):
        flash('Dataset recipes.csv tidak ditemukan!', 'warning')
        return redirect(url_for('home'))
    
    df = pd.read_csv('recipes.csv')
    df['gambar'] = df['gambar_url']
    
    # Get weather-based recommendations
    rekomendasi = get_weather_recommendations(df, weather_category, season, top_n=6)
    
    # List of Indonesian cities for dropdown
    cities = [
        'Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang', 
        'Makassar', 'Palembang', 'Tangerang', 'Depok', 'Bekasi',
        'Yogyakarta', 'Denpasar', 'Malang', 'Bogor', 'Balikpapan',
        'Manado', 'Padang', 'Pontianak', 'Banjarmasin', 'Pekanbaru'
    ]
    
    return render_template('rekomendasi_cuaca.html', 
                           rekomendasi=rekomendasi,
                           weather_info=weather_info,
                           weather_tags=weather_tags,
                           cities=cities,
                           selected_city=city,
                           username=session.get('username'),
                           is_logged_in='user_id' in session)


# =======================================
# PROFIL KESEHATAN
# =======================================
@app.route('/profil_kesehatan', methods=['GET', 'POST'])
def profil_kesehatan():
    if 'user_id' not in session:
        flash('Silakan login untuk mengatur profil kesehatan.', 'warning')
        return redirect(url_for('login', next=url_for('profil_kesehatan')))
    
    user_id = session['user_id']
    health_profile = HealthProfile.query.filter_by(user_id=user_id).first()
    
    if request.method == 'POST':
        # Ambil data dari form
        diet_goal = request.form.get('diet_goal', '')
        activity_level = request.form.get('activity_level', '')
        diet_preference = request.form.get('diet_preference', 'normal')
        
        # Kondisi kesehatan (multiple checkbox)
        health_conditions = ','.join(request.form.getlist('health_conditions'))
        
        # Alergi (multiple checkbox)
        allergies = ','.join(request.form.getlist('allergies'))
        
        # Preferensi nutrisi
        low_sugar = 'low_sugar' in request.form
        low_salt = 'low_salt' in request.form
        low_fat = 'low_fat' in request.form
        high_protein = 'high_protein' in request.form
        high_fiber = 'high_fiber' in request.form
        
        if health_profile:
            # Update existing profile
            health_profile.diet_goal = diet_goal
            health_profile.activity_level = activity_level
            health_profile.diet_preference = diet_preference
            health_profile.health_conditions = health_conditions
            health_profile.allergies = allergies
            health_profile.low_sugar = low_sugar
            health_profile.low_salt = low_salt
            health_profile.low_fat = low_fat
            health_profile.high_protein = high_protein
            health_profile.high_fiber = high_fiber
        else:
            # Create new profile
            health_profile = HealthProfile(
                user_id=user_id,
                diet_goal=diet_goal,
                activity_level=activity_level,
                diet_preference=diet_preference,
                health_conditions=health_conditions,
                allergies=allergies,
                low_sugar=low_sugar,
                low_salt=low_salt,
                low_fat=low_fat,
                high_protein=high_protein,
                high_fiber=high_fiber
            )
            db.session.add(health_profile)
        
        db.session.commit()
        flash('Profil kesehatan berhasil disimpan!', 'success')
        return redirect(url_for('rekomendasi_kesehatan'))
    
    # Prepare current values for form
    current_conditions = []
    current_allergies = []
    
    if health_profile:
        if health_profile.health_conditions:
            current_conditions = [c.strip() for c in health_profile.health_conditions.split(',')]
        if health_profile.allergies:
            current_allergies = [a.strip() for a in health_profile.allergies.split(',')]
    
    return render_template('profil_kesehatan.html',
                           health_profile=health_profile,
                           current_conditions=current_conditions,
                           current_allergies=current_allergies,
                           username=session.get('username'),
                           is_logged_in=True)


# =======================================
# REKOMENDASI KESEHATAN
# =======================================
@app.route('/rekomendasi_kesehatan')
def rekomendasi_kesehatan():
    if 'user_id' not in session:
        flash('Silakan login untuk mendapatkan rekomendasi kesehatan.', 'warning')
        return redirect(url_for('login', next=url_for('rekomendasi_kesehatan')))
    
    user_id = session['user_id']
    health_profile = HealthProfile.query.filter_by(user_id=user_id).first()
    
    if not health_profile:
        flash('Lengkapi profil kesehatan Anda terlebih dahulu untuk mendapatkan rekomendasi.', 'info')
        return redirect(url_for('profil_kesehatan'))
    
    if not os.path.exists('recipes.csv'):
        flash('Dataset recipes.csv tidak ditemukan!', 'warning')
        return redirect(url_for('home'))
    
    df = pd.read_csv('recipes.csv')
    df['gambar'] = df['gambar_url']
    
    # Get health-based recommendations
    rekomendasi = get_health_recommendations(df, health_profile, top_n=6)
    
    # Get health summary for display
    health_summary = get_health_summary(health_profile)
    
    # Get keywords for display
    health_keywords = get_health_keywords(health_profile)
    
    return render_template('rekomendasi_kesehatan.html',
                           rekomendasi=rekomendasi,
                           health_profile=health_profile,
                           health_summary=health_summary,
                           health_keywords=health_keywords,
                           username=session.get('username'),
                           is_logged_in=True)


# =======================================
# MAIN
# =======================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
