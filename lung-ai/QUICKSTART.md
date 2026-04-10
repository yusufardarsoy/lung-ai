# 🚀 LUNG-AI: Hızlı Başlangıç Rehberi

Projeyi **2 dakikada** hazırla ve çalıştır!

---

## 1️⃣ Kurulum (1 dakika)

```bash
# GitHub reposundan clone et (veya klasörü indir)
cd lung-ai

# Python 3.8+ kontrol et
python --version

# Virtual environment oluştur (önerilir)
python -m venv venv

# Aktifleştir
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## 2️⃣ Uygulamayı Başlat (30 saniye)

```bash
streamlit run app.py
```

✅ Tarayıcı otomatik açılacak: **`http://localhost:8501`**

---

## 3️⃣ Test Et (30 saniye)

1. **Bir BT görüntüsü yükle** (JPG/PNG)
   - Kendi görüntün varsa: seç
   - Yoksa: örnek görüntü online bul
2. **Sistem otomatik tahmin yapacak**
3. **Sonuçlar gösterilecek:**
   - Sınıflandırma (Sağlıklı/Malign/Benign)
   - Güven puanı (%)
   - Sınıf dağılımı grafiği

---

## 📚 Kodu Öğrenmeye Başla

### Phase 1: Yapı Tanı (30 dakika)

```bash
# Proje dosyalarını gez
ls -la lung-ai/
# Çıkış:
# - app.py          (Streamlit UI)
# - model.py        (ML Model)
# - requirements.txt (Bağımlılıklar)
# - README.md       (Detaylı docu)
# - CONCEPTS.md     (ML Konseptleri - OKU!)

# Terminal'de model'i test et
python sample_test.py
```

**Çıktısını oku:**
- Model yapısı
- Tahmin örneği
- Parametre sayıları

### Phase 2: Kodları Oku (1-2 saat)

1. **`model.py` OKU** (Başlangıç)
   - `LungAIModel` klası
   - `build_model()` fonksiyonu
   - Detaylı yorumlar var!

2. **`app.py` OKU** (Orta)
   - Streamlit arayüzü
   - Görüntü ön işleme
   - Sonuç gösterilmesi

3. **`CONCEPTS.md` OKU** (Öğrenme)
   - Transfer Learning
   - ResNet50 mimarisi
   - CNN katmanları
   - Tüm ML konseptleri

### Phase 3: Deneyimle (1-2 gün)

Kodu değiştir ve oranları gözlemle:

```python
# model.py'de şunu bul:
x = Dense(1024, activation='relu')(x)  # 1024 node

# Değiştir:
x = Dense(512, activation='relu')(x)   # 512 node

# Sonuç?
# - Daha az parametre (hızlı)
# - Düşük performans (bazen)
```

---

## 🎯 İlk 5 Gün Plan

| Gün | Task | Zaman |
|-----|------|-------|
| 1️⃣ | Çalıştırma + UI gezme | 30 min |
| 2️⃣ | CONCEPTS.md okuma | 1-2 saat |
| 3️⃣ | model.py detay oku | 1-2 saat |
| 4️⃣ | app.py detay oku | 1 saat |
| 5️⃣ | Kod değiştir ve dene | 1-2 saat |

---

## 🐛 Sorunlar & Çözümler

### ❌ `ModuleNotFoundError: No module named 'tensorflow'`

```bash
# Çözüm: Bağımlılıkları yükle
pip install -r requirements.txt

# Veya manuel:
pip install tensorflow==2.13.0 streamlit==1.28.1
```

### ❌ `Port 8501 is already in use`

```bash
# Çözüm: Başka port kullan
streamlit run app.py --server.port 8502
```

### ❌ GPU'ya erişim yok / Çok yavaş

```bash
# Normal - CPU'da normal:
# Model yükleme: 3-5 saniye
# Tahmin: 2-3 saniye

# GPU istersen (opsiyonel):
pip install tensorflow[and-cuda]==2.13.0  # (sadece NVIDIA GPU)
```

---

## 📤 GitHub'a Yükle

```bash
# Repo başlat
git config --global user.name "Adın"
git config --global user.email "email@example.com"
git init

# İlk commit
git add .
git commit -m "Initial: LUNG-AI MVP - Transfer Learning for CT Lung Cancer Detection"

# GitHub'a bağla (GitHub'da repo oluştur önce)
git remote add origin https://github.com/USERNAME/lung-ai.git
git branch -M main
git push -u origin main

# GitHub sayfası:
# ✅ README.md görun
# ✅ Dosyalar listelenir
# ✅ "Star" alabilir :)
```

---

## 🎓 Kaynaklar

**Bu Projede Öğreneceksin:**
- ✅ Transfer Learning
- ✅ ResNet50 mimarisi
- ✅ Keras API
- ✅ Streamlit UI
- ✅ Medical AI (temel)

**Sonra Öğrenebilirsin:**
- Grad-CAM (visualization)
- Data augmentation (training)
- Fine-tuning
- Ensemble models
- Production (Flask, FastAPI)

---

## 🤔 Sık Sorulan Sorular

**S: Model eğitilmiş mi?**
A: Hayır, bu MVP. Pre-trained ResNet50 + random initialization. Gerçek veriye eğitmek istersin.

**S: Kesin teşhis yapabilir mi?**
A: HAYIR. Bu bir yardımcı araçtır. Doktor onayı zorunludur.

**S: Parametre sayısı niçin 24.8M?**
A: ResNet50 = 23.5M. Custom layers = 1.3M. Total = 24.8M.

---

## 💡 Next Steps

1. ✅ Proje çalıştığını gördün
2. ✅ Kodları okuyor musun?
3. **→ Sonra:** LUNA16 veri seti ile eğit
4. **→ Sonra:** Grad-CAM ekle (visualization)
5. **→ Sonra:** Production'a al (API)

---

**Başarılar! 🫁 Mekanizmayı adım adım öğreneceksin.**
