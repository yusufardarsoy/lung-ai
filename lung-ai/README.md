# 🫁 LUNG-AI: Akciğer BT Kanser Teşhis Sistemi

![Status](https://img.shields.io/badge/status-MVP-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Derin öğrenme ve Transfer Learning kullanarak akciğer BT (CT) görüntülerinden kanser riskini tespit eden yapay zeka sistemi.

> 💡 **Bu, bir MVP ve eğitim amaçlı prototiptir.** Kesin tıbbi teşhis için uzman radyolog görüşü gereklidir.

---

## 📋 İçindekiler

- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Proje Yapısı](#proje-yapısı)
- [Mimarisi & ML Konseptleri](#mimarisi--ml-konseptleri)
- [Nasıl Çalışır](#nasıl-çalışır)
- [Kodu Anlama Rehberi](#kodu-anlama-rehberi)
- [Geliştirme & İyileştirmeler](#geliştirme--iyileştirmeler)
- [GitHub'a Yüklemek](#githuba-yüklemek)

---

## 🚀 Hızlı Başlangıç

### 1️⃣ Ortam Kurulumu

```bash
# Python 3.8+ gerekli

# Virtual environment oluştur (opsiyonel ama tavsiye edilir)
python -m venv venv

# Virtual environment'ı aktifleştir
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2️⃣ Uygulamayı Çalıştır

```bash
streamlit run app.py
```

Tarayıcınızda otomatik açılacak: `http://localhost:8501`

### 3️⃣ Test Et

- BT görüntüsü (JPG/PNG) seçin
- Sistem tahmin yapar
- Sonuçları görüntüleyin

---

## 📁 Proje Yapısı

```
lung-ai/
├── app.py                 # 🎨 Streamlit web arayüzü
├── model.py               # 🧠 ML model ve Transfer Learning
├── requirements.txt       # 📦 Python bağımlılıkları
├── README.md             # 📖 Bu dosya
└── .gitignore            # ⚙️ Git ignore kuralları
```

### Dosya Açıklamaları

| Dosya | Rol | Öğreneceksin |
|-------|-----|-------------|
| `model.py` | ML mimarisi ve model yönetimi | Transfer Learning, ResNet50, Keras |
| `app.py` | Web arayüzü | Streamlit, UI/UX, data visualization |
| `requirements.txt` | Bağımlılıklar | Python dependency management |

---

## 🤖 Mimarisi & ML Konseptleri

### Transfer Learning Nedir?

**Sorun:** Sıfırdan eksiksiz bir CNN eğitmek istersen:
- 🔴 Çok fazla veri lazım (100K+ görüntü)
- 🔴 Çok fazla GPU saati (haftalar)
- 🔴 Kompleks tuning

**Çözüm: Transfer Learning**

```
ImageNet (1.2M görüntü, 1000 sınıf)
        ↓
   ResNet50 Eğitilir
 (edge, texture, shapes öğrenir)
        ↓
Ağırlıkları dondur + Kendi katmanları ekle
        ↓
Tıbbi veriye adapte et (fine-tuning)
```

### Model Mimarisi

```
INPUT: 224×224×3 (RGB Görüntü)
    ↓
RESNET50 (23.5M params, ImageNet pretrained)
    ├─ 50 Katman (Conv, BatchNorm, ReLU)
    ├─ Spatial features öğrenir
    └─ Output: 7×7×2048 feature map
    ↓
GLOBAL AVERAGE POOLING
    ├─ (7,7,2048) → (2048,)
    └─ Spatial info'yu ortalama ile kaybet'ma
    ↓
DENSE(1024, relu) + DROPOUT(0.3)
    └─ High-level patterns öğren
    ↓
DENSE(128, relu) + DROPOUT(0.2)
    √─ Daha kompakt representation
    ↓
OUTPUT: DENSE(3, softmax)
    ├─ Logit 1: P(Sağlıklı)
    ├─ Logit 2: P(Malign)
    └─ Logit 3: P(Benign)
```

### Toplam Parametreler

- **ResNet50:** 23.5M (frozen = eğitilmez)
- **Custom Layers:** 1.3M (trainable)
- **Total:** ~24.8M

---

## 📊 Nasıl Çalışır

### Veri Akışı (Pipeline)

```
1. Kullanıcı görüntü yükler
        ↓
2. Görüntü RGB'ye çevrilir
        ↓
3. 224×224'e yeniden boyutlandırılır
        ↓
4. Pixel değerleri normalize edilir (0-255 → 0-1)
        ↓
5. Model'e stream yapılır
        ↓
6. Tahmin yapılır (inference)
        ↓
7. Sonuçlar gösterilir
```

### Tahmin Süreci (Inference)

```python
# Input: (224, 224, 3) normalized image
image_batch = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)

# Forward pass
predictions = model.predict(image_batch)  # (1, 3)

# Output örneği:
# predictions = [[0.15, 0.75, 0.10]]
#                Sağlıklı=15%, Malign=75%, Benign=10%

predicted_class = np.argmax(predictions[0])  # 1 (Malign)
confidence = predictions[0][1] * 100         # 75%
```

---

## 📚 Kodu Anlama Rehberi

### 1. Model Mimarisini Anlamak (`model.py`)

**Başlangıç: Klasa Tanımı**

```python
class LungAIModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape      # Görüntü boyutu
        self.num_classes = num_classes       # Sınıf sayısı (3)
        self.model = None                    # Model henüz yüklü değil
```

**Daha Sonra: `build_model()` Çağrısı**

```python
# 1. ResNet50 yükle
base_model = ResNet50(weights='imagenet', include_top=False, ...)

# 2. Dondsur (ağırlıkları değiştirme)
base_model.trainable = False

# 3. Kendi katmanlarını ekle
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)  # Flatten
x = Dense(1024, activation='relu')(x)  # Lineer
...
```

**Aktivasyon Fonksiyonları:**

| Fonksiyon | Formül | Kullanım |
|-----------|--------|---------|
| **ReLU** | $f(x) = \max(0, x)$ | Hidden layers'da (non-linearity) |
| **Softmax** | $f(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$ | Output (multi-class) |

**Dropout Nedir?**

```
Without Dropout:        With Dropout (%30 kısılmış):
[●●●●●●●●●●]    →    [●●●○●●○●●○]
 Tüm neurons         Random kısılmış neurons
 (Overfitting riski)   (Genelleme iyileşir)
```

---

### 2. Görüntü Ön İşleme (`app.py` - `preprocess_image()`)

```python
def preprocess_image(image: Image.Image) -> np.ndarray:
    # 1. RGB dönüştür
    image_rgb = image.convert('RGB')
    
    # 2. 224×224'e yeniden boyutlandır
    image_resized = image_rgb.resize((224, 224), ...)
    
    # 3. NumPy array'e dönüştür
    image_array = np.array(image_resized, dtype=np.float32)
    
    # 4. Normalize (0-255 → 0-1)
    image_normalized = image_array / 255.0
    
    return image_normalized  # Shape: (224, 224, 3), Values: 0-1
```

**Neden Normalize?**
- ✅ Gradient descent'i hızlandırır
- ✅ Sayısal kararlılık
- ✅ Ağırlıkları sıfırın civarda tutar

---

### 3. Streamlit Arayüzünü Anlamak (`app.py`)

**Caching Mekanizması:**

```python
@st.cache_resource
def load_model():
    """Ilk sefer yükler, sonra cache'den alır"""
    model = LungAIModel()
    model.build_model()
    return model
```

**Neden cache?**
```
1. Yazı sayfa yüklendi → Model yüklendi (3 saniye)
2. Yazı bir resim seç → Sayfanın tamamı tekrar çalıştı
   ✅ Ama model cache'de → hızlı
```

---

## 🔧 Geliştirme & İyileştirmeler

### Sonraki Adımlar

#### Phase 2: Fine-tuning
```python
# ResNet50'nin son katmanlarını aç (çöz)
base_model.trainable = True

# Sadece son 50 katmanı eğit
for layer in base_model.layers[:-50]:
    layer.trainable = False
```

#### Phase 3: Grad-CAM (Visualization)
```python
# Model hangi bölgeye bakmıştır? Görselleştir
# Çıkış: Heatmap (sıcak bölgeler = önemli)
```

#### Phase 4: Ensemble Models
```python
# Birden fazla model kullan + ortalama al
# - ResNet50
# - EfficientNet
# - DenseNet
```

#### Phase 5: Gerçek Veri Eğitimi
```
# LUNA16 veri seti kullan
# 800+ hasta, 1000+ nodül
# Accuracy: 85-95%
```

---

## 📤 GitHub'a Yüklemek

### 1. Git Repo Başlatılması

```bash
cd lung-ai
git init
git config user.name "Adın"
git config user.email "email@example.com"
```

### 2. İlk Commit

```bash
git add .
git commit -m "Initial commit: LUNG-AI MVP with Transfer Learning"
```

### 3. GitHub'a Bağlama

```bash
# GitHub'da lung-ai repo oluştur

git remote add origin https://github.com/USERNAME/lung-ai.git
git branch -M main
git push -u origin main
```

### 4. GitHub Sayfada Görünecek

```
✅ Files:
   - app.py
   - model.py
   - requirements.txt
   - README.md
   - .gitignore

✅ Description:
   LUNG-AI: Deep Learning for CT Lung Cancer Detection (MVP)

✅ Topics:
   deep-learning, transfer-learning, medical-imaging, streamlit, tensorflow
```

---

## 🎓 Öğrenme Kaynakları

### Transfer Learning
- [FastAI Course - Part 1](https://course.fast.ai/) - Pratik yaklaşım
- [CS231n - Transfer Learning](http://cs231n.stanford.edu/)

### ResNet50 Detaylı
- [He et al. 2015 - Deep Residual Learning](https://arxiv.org/abs/1512.03385)

### Medical Imaging
- [LUNA16 Dataset](https://luna16.grand-challenge.org/)
- [Grad-CAM Visualization](https://arxiv.org/abs/1610.02055)

### Streamlit
- [Streamlit Docs](https://docs.streamlit.io/)
- [Streamlit Component Gallery](https://streamlit.io/components)

---

## 📋 Checklist: Kod Anlama

- [ ] `model.py`'deki `build_model()` fonksiyonunu oku ve anla
- [ ] Transfer Learning vs From Scratch farkını öğren
- [ ] Dropout, ReLU, Softmax nedir öğren
- [ ] `app.py`'de `@st.cache_resource` ne yapar oku
- [ ] `preprocess_image()` fonksiyonunun adımlarını takip et
- [ ] Görüntü normalize işlemini anla (neden 0-1?)
- [ ] Model architecture'ını kağıda çiz
- [ ] Bir CT görüntü yükle ve tahmin yaptır

---

## ⚖️ Lisans

MIT License - Özgürce kullan ve dağıt.

---

## 🤝 Katkılar

Bug bulursan veya iyileştirme fikirn varsa:
1. Fork repo
2. Feature branch oluştur (`git checkout -b feature/AmazingFeature`)
3. Commit et (`git commit -m 'Add AmazingFeature'`)
4. Push et (`git push origin feature/AmazingFeature`)
5. Pull Request aç

---

## ⚠️ Yasal Uyarı

```
LUNG-AI, yapay zeka tarafından oluşturulmuş bir yapıdır ve
KESINLIKLE tıbbi teşhis veya tedavi kararı almak için kullanılmaz.

Kullanım Şartları:
- ✅ Eğitim amaçlı
- ✅ Araştırma ve prototipleme
- ❌ Klinik kullanım
- ❌ Kesin teşhis
- ❌ Tıbbi tavsiye

Sorumluluk:
Sistem kullanımından kaynaklanan herhangi bir zarar için
geliştirici sorumlu değildir.
```

---

**Başarılar! Mekanizmayı adım adım kavraman dilerim. 🚀**
