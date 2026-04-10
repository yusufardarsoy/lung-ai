"""
LUNG-AI: ML Konseptleri Detaylı Rehberi
========================================

Bu dosya, kodda karşılaşacağın önemli makine öğrenmesi konseptlerini
açıklıyor. ÖĞRENME VERSİYONÜ.
"""

# ============================================================================
# 1. CNN (Convolutional Neural Network) NEDİR?
# ============================================================================

"""
Neden CNN?

Standart neural network:      CNN:
  Input → Dense → Dense    = Convolutional layers
  (pikselleri sıra sıra)       (görüntü yapısını korur)
  
Avantajları:
  ✅ Görüntü yapısı korunur (pixel komşuluğu)
  ✅ Parametreler paylaşılır (aynı filter birden fazla yerde)
  ✅ Daha az parametre (dense'den 100x az)
  ✅ Özellikler transfer edilebilir (edge → kontür → nesne)

CNN Katmanları:

1. CONVOLUTIONAL LAYER
   - Görüntü üzerinde kayan filtre (kernel)
   - Yerel özellikler öğrenir
   
   ```
   Input: 224×224×3
   Filter: 3×3 (örneğin, edge detector)
   Output: 222×222×filters
   ```

2. POOLING LAYER
   - Dimensionality reduction
   - Max pooling: En büyük aktivasyon
   - Avg pooling: Ortalama aktivasyon
   
   ```
   Input: 7×7
   Pooling size: 2×2
   Output: 3×3 (boyut yarılandı)
   ```

3. BATCH NORMALIZATION
   - Aktivasyonları normalize et
   - Training hızlandır (~10x)
   - Gradient flow iyileşir
   
   ```
   Before: [100, 0.1, 50, 1000]  // Çok farklı scale'ler
   After:  [0.5, -1.2, 0.1, 1.8]  // Sıfır civarında
   ```

4. ACTIVATION FUNCTION
   - ReLU: f(x) = max(0, x)
     √ Negatif değerleri sıfırla
     √ Computational efficiency
   
   - Softmax: Multi-class classification
     √ Output: Olasılıklar (0-1)
     √ Toplamı: 1
     
     Formül:
     softmax(x_i) = exp(x_i) / Σ exp(x_j)
     
     Örnek:
     Input:  [2.0, 1.0, 0.1]
     Output: [0.659, 0.242, 0.099]  // Olasılıklar
"""


# ============================================================================
# 2. TRANSFER LEARNING
# ============================================================================

"""
Transfer Learning = Önceki bilgiyi yeni probleme uyarlama

Analoji (Türkçe konuşmayı öğrenmek):
  1. Dil bilgisini öğren (genel rules)
  2. Türkçe kelime ve dilbilgisini ekle
  3. Sonuç: Hızlı öğrenme
  
CNN'de, aynı işlem:

Pre-training (ImageNet, 2012):
  - 1.2M resim, 1000 sınıf
  - 14-16 hafta eğitim (GPU'da)
  - Katman 1-40: Edge, color, texture öğren
  
Fine-tuning (Tıbbi veriler, seniniz):
  - 1G resim, 3 sınıf
  - 1-2 saat eğitim (transfer layer'lardan başlayarak)
  - Son katmanlar: Tıbbi özellikler öğren

Transfer Learning Stratejileri:

1. FROZEN BASE + TRAIN TOP (Bizim yaptığımız)
   Avantaj: Hızlı, az veri yeter, calculus az
   Dezavantaj: Adapte yeteneği sınırlı
   
   ```
   ResNet50 (frozen) [23.5M params - don't train]
   ↓
   Custom Dense Layers [1.3M params - train these]
   ```

2. PARTIAL FINE-TUNING
   Avantaj: Daha iyi adapte
   Dezavantaj: Daha fazla eğitim saati
   
   ```
   ResNet50 Layers 1-40 (frozen)
   ResNet50 Layers 41-50 (train - low learning rate)
   Custom Dense (train)
   ```

3. FULL FINE-TUNING
   Avantaj: Maksimum performans
   Dezavantaj: Çok veri lazım, overfitting riski
   
   ```
   ResNet50 (train - çok düşük learning rate)
   Custom Dense (train)
   ```
"""


# ============================================================================
# 3. OVERFITTING vs UNDERFITTING
# ============================================================================

"""
Overfitting: Modelin EĞITIM verisine çok uyum sağlaması

Örnek:
  - Eğitim accuracy: 97%
  - Test accuracy: 65%  ← Felaket!
  - Nedeni: Eğitim verisinin detaylarına (noise) uydu
  
Çözüm:
  ✅ Dropout (rastgele node'ları öldür)
  ✅ L1/L2 Regularization (ağırlıkları cezalandır)
  ✅ Early Stopping (eğitim durdur)
  ✅ Data Augmentation (farklı versiyonlar)

Underfitting: Model veriyi yeterince ögrenemiyor

Örnek:
  - Eğitim accuracy: 60%
  - Test accuracy: 62%
  - Nedeni: Model çok basit
  
Çözüm:
  ✅ Daha büyük model
  ✅ Daha fazla eğitim
  ✅ Daha iyi features

Optimal Balance:
  
  Accuracy
    │     ╱─── Training
    │    ╱
    │   ╱
    │  ╱      Optimal point
    │ ╱       ✓ (burası)
    │╱        
   ─┴──────────────
  0 1 2 3 4 5 Epochs
  
  - Sol: Underfitting
  - Orta: Optimal
  - Sağ: Overfitting
"""


# ============================================================================
# 4. BATCH NORMALIZATION NEDİR?
# ============================================================================

"""
Problem (Internal Covariate Shift):

  İlk katman: pikseller 0-255 ölçeğinde
  5. katman: aktivasyonlar 0-1000 ölçeğinde
  
  Sorun: Large activation → Large gradients → Instable training

Çözüm: Batch Normalization

Formül:
  ŷ = (y - mean(batch)) / sqrt(var(batch) + ε)
  
Örnek:
  Batch: [1000, 500, 2000, 1500]
  Mean: 1250
  
  Normalized: [
    (1000 - 1250) / std ≈ -0.5,
    (500 - 1250) / std ≈ -1.5,
    ...
  ]
  
  Sonuç: [-0.5, -1.5, 0.8, 0.2]  ← Sıfırda merkezli

Avantajları:
  ✅ Training 10x hızlanır
  ✅ Learning rate daha yüksek olabilir
  ✅ Daha stabil gradient flow
  ✅ Hafif regularization etkisi
"""


# ============================================================================
# 5. DROPOUT NEDIR?
# ============================================================================

"""
Problem: Overfitting - Model eğitim verisine çok uydu

Çözüm: Dropout - Eğitim sırasında rastgele node'ları sıfırla

Sütün:
  
  Normal:        Dropout (%50):
  ●――●――●――●  →  ●○○●○○●○
  │ │ │ │       │   │   │
  ●――●――●――●      ●   ●   ●
  
  Tüm bağlantılar kesildi
  → Ağ "co-adaptation" geliştiremez
  → Sağlam özellikler öğrenir

Mekanizm:

  def forward(x):
    if training:
      mask = RandomBernoulli(p=0.7)  # %70 keep
      return x * mask / 0.7           # scale for average
    else:
      return x  # Test sırasında tüm node'lar kullan

Neden scaling?
  - Training: Ortalama y = x * 0.7
  - Test: Ortalama y = x * 1.0
  - Uyum için: Training çıktısını 0.7'ye böl
  
Test sırasında Dropout yok!
  (Dropout sadece eğitim sırasında kullanılır - st.cache_resource'ın bir nedeni)
"""


# ============================================================================
# 6. LOSS FUNCTION (Hata Fonksiyonu)
# ============================================================================

"""
Amaç: "Modelin ne kadar kötü performans gösterdiğini" ölçmek

Categorical Crossentropy (Bizim tercih):

  Kullanım: Multi-class classification (3+ sınıf)
  
  Formül:
    L = -Σ y_i * log(ŷ_i)
    
    Türkçe: "-log(doğru sınıfın olasılığı)"
  
  Örnek:
    Gerçek: [0, 1, 0]  (class 1 doğru)
    Tahmin: [0.1, 0.8, 0.1]
    
    L = -1 * log(0.8)
      = -(-0.223)
      = 0.223  ← Düşük hata (iyi)
    
    Tahmin yanlış olsaydı:
    Tahmin: [0.8, 0.1, 0.1]
    L = -1 * log(0.1) = 2.30  ← Yüksek hata (kötü)

Diğer Loss Functions:

  MAE (Mean Absolute Error):
    - Regression için
    - L = mean(|y - ŷ|)
  
  MSE (Mean Squared Error):
    - Regression için
    - Büyük hatalar cezalandırma (squared)
  
  Binary Crossentropy:
    - Binary classification (2 sınıf)
    - Daha hızlı
"""


# ============================================================================
# 7. OPTIMIZER (Ağırlık Güncellemesi Algoritması)
# ============================================================================

"""
Amaç: Loss fonksiyonunu minimize etmek

Gradient Descent Varyasyonları:

1. VANILLA GRADIENT DESCENT
   w_new = w_old - learning_rate * gradient
   
   Problem: Sabit learning rate → oscillation
   
2. MOMENTUM
   momentum = 0.9 * momentum + gradient
   w_new = w_old - learning_rate * momentum
   
   Avantaj: "İnişin hızını tutar" (physics)
   
3. ADAM (Adaptive Moment Estimation) - Bizim tercih
   
   Combine:
   - Momentum (exponential moving average gradients)
   - RMSprop (adaptive learning rate per parameter)
   
   m = 0.9 * m + 0.1 * gradient      # momentum
   v = 0.999 * v + 0.001 * grad²     # velocity
   w = w - (lr * m) / (sqrt(v) + ε)  # update
   
   Avantaj:
   ✅ Fast convergence
   ✅ Less hyperparameter tuning
   ✅ Adaptive learning rate
   ✅ Sağlam ve stabil

Learning Rate (lr):
  
  Çok düşük (0.00001):
    ✅ Stabil
    ❌ Çok yavaş eğitim
  
  Çok yüksek (0.1):
    ✅ Hızlı
    ❌ Oscillation, divergence
  
  Optimal (0.001 - 0.0001):
    ✅ Dengeli

Bizim setting:
  optimizer = Adam(learning_rate=0.001)
  → Başlangıç değeri, daha sonra tune edilebilir
"""


# ============================================================================
# 8. METRICS (ÖLÇÜMLER)
# ============================================================================

"""
Accuracy (Doğruluk):

  Formül: Accuracy = (True Positives + True Negatives) / Total
  
  Örnek:
    100 test görüntü
    93 doğru sınıflandırıldı
    Accuracy = 93/100 = 93%

Confusion Matrix:

                 Predicted
                 Pos  Neg
  Actual  Pos  │ TP  │ FN  │
          Neg  │ FP  │ TN  │

  TP (True Positive):  Malign dedik, gerçekten malign
  FP (False Positive): Sağlıklı dedik, aslında malign ← Tehlikeli!
  TN (True Negative):  Sağlıklı dedik, gerçekten sağlıklı
  FN (False Negative): Malign dedik, aslında sağlıklı ← Bahtsız!

Tıbbi bağlamda:
  - FP (yanlış alarm) ≈ Gereksiz muayena
  - FN (yanlış negatif) ≈ Kanser atlattı!!!
  
  Tercihen: Higher Sensitivity (FN'yi minimize et)

Hassasiyet & Spesifiklik:

  Sensitivity = TP / (TP + FN)  ← Hastalıları bulma yeteneği
  Specificity = TN / (TN + FP)  ← Sağlıkları bulma yeteneği
  
  Bizim hedef: TP'yi maximize etmek (kanser hastaları bulmak)
"""


# ============================================================================
# 9. EPOCH, BATCH, ITERATION
# ============================================================================

"""
Definitions:

  Epoch: Tüm eğitim verisinin 1 kez geçişi
    - 1000 görüntü = 1 epoch
  
  Batch: İçerisinde ağırlıklar güncelleştirilen veri alt kümesi
    - 32 göturmü = 1 batch
  
  Iteration: 1 forward + 1 backward pass
    - 1000/32 ≈ 31 iteration = 1 epoch

Örnek (1000 görüntü, batch_size=32):

  Epoch 1:
    Iteration 1: Batch 1 (görüntü  1-32)   → ağırlıkları güncelle
    Iteration 2: Batch 2 (görüntü 33-64)   → ağırlıkları güncelle
    ...
    Iteration 31: Batch 31 (görüntü 969-1000) → ağırlıkları güncelle
    ✅ Epoch 1 tamamlandı
  
  Epoch 2: (aynısı tekrarlan)
  ...

Batch size seçimi:

  Küçük (8):
    ✅ Less memory
    ❌ Noisier updates
  
  Büyük (256):
    ✅ Stable updates
    ❌ More memory
  
  Medium (32-64): Optimal
"""


# ============================================================================
# 10. GRADIENT DESCENT GÖRSELLEŞTIRILMIŞ
# ============================================================================

"""
Amaç: Loss landscape'in minimum noktasına ulaşmak

    Loss
     ▲
     │      ╱╲╱╲    ← Local minima
     │     ╱  X  ╲
     │    ╱       ╲
     │   ╱         ╲╲
     │  ╱           ╲╱
     │ ╱             ╲  ← Global minimum
     │╱               ╲ ← Arzu ettiğimiz
     └─────────────────→ Ağırlık parametresini
    
Gradient Descent'ın yol izlemesi:

    Loss
     │
    10├─● (başlangıç)
     │  │ ╲
     │  │  ╲ 
     5├─│──●← (iteration 1)
     │  │    ╲
     │  │     ●← (iteration 2)
     0├─│─────●← (iteration 3, minimum)
     └─┴─────────────

Learning rate etkisi:

  lr = 0.01 (çok düşük):
    ●────●────●────●────●ー → Çok yavaş
  
  lr = 0.1 (çok yüksek):
    ●   ●   ●   ●ーーー → Oscillation
           ╲ ╱
  
  lr = 0.001 (optimal):
    ●─●─●─●─●─●─●─●ー → Stabil ve hızlı
"""


# ============================================================================
# ÖZET
# ============================================================================

"""
Kodda karşılaşacağın 10 Konsept

1. ✅ CNN: Görüntü işleme için katmanlar
2. ✅ Transfer Learning: Önceki bilgiyi yeniden kullan
3. ✅ Overfitting: Modeli kontrol et (Dropout)
4. ✅ Batch Norm: Training'i hızlandır
5. ✅ Dropout: Rastgele node'ları sıfırla
6. ✅ Categorical Crossentropy: Loss function
7. ✅ Adam Optimizer: Ağırlık güncelleme
8. ✅ Accuracy: Ölçme metriği
9. ✅ Epoch/Batch: Eğitim bölümleri
10. ✅ Gradient Descent: Minimum bulma

Sonraki Adımlar:
- model.py'deki build_model() fonksiyonu satır satır oku
- Her bir katmanın yol izlemesini kağıda çiz
- Test et, değiştir, tekrar dene (learning!)
"""
