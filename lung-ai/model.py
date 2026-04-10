"""
LUNG-AI: Öğrenme İçin Detaylı ML Model Modülü
================================================

Bu dosya, akciğer CT görüntüleri üzerinde kanser tespiti yapan derin öğrenme modelini tanımlar.

Transfer Learning Nedir?
- ImageNet (1.2M+ resim) ile önceden eğitilmiş ResNet50 modelini TIBBİ verilerle adapte ediyoruz.
- Sıfırdan eğitmek yerine, zaten öğrenilmiş özellikler (edge, texture vb) kullanıyoruz.
- Hız ve doğruluk açısından çok daha verimli.

Mimari Adımları:
1. Base Model: ResNet50 (ImageNet ağırlıkları)
2. Global Average Pooling: Çıktıyı düzleştir
3. Dense Katmanlar: Tıbbi sınıflandırma için öğrenme
4. Output: 3 sınıf (Sağlıklı, Malign, Benign)
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pathlib import Path


class LungAIModel:
    """
    Akciğer kanser tespiti için Transfer Learning modeli.
    
    Sınıf Açıklaması:
    - Modeli yükler, eğitir ve tahmin yapar
    - Daha organize ve kullanımı kolay bir yapı sağlar
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        """
        Model başlatıcısı.
        
        Parametreler:
        - input_shape: Görüntü boyutu (ResNet50 için standart 224x224)
        - num_classes: Sınıf sayısı (0=Sağlıklı, 1=Malign, 2=Benign)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_names = ["Sağlıklı", "Malign (Kötü Huylu)", "Benign (İyi Huylu)"]
    
    def build_model(self):
        """
        Transfer Learning modeli inşa etme.
        
        Archi Detayları:
        1. Base Model (ResNet50): 50 katmanlı CNN, ImageNet eğitili
        2. include_top=False: Son sınıflandırma katmanını kaldırıyoruz
           (ImageNet'in 1000 sınıfı yerine kendi 3 sınıfımızı koyacağız)
        3. Global Average Pooling: Spatial dimensions'ı flatten etme
        4. Dense katmanlar: Lineer öğrenme (fully connected)
        """
        
        # Adım 1: ImageNet ile eğitilmiş ResNet50'yi yükle
        base_model = ResNet50(
            weights='imagenet',  # Önceden eğitilmiş ağırlıklar
            include_top=False,   # Son sınıflandırma katmanını gitir
            input_shape=self.input_shape
        )
        
        # Adım 2: Base model'in katmanlarını dondurmak (seçmeli)
        # Neden donduralım? ImageNet'ten öğrenilen genel özellikler (edge, color) 
        # tıbbi görüntülerde de geçerli. İlk katmanların ağırlıklarını değiştirmiyoruz.
        # (İleri aşamada Fine-tuning yapacaksak açabiliriz)
        base_model.trainable = False
        
        # Adım 3: Kendi sınıflandırma katmanlarını ekle
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Base model'den geçir
        x = base_model(inputs, training=False)
        
        # Global Average Pooling: (7, 7, 2048) -> (2048,)
        # Spatial information'ı kaybetmeyip ortalama alıyoruz
        x = GlobalAveragePooling2D()(x)
        
        # Dense katman 1: 1024 node, ReLU aktivasyonu
        # ReLU = max(0, x) - Non-lineer ilişkiler öğretir
        x = Dense(1024, activation='relu')(x)
        
        # Dropout: %30 node'u rastgele zaman eğitim sırasında öldürür
        # Neden? Overfitting'i önlemek (model başka veriye genelleyebilir)
        x = Dropout(0.3)(x)
        
        # Dense katman 2: 128 node
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Çıkış katmanı: 3 node (3 sınıf için)
        # Softmax: Olasılıkları 0-1 arasında ve toplamı 1 olacak şekilde dağıtır
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Model oluştur: Input -> Base Model -> Custom Layers -> Output
        self.model = Model(inputs=inputs, outputs=outputs)
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Modeli eğitim için hazırla.
        
        Parametreler:
        - learning_rate: Ağırlık güncellemesi ne kadar agresif olacak
        
        Optimizer (Optimizasyon Algoritması):
        - Adam: Gradient descent'in geliştirilmiş versiyonu
        - Learning rate: Ne kadar büyük adımlarla ağırlıkları değiştireceğiz
        
        Loss Function (Hata Fonksiyonu):
        - Categorical Crossentropy: Multi-class sınıflandırma için standard
        
        Metrics (Ölçümler):
        - Accuracy: Doğruluk oranı (kaç tanesini doğru sınıflandırdı)
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model derlenmiştir (compiled).")
        print(f"   Optimizer: Adam (lr={learning_rate})")
        print(f"   Loss: Categorical Crossentropy")
    
    def summary(self):
        """Modelin yapısını görüntüle (katman sayısı, parametre sayısı vb)"""
        if self.model is None:
            print("❌ Önce modeli build_model() ile oluşturmalısın!")
            return
        
        self.model.summary()
    
    def predict_from_image(self, image_array):
        """
        Bir görüntü üzerinde tahmin yap.
        
        Parametreler:
        - image_array: NumPy array, shape (224, 224, 3), values 0-1 arasında
        
        Çıkış:
        - predicted_class: Tahmin edilen sınıf indeksi (0, 1 veya 2)
        - confidence: Güven puanı (0-100%)
        - all_predictions: Tüm sınıflara ait olasılıklar
        """
        if self.model is None:
            raise ValueError("Model henüz build edilmemiş!")
        
        # Batch dimensionu ekle (model batch input bekliyor)
        # (224, 224, 3) -> (1, 224, 224, 3)
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Tahmin yap
        predictions = self.model.predict(image_batch, verbose=0)
        
        # predictions shape: (1, 3) -> [0.1, 0.7, 0.2]
        # En yüksek olasılık: argmax
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        return {
            'class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'all_predictions': predictions[0],  # [healthy%, malign%, benign%]
            'probabilities': dict(zip(self.class_names, predictions[0]))
        }


def get_image_preprocessing_pipeline():
    """
    Görüntü ön işleme ve data augmentation pipeline'ı.
    
    Why Augmentation?
    - Modeli daha genel tutmak için (rotation, shift, zoom vb)
    - Veri sayısını artırmış gibi görmek
    - Overfit'i azaltmak
    
    Standart preprocessing (ImageNet için):
    - Piksel değerleri 0-255 -> -1 ile +1 arasına normalize
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,           # 0-255 -> 0-1 normalize
        rotation_range=20,        # 0-20 derece rastgele döndür
        width_shift_range=0.2,    # %20 sağa/sola kaydır
        height_shift_range=0.2,   # %20 yukarı/aşağı kaydır
        zoom_range=0.2,           # %20 yakınlaştır/uzaklaştır
        horizontal_flip=True,     # İki tarafa çevir
        fill_mode='nearest'       # Boş alanları doldur
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, validation_datagen


# KISA REFERANS
# ==============
# Model mimarisi:
#   ResNet50 (base) 
#   ↓
#   Global Average Pooling (7x7x2048 -> 2048)
#   ↓
#   Dense(1024, relu) + Dropout(0.3)
#   ↓
#   Dense(128, relu) + Dropout(0.2)
#   ↓
#   Dense(3, softmax) -> [Sağlıklı, Malign, Benign]
#
# Total Parameters: ~23M (ResNet50: ~23.5M)
#   - ResNet base: Frozen (eğitilmemiş)
#   - Custom layers: Train etmek için açık
