"""
LUNG-AI: Modeli Programmatik Olarak Kullanma (Streamlit dışında)
=================================================================

Bu script, Streamlit arayüzü olmadan modeli doğrudan Python'da nasıl 
kullanacağını gösterir. ÖĞRENME AMAÇLI.

Kullanım:
    python sample_test.py
"""

import numpy as np
from PIL import Image
from model import LungAIModel
import os


def example_1_model_loading():
    """
    ÖRNEK 1: Modeli Yükleme ve Derleme
    ===================================
    
    Neler olur?
    1. Transfer Learning modelini oluştur
    2. ResNet50 + custom layers
    3. Eğitim için derle
    """
    print("\n" + "="*60)
    print("ÖRNEK 1: Model Yükleme")
    print("="*60)
    
    # Modeli oluştur
    model = LungAIModel(input_shape=(224, 224, 3), num_classes=3)
    
    # Model mimarisini inşa et
    print("\n📐 Model inşa ediliyor...")
    model.build_model()
    
    # Eğitim için derle
    print("\n🔧 Model derleniyor (compilation)...")
    model.compile_model(learning_rate=0.001)
    
    # Model yapısını göster
    print("\n📊 Model Yapısı:")
    model.summary()
    
    return model


def example_2_synthetic_prediction(model):
    """
    ÖRNEK 2: Sentetik (Yapay) Veri Üzerinde Tahmin
    ================================================
    
    Neden sentetik veri?
    - Hızlı test
    - Gerçek CT verisi olmadığında demo
    - Tahmin mekanizmasını anlamak
    """
    print("\n" + "="*60)
    print("ÖRNEK 2: Sentetik Görüntü Tahmini")
    print("="*60)
    
    print("\n🎲 Rastgele görüntü oluşturuluyor...")
    # Rastgele RGB görüntü (224×224)
    random_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    print(f"   Görüntü shape: {random_image.shape}")
    print(f"   Pixel değerleri: {random_image.min():.3f} - {random_image.max():.3f}")
    
    print("\n🤖 Tahmin yapılıyor...")
    prediction = model.predict_from_image(random_image)
    
    print(f"\n✅ Tahmin Sonuçları:")
    print(f"   Sınıf: {prediction['class_name']}")
    print(f"   Güven: {prediction['confidence']:.2f}%")
    print(f"\n   Tüm Olasılıklar:")
    for class_name, prob in prediction['probabilities'].items():
        bar = "█" * int(prob * 50)
        print(f"      {class_name:20s}: {prob*100:5.1f}% {bar}")
    
    return prediction


def example_3_load_real_image(model, image_path):
    """
    ÖRNEK 3: Gerçek Görüntü Yükleme ve Tahmin
    ===========================================
    
    Eğer bir test görüntüsü varsa, bunu kullan.
    """
    print("\n" + "="*60)
    print("ÖRNEK 3: Gerçek Görüntü Tahmini")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"❌ Görüntü bulunamadı: {image_path}")
        print("   Atlaniyor...")
        return None
    
    print(f"\n📂 Görüntü yükleniyor: {image_path}")
    
    # Görüntü yükle
    image = Image.open(image_path).convert('RGB')
    print(f"   Orijinal boyut: {image.size}")
    
    # Ön işleme (normalize ve resize)
    from app import preprocess_image
    processed = preprocess_image(image)
    print(f"   İşlenmiş boyut: {processed.shape}")
    print(f"   Pixel değerleri: {processed.min():.3f} - {processed.max():.3f}")
    
    # Tahmin
    print("\n🤖 Tahmin yapılıyor...")
    prediction = model.predict_from_image(processed)
    
    print(f"\n✅ Tahmin Sonuçları:")
    print(f"   Sınıf: {prediction['class_name']}")
    print(f"   Güven: {prediction['confidence']:.2f}%")
    
    return prediction


def example_4_batch_prediction(model):
    """
    ÖRNEK 4: Grup (Batch) Tahmin
    ==============================
    
    Pratikte, genellikle beraber birden fazla görüntü tahmin ederiz.
    Bu nasıl yapılır?
    """
    print("\n" + "="*60)
    print("ÖRNEK 4: Batch Tahmin")
    print("="*60)
    
    batch_size = 4
    print(f"\n🔀 {batch_size} rastgele görüntü grup'u oluşturuluyor...")
    
    # Batch oluştur (4 görüntü)
    batch_images = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
    print(f"   Batch shape: {batch_images.shape}")
    
    # Batch tahmin
    print("\n🤖 Batch tahmin yapılıyor...")
    predictions = model.model.predict(batch_images, verbose=0)
    
    print(f"\n✅ Batch Tahmin Sonuçları:")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   (4 görüntü × 3 sınıf olasılığı)\n")
    
    for idx, pred in enumerate(predictions):
        class_idx = np.argmax(pred)
        confidence = pred[class_idx] * 100
        class_name = model.class_names[class_idx]
        print(f"   Görüntü {idx+1}: {class_name:20s} ({confidence:.1f}%)")


def example_5_model_info():
    """
    ÖRNEK 5: Model İstatistikleri
    =============================
    
    Model hakkında teknik bilgiler
    """
    print("\n" + "="*60)
    print("ÖRNEK 5: Model İstatistikleri")
    print("="*60)
    
    model = LungAIModel()
    model.build_model()
    
    # Parametre sayısı
    total_params = model.model.count_params()
    print(f"\n📊 Model Parametreleri:")
    print(f"   Toplam: {total_params:,} parametri")
    print(f"   (~{total_params/1e6:.1f}M)")
    
    # Katmanlar
    print(f"\n📚 Model Katmanları:")
    print(f"   Toplam katman sayısı: {len(model.model.layers)}")
    
    print(f"\n   İlk 5 katman:")
    for i, layer in enumerate(model.model.layers[:5]):
        print(f"      {i+1}. {layer.name:30s} - {layer.__class__.__name__}")
    
    print(f"\n   Son 5 katman:")
    for i, layer in enumerate(model.model.layers[-5:], start=len(model.model.layers)-4):
        print(f"      {i}. {layer.name:30s} - {layer.__class__.__name__}")
    
    # Input/Output shape
    print(f"\n📥/📤 Input/Output:")
    print(f"   Input shape:  {model.model.input_shape}")
    print(f"   Output shape: {model.model.output_shape}")


# ============================================================================
# MAIN - Tüm örnekleri çalıştır
# ============================================================================

if __name__ == "__main__":
    print("""
    
    🫁 LUNG-AI: Öğrenme Örnekleri
    =============================
    
    Bu script, modelin nasıl çalıştığını adım adım gösterir.
    Kod satırlarını okuyarak mekanizmayı anlayabilirsin.
    
    """)
    
    # Örnek 1: Model yükleme
    model = example_1_model_loading()
    
    # Örnek 2: Sentetik tahmin
    example_2_synthetic_prediction(model)
    
    # Örnek 3: Gerçek görüntü (varsa)
    # example_3_load_real_image(model, "path/to/image.jpg")
    
    # Örnek 4: Batch tahmin
    example_4_batch_prediction(model)
    
    # Örnek 5: Model info
    example_5_model_info()
    
    print("\n" + "="*60)
    print("✅ Tüm örnekler tamamlandı!")
    print("="*60)
    print("""
    
    Sonraki Adımlar:
    1. Kodu satır satır oku
    2. model.summary() çıktısını anla
    3. Tahmin sonuçlarını gözlemle
    4. Değerleri değiştirip denetle (learning)
    
    """)
