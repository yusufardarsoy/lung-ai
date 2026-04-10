"""
LUNG-AI: Streamlit Web Uygulaması
==================================

Bu dosya, kullanıcıların akciğer CT görüntülerini yükleyip analiz ettiği web arayüzü.

Streamlit Nedir?
- Python'dan direkt web uygulaması yapabileceğin bir kütüphane
- GUI oluşturmak için karmaşık HTML/CSS/JS gerekmez
- Ideal: ML prototipler, veri dashboard'ları, MVP'ler

Nasıl çalışır?
- Scriptinin çalıştığını düşün (üstten aşağıya)
- Kullanıcı input verince, script yeniden çalışır (reactive)
- Streamlit caching ile yavaş işlemleri hızlandırırız
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
from model import LungAIModel
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. STREAMLIT SAYFA KONFIGÜRASYONU (En başta olmalı)
# ============================================================================

st.set_page_config(
    page_title="LUNG-AI: Akciğer Analiz Sistemi",
    page_icon="🫁",
    layout="wide",  # Geniş layout (sidebar + content)
    initial_sidebar_state="expanded"
)

# Custom CSS (Koyu tema, profesyonel görünüm)
st.markdown("""
    <style>
    /* Ana container arka planı */
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: #58a6ff;
    }
    
    /* Butonlar ve bileşenler */
    .stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
    }
    
    .stButton > button:hover {
        background-color: #2ea043;
    }
    
    /* Info box'lar */
    .stInfo {
        background-color: #161b22;
        border-left: 3px solid #58a6ff;
        border-radius: 6px;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. CACHE İLE MODEL YÜKLEME (Hızlama)
# ============================================================================

@st.cache_resource
def load_model():
    """
    Modeli cache'leme (saklama) ile yükle.
    
    @st.cache_resource nedir?
    - Function'ı ilk çağırışta çalıştırır
    - Sonucu RAM'da saklar
    - Tekrar çağırışta cache'den döner (çok hızlı)
    - Sayfa yenilense bile modeli tekrar yüklemiyor
    
    Neden önemli?
    - ResNet50 modeli ~100MB ve yüklemesi ~2-3 saniye
    - Her sayfa yenilenmesinde yeniden yüklesek, UX kötü olurdu
    """
    st.info("🔄 Model yükleniyor... (İlk sefer ~3-5 saniye)")
    
    try:
        lung_model = LungAIModel(input_shape=(224, 224, 3), num_classes=3)
        lung_model.build_model()
        lung_model.compile_model()
        
        st.success("✅ Model başarıyla yüklendi!")
        return lung_model
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {str(e)}")
        return None


# ============================================================================
# 3. GÖRÜNTÜ İŞLEME VE NORAMLİZE ETME
# ============================================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Ham görüntüyü model için hazırla.
    
    Adımlar:
    1. RGB'ye dönüştür (bazı PNG'ler RGBA olabilir)
    2. 224x224'e yeniden boyutlandır (ResNet50 input size)
    3. NumPy array'e dönüştür
    4. 0-1 arasına normalize (piksel: 0-255 -> 0-1)
    5. Önişleme (ImageNet normalizasyonu - opsiyonel, tahmin için değil)
    
    Çıkış:
    - Shape: (224, 224, 3)
    - Values: 0.0 - 1.0
    """
    # Adım 1: RGB'ye çevir
    image_rgb = image.convert('RGB')
    
    # Adım 2: Boyutlandır
    image_resized = image_rgb.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Adım 3: NumPy array'e dönüştür
    image_array = np.array(image_resized, dtype=np.float32)
    
    # Adım 4: Normalize (0-255 -> 0-1)
    image_normalized = image_array / 255.0
    
    return image_normalized


def create_confidence_chart(probs_dict):
    """
    Sınıf olasılıklarından bir görselleştirme çart oluştur.
    
    Input example:
    - {'Sağlıklı': 0.15, 'Malign': 0.75, 'Benign': 0.10}
    
    Çıkış:
    - Matplotlib bar chart (Streamlit'te gösterilir)
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 4))
    classes = list(probs_dict.keys())
    probs = list(probs_dict.values())
    colors = ['#238636', '#da3633', '#ffd700']  # Yeşil, Kırmızı, Sarı
    
    bars = ax.bar(classes, probs, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax.set_ylabel('Olasılık (%)', color='#c9d1d9', fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#161b22')
    
    # Yüzde etiketleri ekle
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob*100:.1f}%',
                ha='center', va='bottom', color='#c9d1d9', fontweight='bold')
    
    ax.tick_params(colors='#c9d1d9')
    plt.xticks(rotation=15, ha='right')
    
    return fig


# ============================================================================
# 4. STREAMLIT ANA ARAYÜZ
# ============================================================================

# BAŞLIK ve AÇIKLAMA
st.title("🫁 LUNG-AI: BT Tarama ve Kanser Teşhis Sistemi")
st.markdown("""
    **LUNG-AI**, akciğer bilgisayarlı tomografi (BT) görüntülerinden kanser riskini tespit etmeye yardımcı olan 
    bir yapay zeka sistemidir. Bu sistem **MVP** (Minimum Viable Product) aşamasında olup, 
    **kesin tıbbi teşhis için radyolog onayı gereklidir**.
""")

st.divider()

# SIDEBAR: Bilgi paneli
with st.sidebar:
    st.header("ℹ️ Sistem Hakkında")
    st.markdown("""
    ### Transfer Learning Mimarisi
    - **Base Model:** ResNet50 (ImageNet pretrained)
    - **Katmanlar:** 50 + Custom Dense layers
    - **Input:** 224×224 RGB Görüntü
    - **Output:** [Sağlıklı, Malign, Benign]
    
    ### Veri Seti
    - LIDC-IDRI protokolü uyumlu (simülasyon)
    - 1018+ Hasta CT Scanı
    - Radyolog annotasyonları
    
    ### Uyarı ⚠️
    - **Bu sistem tıbbi tavsiye sunmaz**
    - **Resmi tanı için doktor görünüşü zorunludur**
    - Eğitim amaçlı prototip
    """)
    
    st.divider()
    st.caption("🚀 Powered by TensorFlow + Streamlit")


# ANA İÇERİK - ÜST BÖLÜM: Dosya Yükleme
st.subheader("📤 Adım 1: BT Görüntüsünü Yükleyin")
uploaded_file = st.file_uploader(
    "CT Scan görüntüsü (JPG, PNG) seçin",
    type=["jpg", "png", "jpeg"],
    help="224×224 pixel veya daha büyük görüntüler önerilir"
)

if uploaded_file is not None:
    # Model'i yükle
    model = load_model()
    
    if model is None:
        st.error("Model yüklenemedi. Lütfen tekrar deneyin.")
    else:
        st.divider()
        st.subheader("🔍 Analiz Sonuçları")
        
        # İki kolon: Sol = Görüntü, Sağ = Sonuçlar
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown("**Yüklenen Görüntü:**")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="CT Kesiti")
            
            st.info(f"""
            **Görüntü Bilgisi:**
            - Boyut: {image.size[0]} × {image.size[1]}
            - Format: {image.format}
            - Mode: {image.mode}
            """)
        
        with col2:
            st.markdown("**Sistem Tahminleri:**")
            
            # Analiz spinner
            with st.spinner('🤖 Algoritma dokuları inceliyor... (0-3 saniye)'):
                try:
                    # Görüntü ön işleme
                    processed_image = preprocess_image(image)
                    
                    # Tahmin yap
                    prediction_result = model.predict_from_image(processed_image)
                    
                    # Sonuçları görüntüle
                    predicted_class = prediction_result['class']
                    class_name = prediction_result['class_name']
                    confidence = prediction_result['confidence']
                    probabilities = prediction_result['probabilities']
                    
                    # Risk seviyesine göre renk kodu
                    if predicted_class == 0:  # Sağlıklı
                        st.success(f"✅ **{class_name}**")
                        risk_color = "green"
                        recommendation = "Şu an için acil müdahale gerekmemektedir."
                    elif predicted_class == 1:  # Malign
                        st.error(f"🚨 **{class_name}**")
                        risk_color = "red"
                        recommendation = "⚠️ **ACIL:** Onkoloji konsültasyonu önerilir!"
                    else:  # Benign
                        st.warning(f"⚠️ **{class_name}**")
                        risk_color = "orange"
                        recommendation = "Periyodik takip önerilir (6-12 ay)."
                    
                    # Güven puanı göster
                    col_conf1, col_conf2 = st.columns([1, 2])
                    with col_conf1:
                        st.metric(
                            label="Güven Puanı",
                            value=f"{confidence:.1f}%",
                            delta=None
                        )
                    
                    with col_conf2:
                        st.progress(confidence / 100, text=f"{confidence/100:.0%}")
                    
                    # Detaylı sınıf olasılıkları
                    st.markdown("**Sınıflandırma Dağılımı:**")
                    fig = create_confidence_chart(probabilities)
                    st.pyplot(fig)
                    
                    # Tıbbi Tavsiye
                    st.markdown("---")
                    st.markdown(f"### 🏥 Tıbbi Tavsiye")
                    st.markdown(recommendation)
                    
                except Exception as e:
                    st.error(f"❌ Tahmin yapılırken hata oluştu: {str(e)}")
        
        # FOOTER
        st.divider()
        st.caption("""
        ⚠️ **Yasal Uyarı:** Bu sistem yapay zeka tarafından oluşturulmuş bir yardımcı tavsiyedir. 
        Kesin teşhis ve tedavi kararları için mutlaka uzman radyolog ve hekime danışınız.
        """)

else:
    # Dosya yüklenmediyse bilgilendirici mesaj
    st.info("""
    👈 Başlamak için sol taraftaki yükleyici ile bir BT görüntüsü seçin.
    
    ### Nasıl Kullanılır?
    1. **Görüntü Seç:** `.jpg` veya `.png` formatında bir CT kesiti yükleyin
    2. **Analiz Bekle:** Model tarafından otomatik olarak incelenecek (~2-3 saniye)
    3. **Sonuç Gör:** Sınıflandırma, güven puanı ve tıbbi tavsiye
    
    ### Örnek Durumlar
    - ✅ **Sağlıklı:** Normal parankim, lezyon yok
    - 🚨 **Malign:** Kanser riski yüksek (acil takip)
    - ⚠️ **Benign:** İyi huylu lezyonlar (periyodik takip)
    """)
