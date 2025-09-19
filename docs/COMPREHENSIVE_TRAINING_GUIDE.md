# 🎯 WAKEWORD DETECTION COMPLETE TRAINING GUIDE
## EFSANEVI DETAYLI KAPSAMLI BELGELEME

---

## 📋 İÇİNDEKİLER
1. [Sistem Genel Bakış](#sistem-genel-bakış)
2. [Veri Seti Hazırlama](#veri-seti-hazırlama)
3. [Model Mimarisi](#model-mimarisi)
4. [Eğitim Süreci](#eğitim-süreci)
5. [Konfigürasyon Parametreleri](#konfigürasyon-parametreleri)
6. [Veri Artırma Teknikleri](#veri-artırma-teknikleri)
7. [Performans Optimizasyonu](#performans-optimizasyonu)
8. [Sorun Giderme](#sorun-giderme)
9. [En İyi Uygulamalar](#en-iyi-uygulamalar)

---

## 🏗️ SİSTEM GENEL BAKIŞ

### Mimari Yapısı
```
Audio Input → Feature Extraction → CNN → LSTM → Classification → Output
    ↓              ↓               ↓     ↓          ↓           ↓
  .wav/.mp3    Mel-Spectrogram   2D Conv  Sequence   Softmax   Wake/No-Wake
  16kHz        80x31 matrix     128 ch   256 hidden   2 classes   Probability
```

### Temel Bileşenler
1. **AudioProcessor**: Ses dosyası yükleme ve ön işleme
2. **EnhancedWakewordDataset**: Veri seti yönetimi ve artırma
3. **WakewordModel**: CNN+LSTM sinir ağı mimarisi
4. **WakewordTrainer**: Eğitim süreci yönetimi
5. **Gradio Arayüzü**: Görsel arayüz ve canlı izleme

---

## 📊 VERİ SETİ HAZIRLAMA

### 1.1. Pozitif Veriler (Wakeword Kayıtları)

#### Miktar ve Kalite Gereksinimleri
- **Minimum Miktar**: 100-500 temiz wakeword kaydı
- **İdeal Miktar**: 1000+ wakeword kaydı
- **Kalite Kriterleri**:
  - SNR (Sinyal/Gürültü Oranı): ≥ 20dB
  - Clipping olmamalı (ses zirve noktaları -3dB'den fazla olmamalı)
  - Arka plan gürültüsü minimum düzeyde
  - Net ve anlaşabilir telaffuz

#### Çeşitlilik Gereksinimleri
```
🎤 Farklı Mikrofon Çeşitliliği:
• Smartphone mikrofonları
• USB mikrofonlar
• Bluetooth mikrofonlar
• Laptop dahili mikrofonlar
• Profesyonel kayıt cihazları

👥 Farklı Konuşanlar:
• Erkek/Kadın/Çocuk sesleri
• Farklı yaş grupları
• Farklı aksanlar
• Farklı konuşma hızları
• Farklı ses tonları (yüksek/alçak)

🌍 Farklı Ortamlar:
• Sessiz oda (SNR > 30dB)
• Ofis ortamı (SNR 20-25dB)
• Dış mekan (SNR 15-20dB)
• Araba içi (SNR 10-15dB)
• Kafe/restoran (SNR 5-10dB)

📐 Teknik Çeşitlilik:
• Farklı sample rate'ler (16kHz, 44.1kHz)
• Farklı bit derinlikleri (16-bit, 24-bit)
• Farklı formatlar (WAV, FLAC, MP3)
• Farklı süreler (0.5s - 3.0s)
```

#### Kayıt İpuçları
1. **Mesafe**: Mikrofona 15-30cm mesafeden konuşun
2. **Hacim**: Normal konuşma ses seviyesinde
3. **Arka Plan**: Mümkün olduğunca sessiz ortam
4. **Tekrar**: Her wakeword'u 3-5 kez kaydedin
5. **Doğallık**: Günlük konuşma tarzında, yapay olmayan

### 1.2. Negatif Veriler

#### A. Hard Negative Samples (Fonetik Benzer Kelimeler)
```
🎯 Phonetically Benzer Kelimeler:
• "hey" → "hey computer", "hey google", "hey siri"
• "ok" → "okay", "ok google", "ok computer"
• "day" → "hey", "they", "say"
• "night" → "light", "right", "bright"
• "computer" → "commuter", "compute", "commute"

📊 Miktar Oranları:
• Her wakeword için 4-5 hard negative
• Toplamda wakeword sayısının 4.5 katı
• Örnek: 100 wakeword → 450 hard negative
```

#### B. Random Negative Samples (Genel Konuşma Sesleri)
```
🗣️ Konuşma Çeşitliliği:
• Günlük konuşmalar
• Telefon görüşmeleri
• Radyo/TV yayınları
• Podcast'ler
• Toplantı kayıtları
• Sokak sesleri
• Alışveriş merkezi sesleri

📊 Miktar Oranları:
• Her wakeword için 8-9 random negative
• Toplamda wakeword sayısının 8.75 katı
• Örnek: 100 wakeword → 875 random negative
```

#### C. Background Noise Samples (Arka Plan Gürültüleri)
```
🔊 Gürültü Çeşitliliği:
• Beyaz gürültü (white noise)
• Pembe gürültü (pink noise)
• Kahverengi gürültü (brown noise)
• Fan/klimalar sesi
• Trafik gürültüsü
• İnsan kalabalığı sesi
• Müzik (farklı türler)
• Yağmur/rüzgar sesi
• Elektronik cihaz sesleri

📊 Miktar Oranları:
• Minimum 66 saat arka plan gürültüsü
• Her wakeword için 10 background sample
• Toplamda wakeword sayısının 10 katı
• Örnek: 100 wakeword → 1000 background sample
```

### 1.3. Veri Dengeleme ve Oranları

#### İdeal Veri Dağılımı
```
📈 Optimum Veri Seti Oranları:
1 wakeword : 4.5 hard_negative : 8.75 random_negative : 10 background

📊 Örnek Dağılım (100 wakeword için):
• Wakeword: 100 samples (%4.2)
• Hard Negative: 450 samples (%18.8)
• Random Negative: 875 samples (%36.5)
• Background: 1000 samples (%41.7)
• TOPLAM: 2425 samples (%100)

⚖️ Dengesiz Verinin Etkileri:
• Az wakeword → Model wakeword'u öğrenemez
• Çok wakeword → Model her şeyi wakeword sanar
• Az negative → False positive artar
• Çok negative → False negative artar
```

#### Train/Validation/Test Split
```
🎯 Optimum Split Oranları:
• Training: %70 (%80 validation/test hariç)
• Validation: %20 (%25 training hariç)
• Test: %10 (%12.5 training+validation hariç)

📊 2425 sample için:
• Training: 1698 samples
• Validation: 485 samples
• Test: 242 samples

⚠️ Önemli Notlar:
• Her kategoriden orantılı dağılım olmalı
• Aynı kişi/kayıt farklı setlere dağılmamalı
• Random seed ile tutarlılık sağlanmalı
```

---

## 🧠 MODEL MİMARİSİ

### 2.1. CNN+LSTM Mimarisi Detaylı Analiz

#### Katman Yapısı
```python
# Input Shape: (batch, 1, 80, 31) - 80 mel bands, 31 time frames
Input → [Conv2D(1→32)] → [ReLU] → [Conv2D(32→64)] → [ReLU] →
[Conv2D(64→128)] → [ReLU] → [AdaptiveAvgPool2d(1,1)] →
[Flatten] → [LSTM(128→256×2)] → [Dropout(0.6)] → [Linear(256→2)]
```

#### CNN Katmanları Detaylı
```
🔲 Convolutional Layer 1:
• Input: (1, 80, 31) - 1 channel, 80 mel bands, 31 time frames
• Filters: 32 filters of size 3×3
• Padding: 1 pixel (same padding)
• Output: (32, 80, 31)
• Activation: ReLU
• Parameters: (3×3×1 + 1) × 32 = 320 parameters
• Purpose: Temel frekans özelliklerini çıkarma

🔲 Convolutional Layer 2:
• Input: (32, 80, 31)
• Filters: 64 filters of size 3×3
• Padding: 1 pixel
• Output: (64, 80, 31)
• Activation: ReLU
• Parameters: (3×3×32 + 1) × 64 = 18,496 parameters
• Purpose: Daha karmaşık akustik desenleri tanıma

🔲 Convolutional Layer 3:
• Input: (64, 80, 31)
• Filters: 128 filters of size 3×3
• Padding: 1 pixel
• Output: (128, 80, 31)
• Activation: ReLU
• Parameters: (3×3×64 + 1) × 128 = 73,856 parameters
• Purpose: Yüksek seviye akustik özellikleri çıkarma

🔲 Adaptive Average Pooling:
• Input: (128, 80, 31)
• Output: (128, 1, 1)
• Purpose: Sabit boyutlu vektör çıkarma
• Avantaj: Farklı ses uzunlukları için çalışır
```

#### LSTM Katmanları Detaylı
```
🔄 LSTM Layer 1:
• Input: (batch, 1, 128) - sequence length=1, features=128
• Hidden Size: 256
• Num Layers: 2
• Dropout: 0.6 (between layers)
• Batch First: True
• Output: (batch, 1, 256)
• Parameters: 4 × (128×256 + 256×256 + 256) × 2 = 788,992 parameters
• Purpose: Zaman serisi içindeki desenleri öğrenme

🎯 LSTM Çalışma Prensibi:
• Forget Gate: Hangi bilgileri unutacağına karar verir
• Input Gate: Yeni bilgileri hafızaya alır
• Output Gate: Çıktıyı üretir
• Cell State: Uzun süreli hafıza
• Hidden State: Kısa süreli hafıza

💡 LSTM Avantajları:
• Uzun vadeli bağımlılıkları öğrenebilir
• Ses dizilerindeki temporal desenleri yakalar
• Değişken uzunluktaki girdileri işleyebilir
```

#### Final Katmanlar
```
🎯 Dropout Layer:
• Dropout Rate: 0.6 (%60 nöron rastgele devre dışı)
• Purpose: Overfitting'i önleme
• Training sırasında aktif, inference'da pasif

🎯 Linear Layer:
• Input: 256 features (LSTM output)
• Output: 2 classes (wakeword, negative)
• Activation: Softmax (implicit in CrossEntropyLoss)
• Parameters: (256 + 1) × 2 = 514 parameters
• Purpose: Sınıflandırma kararı

📊 Toplam Parametre Sayısı:
• CNN Katmanları: 320 + 18,496 + 73,856 = 92,672
• LSTM Katmanları: 788,992
• Final Katman: 514
• TOPLAM: 882,178 parametre (~882K)
```

### 2.2. Model Parametre Optimizasyonu

#### Hidden Size Seçimi
```
🔢 Hidden Size Seçenekleri:
• 128: Küçük veri setleri için, hızlı eğitim
• 256: Dengeli performans, optimum değer
• 512: Büyük veri setleri için, daha iyi accuracy
• 1024: Çok büyük veri setleri, yavaş eğitim

⚖️ Trade-off Analizi:
• Küçük Hidden Size:
  - Avantaj: Hızlı eğitim, az memory kullanımı
  - Dezavantaj: Düşük model kapasitesi, underfitting riski

• Büyük Hidden Size:
  - Avantaj: Yüksek model kapasitesi, better accuracy
  - Dezavantaj: Yavaş eğitim, yüksek memory kullanımı, overfitting riski
```

#### Dropout Oranı
```
🎯 Dropout Seçenekleri:
• 0.2-0.3: Büyük veri setleri için
• 0.4-0.6: Orta büyüklükte veri setleri için (optimum)
• 0.7-0.8: Küçük veri setleri için

⚠️ Dropout Yan Etkileri:
• Çok düşük dropout → Overfitting
• Çok yüksek dropout → Underfitting
• Training: Rastgele nöronları devre dışı bırakır
• Inference: Tüm nöronlar aktif, çıkış ölçeklenir
```

---

## 🎯 EĞİTİM SÜRECİ

### 3.1. Training Pipeline Detaylı Anlatım

#### 1. Veri Yükleme ve Ön İşleme
```python
# Adım 1: Ses Dosyalarını Yükle
audio_files = glob.glob("positive_dataset/*.wav")
audio_data = [librosa.load(f, sr=16000)[0] for f in audio_files]

# Adım 2: Normalizasyon
normalized_audio = [audio / np.max(np.abs(audio)) for audio in audio_data]

# Adım 3: Boyut Standardizasyonu
target_length = int(16000 * 1.7)  # 1.7 saniye
padded_audio = [pad_or_truncate(audio, target_length) for audio in normalized_audio]

# Adım 4: Mel-Spectrogram Dönüşümü
mel_specs = [librosa.feature.melspectrogram(
    y=audio, sr=16000, n_mels=80, n_fft=2048,
    hop_length=512, fmin=0, fmax=8000
) for audio in padded_audio]

# Adım 5: Log Scale Dönüşümü
log_mel_specs = [librosa.power_to_db(mel, ref=np.max) for mel in mel_specs]

# Adım 6: Tensor Dönüşümü
tensors = [torch.FloatTensor(mel).unsqueeze(0) for mel in log_mel_specs]
```

#### 2. Batch Oluşturma ve Shuffling
```python
# Adım 1: Dataset Oluşturma
dataset = EnhancedWakewordDataset(
    wakeword_files=wakeword_files,
    hard_negative_files=hard_neg_files,
    random_negative_files=random_neg_files,
    background_files=background_files,
    processor=processor,
    augment=True
)

# Adım 2: DataLoader Oluşturma
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# Adım 3: Batch Örneği
for batch_idx, (data, target) in enumerate(dataloader):
    # data.shape: (32, 1, 80, 31)
    # target.shape: (32,)

    # GPU'ya taşı
    data = data.to(device)
    target = target.to(device)
```

#### 3. Forward Pass ve Loss Hesaplama
```python
# Adım 1: Model Forward Pass
with torch.set_grad_enabled(True):  # Training modu
    output = model(data)  # (32, 2)

    # Adım 2: Loss Hesaplama
    loss = criterion(output, target)

    # Adım 3: Accuracy Hesaplama
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target).sum().item()
    accuracy = 100 * correct / target.size(0)
```

#### 4. Backward Pass ve Optimizasyon
```python
# Adım 1: Gradient Sıfırlama
optimizer.zero_grad()

# Adım 2: Backward Pass
loss.backward()

# Adım 3: Gradient Clipping (Patlama önleme)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Adım 4: Parametre Güncelleme
optimizer.step()

# Adım 5: Learning Rate Scheduling
scheduler.step(val_accuracy)
```

### 3.2. Learning Rate Scheduling

#### ReduceLROnPlateau Mekanizması
```python
# Konfigürasyon
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',           # validation accuracy'yi izle
    factor=0.5,          # learning rate'i yarıya indir
    patience=5,          # 5 epoch boyunca improvement yoksa
    verbose=True,
    min_lr=1e-7          # minimum learning rate
)

# Çalışma Prensibi:
# 1. Her epoch sonunda validation accuracy'yi kontrol et
# 2. Eğer accuracy improvement göstermiyorsa:
#    - patience sayacını artır
#    - patience >= 5 ise learning rate'i factor ile çarp
# 3. Improvement gösterirse sayacı sıfırla
# 4. Learning rate min_lr'nin altına inemez
```

#### Learning Rate Seçimi
```
🎯 Farklı Learning Rate'lerin Etkileri:
• 0.001: Çok hızlı, unstable eğitim, gradient explosion riski
• 0.0005: Hızlı öğrenme, riskli
• 0.0001: Dengeli, stabil eğitim (OPTIMUM)
• 0.00005: Yavaş ama stabil
• 0.00001: Çok yavaş, uzun eğitim süresi

⚡ Learning Rate Adaptasyonu:
• Başlangıç: 0.0001
• Patlama durumu: 0.5 × current_lr
• Minimum: 1e-7
• Maksimum: 0.001
```

### 3.3. Early Stopping Mekanizması

#### Early Stopping Algoritması
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_accuracy):
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
```

#### Early Stopping Parametreleri
```
⏱️ Patience Seçimi:
• 5-10 epoch: Hızlı development için
• 10-15 epoch: Normal eğitim için (OPTIMUM)
• 15-20 epoch: Karmaşık problemler için

📊 Min Delta Seçimi:
• 0.001: Sıkı durdurma (hassas)
• 0.005: Normal durdurma
• 0.01: Esnek durdurma

💡 Early Stopping Avantajları:
• Overfitting'i önler
• Eğitim süresini kısaltır
• En iyi modeli kaydeder
• Computational kaynakları korur
```

---

## ⚙️ KONFİGÜRASYON PARAMETRELERİ

### 4.1. AudioConfig Detaylı Açıklama

```python
class AudioConfig:
    SAMPLE_RATE = 16000      # Örnekleme frekansı (Hz)
    DURATION = 1.7          # Ses süresi (saniye)
    N_MELS = 80             # Mel bant sayısı
    N_FFT = 2048            # FFT boyutu
    HOP_LENGTH = 512        # Adım boyutu
    WIN_LENGTH = 2048       # Pencere boyutu
    FMIN = 0                # Minimum frekans (Hz)
    FMAX = 8000             # Maksimum frekans (Hz)
```

#### Parametre Detaylı Analizi
```
🎵 SAMPLE_RATE (16kHz):
• Neden 16kHz? İnsan sesi için yeterli frekans aralığı
• İnsan konuşması: 85Hz - 255Hz (fundamental)
• Harmonikler: 8kHz'e kadar önemli
• Nyquist teoremi: 16kHz → 8kHz maksimum frekans
• Daha yüksek sample rate: Daha fazla veri, daha fazla işlem
• Daha düşük sample rate: Bilgi kaybı, aliasing

⏱️ DURATION (1.7 saniye):
• Neden 1.7s? Wakeword'lar genellikle 0.5-2s arası
• Çok kısa: Bilgi eksikliği
• Çok uzun: Gereksiz bilgi, computational maliyet
• 1.7s: Dengeli değer, wakeword'u tam kapsar

🎼 N_MELS (80 Mel Bantları):
• Mel skalası: İnsan işitmesi ile uyumlu
• 40 bant: Minimum kabul edilebilir
• 80 bant: Dengeli çözünürlük (OPTIMUM)
• 128 bant: Yüksek çözünürlük, daha fazla computation
• Mel bantları: Düşük frekanslarda yoğun, yüksek frekanslarda seyrek

📊 N_FFT (2048 Nokta FFT):
• Frekans çözünürlüğü belirler
• 2048: 16kHz / 2048 = 7.8Hz frekans çözünürlüğü
• Daha büyük FFT: Daha iyi frekans çözünürlüğü, daha az zaman çözünürlüğü
• Daha küçük FFT: Daha iyi zaman çözünürlüğü, daha az frekans çözünürlüğü

👣 HOP_LENGTH (512 Örnek Adımı):
• Zaman çözünürlüğünü belirler
• 512 / 16000 = 32ms zaman adımı
• 1.7s / 0.032s = 53 zaman adımı
• Daha küçük hop: Daha fazla zaman adımı, daha fazla computation
• Daha büyük hop: Daha az zaman adımı, bilgi kaybı

🪟 WIN_LENGTH (2048 Pencere Boyutu):
• Frekans çözünürlüğünü etkiler
• Genellikle N_FFT ile aynı olur
• Hamming/Hanning window kullanılır
• Window fonksiyonu: Spektral sızıntıyı önler
```

### 4.2. ModelConfig Detaylı Açıklama

```python
class ModelConfig:
    HIDDEN_SIZE = 256      # LSTM gizli katman boyutu
    NUM_LAYERS = 2         # LSTM katman sayısı
    DROPOUT = 0.6          # Dropout oranı
    NUM_CLASSES = 2        # Sınıf sayısı
```

#### Parametre Optimizasyonu
```
🧠 HIDDEN_SIZE (256):
• LSTM kapasitesini belirler
• 128: Küçük model (~400K parametre)
• 256: Dengeli model (~882K parametre) - OPTIMUM
• 512: Büyük model (~1.7M parametre)
• 1024: Çok büyük model (~3.4M parametre)

📚 NUM_LAYERS (2):
• LSTM katman derinliği
• 1 katman: Hızlı, simple ilişkiler
• 2 katman: Dengeli, karmaşık ilişkiler - OPTIMUM
• 3-4 katman: Derin, çok karmaşık ilişkiler
• Fazla katman: Vanishing gradient, eğitim zorluğu

🎭 DROPOUT (0.6):
• Regularization oranı
• 0.2-0.3: Büyük veri setleri
• 0.4-0.6: Orta veri setleri - OPTIMUM
• 0.7-0.8: Küçük veri setleri
• Training: Rastgele nöronları devre dışı bırakır
• Inference: Tüm nöronlar aktif, çıkış ölçeklenir
```

### 4.3. TrainingConfig Detaylı Açıklama

```python
class TrainingConfig:
    BATCH_SIZE = 32        # Batch boyutu
    LEARNING_RATE = 0.0001 # Öğrenme oranı
    EPOCHS = 100           # Epoch sayısı
    VALIDATION_SPLIT = 0.2 # Validation oranı
    TEST_SPLIT = 0.1       # Test oranı
```

#### Training Parametre Analizi
```
📦 BATCH_SIZE (32):
• Memory ve gradient dengesi
• 8-16: Küçük batch, daha iyi generalization
• 32-64: Dengeli batch - OPTIMUM
• 128-256: Büyük batch, daha hızlı eğitim
• Küçük batch: Daha fazla noise, daha iyi generalization
• Büyük batch: Daha stabil gradient, daha hızlı eğitim

🎯 LEARNING_RATE (0.0001):
• Öğrenme hızı
• 0.001: Çok hızlı, unstable
• 0.0001: Dengeli - OPTIMUM
• 0.00001: Çok yavaş
• Adaptif: ReduceLROnPlateau ile optimizasyon

⏰ EPOCHS (100):
• Maksimum epoch sayısı
• Early stopping ile otomatik durma
• Gerçek epoch sayısı genellikle 20-50 arası
• 100: Yeterli büyük sayı, early stopping ile kontrol

✂️ VALIDATION_SPLIT (0.2):
• Validation için ayrılan oran
• 0.1-0.15: Küçük validation seti
• 0.2-0.25: Dengeli - OPTIMUM
• 0.3: Büyük validation seti
• Fazla validation: Eğitim verisi azalır
• Az validation: Overfitting riski
```

### 4.4. AugmentationConfig Detaylı Açıklama

```python
class AugmentationConfig:
    AUGMENTATION_PROB = 0.85    # Artırma olasılığı
    NOISE_FACTOR = 0.15         # Gürültü faktörü
    TIME_SHIFT_MAX = 0.3        # Zaman kaydırma maksimumu
    PITCH_SHIFT_MAX = 1.5       # Perde değiştirme maksimumu
    SPEED_CHANGE_MIN = 0.9      # Hız değiştirme minimumu
    SPEED_CHANGE_MAX = 1.1      # Hız değiştirme maksimumu
```

#### Augmentation Parametre Optimizasyonu
```
🎲 AUGMENTATION_PROB (0.85):
• Her sample için artırma uygulanma olasılığı
• 0.5: Orta seviye artırma
• 0.7-0.9: Yoğun artırma - OPTIMUM
• 1.0: Her sample artırılır
• Çok yüksek: Orijinal veri kaybolabilir
• Çok düşük: Yetersiz çeşitlilik

🔊 NOISE_FACTOR (0.15):
• Gürültü seviyesi
• 0.05: Hafif gürültü
• 0.1-0.2: Orta seviye gürültü - OPTIMUM
• 0.3+: Yoğun gürültü
• SNR ile ilişkili: -20dB ila -6dB arası

⏱️ TIME_SHIFT_MAX (0.3s):
• Zaman kaydırma miktarı
• ±0.1s: Hafif kaydırma
• ±0.2-0.4s: Dengeli kaydırma - OPTIMUM
• ±0.5s+: Yoğun kaydırma
• WAKETIME için: ±0.3s uygun

🎵 PITCH_SHIFT_MAX (±1.5 semiton):
• Perde değiştirme miktarı
• ±0.5: Hafif perde değişimi
• ±1.0-2.0: Dengeli - OPTIMUM
• ±3.0+: Yoğun perde değişimi
• 1 semiton = %12 frekans değişimi

⚡ SPEED_CHANGE (0.9x - 1.1x):
• Hız değiştirme oranı
• 0.95-1.05: Hafif hız değişimi
• 0.9-1.1: Dengeli - OPTIMUM
• 0.8-1.2: Yoğun hız değişimi
• Hem perdeyi hem süreyi değiştirir
```

---

## 🔄 VERİ ARTIRMA TEKNİKLERİ

### 5.1. Zaman Tabanlı Artırmalar

#### Time Shifting (Zaman Kaydırma)
```python
def time_shift(audio, max_shift_seconds=0.3, sample_rate=16000):
    """
    Sesi zaman içinde kaydırır
    """
    max_shift_samples = int(max_shift_seconds * sample_rate)
    shift_amount = random.randint(-max_shift_samples, max_shift_samples)

    # np.roll ile dairesel kaydırma
    shifted_audio = np.roll(audio, shift_amount)

    return shifted_audio
```

**Detaylı Açıklama:**
```
⏰ Time Shifting Mekanizması:
• Amaç: Wakeword'un farklı zamanlarda olması senaryosu
• Çalışma: np.roll ile dairesel kaydırma
• Aralık: ±0.3 saniye (±4800 sample)
• Etki: Modelin zaman bağımsız öğrenmesi
• Limit: Ses sınırını aşmamalı

🎯 Uygulama Detayları:
• Pozitif kaydırma: Wakeword'u ileri alma
• Negatif kaydırma: Wakeword'u geri alma
• Sınırlar: Ses uzunluğu içinde kalmalı
• Random: Her seferinde farklı miktar
```

#### Speed Changing (Hız Değiştirme)
```python
def speed_change(audio, speed_factor):
    """
    Ses hızını değiştirir
    """
    # librosa time_stretch kullanımı
    stretched = librosa.effects.time_stretch(audio, rate=speed_factor)

    # Orijinal uzunluğa getirme
    if len(stretched) < len(audio):
        stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
    else:
        stretched = stretched[:len(audio)]

    return stretched
```

**Detaylı Açıklama:**
```
⚡ Speed Changing Etkileri:
• Amaç: Farklı konuşma hızları senaryosu
• Çalışma: librosa time_stretch ile PSOLA algoritması
• Aralık: 0.9x - 1.1x (%10 hız değişimi)
• Etkiler: Hem perdeyi hem süreyi değiştirir
• Kalite: PSOLA ile doğal ses koruma

🎵 Akustik Etkiler:
• 0.9x: Daha yavaş, daha bass perde
• 1.1x: Daha hızlı, daha tiz perde
• Doğallık: PSOLA sayesinde doğal kalır
• Uyum: Time stretching ile uyumlu
```

### 5.2. Frekans Tabanlı Artırmalar

#### Pitch Shifting (Perde Değiştirme)
```python
def pitch_shift(audio, n_steps, sample_rate=16000):
    """
    Ses perdesini değiştirir
    """
    # librosa pitch_shift kullanımı
    shifted = librosa.effects.pitch_shift(
        y=audio,
        sr=sample_rate,
        n_steps=n_steps
    )

    return shifted
```

**Detaylı Açıklama:**
```
🎼 Pitch Shifting Detayları:
• Amaç: Farklı ses tonları senaryosu
• Çalışma: librosa pitch_shift ile PSOLA
• Birim: Semiton (yarım perde)
• Aralık: ±1.5 semiton
• Etki: Sadece perdeyi değiştirir, süreyi korur

🎤 Semiton Kavramı:
• 1 semiton = %12 frekans değişimi
• 12 semiton = 1 oktav
• İnsan kulağı: 1-2 semiton farkı algılar
• Doğallık: ±3 semitone'a kadar doğal

👥 Farklı Sesler için:
• Erkek sesleri: Düşük perde değişimi
• Kadın sesleri: Yüksek perde değişimi
• Çocuk sesleri: Orta perde değişimi
• Uyum: Doğal konuşma aralığında
```

### 5.3. Gürültü Tabanlı Artırmalar

#### Background Noise Mixing (Arka Plan Gürültüsü Karıştırma)
```python
def mix_with_background(audio, background_audio, target_snr_db):
    """
    Sesi arka plan gürültüsü ile karıştırır
    """
    # SNR hesaplama
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(background_audio ** 2)

    # Hedef SNR için ölçeklendirme
    if noise_power > 0:
        target_noise_power = signal_power / (10 ** (target_snr_db / 10))
        scale_factor = np.sqrt(target_noise_power / noise_power)

        # Gürültüyü ölçeklendir ve karıştır
        scaled_noise = background_audio * scale_factor
        mixed = audio + scaled_noise

        # Normalize etme
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val * 0.95

        return mixed

    return audio
```

**Detaylı Açıklama:**
```
🔊 SNR (Signal-to-Noise Ratio) Kavramı:
• Tanım: Sinyal gücünün gürültü gücüne oranı
• Birim: dB (desibel)
• Formül: SNR_dB = 10 × log10(P_signal / P_noise)
• Yüksek SNR: Temiz sinyal
• Düşük SNR: Gürültülü sinyal

📊 Farklı SNR Seviyeleri:
• 20dB+: Çok temiz (stüdyo kalitesi)
• 10-20dB: Temiz (ofis ortamı)
• 5-10dB: Orta (dış mekan)
• 0-5dB: Gürültülü (kafe, trafik)
• 0dB-: Çok gürültülü (fabrika, konser)

🎯 SNR Seçimi:
• Training: 0-20dB arası geniş range
• Validation: 5-15dB arası dar range
• Real-world: Genellikle 5-15dB arası
```

#### Additive Noise (Toplamsal Gürültü)
```python
def add_noise(audio, noise_factor=0.15):
    """
    Sese rastgele gürültü ekler
    """
    # Beyaz gürültü oluşturma
    noise = np.random.normal(0, noise_factor, len(audio))

    # Gürültüyü ekleme
    noisy_audio = audio + noise

    return noisy_audio
```

**Detaylı Açıklama:**
```
🎲 Gürültü Türleri:
• Beyaz gürültü: Tüm frekanslarda eşit güç
• Pembe gürültü: Düşük frekanslarda daha güçlü
• Kahverengi gürültü: Daha da düşük frekanslarda güçlü
• Mavi gürültü: Yüksek frekanslarda daha güçlü

🔢 Noise Factor Seçimi:
• 0.05: Hafif gürültü
• 0.1-0.2: Orta seviye gürültü - OPTIMUM
• 0.3+: Yoğun gürültü
• Sinyal gücüne göre normalize edilmeli
```

### 5.4. Bileşik Artırma Stratejileri

#### Artırma Pipeline'ı
```python
def comprehensive_augmentation(audio, config):
    """
    Kapsamlı artırma pipeline'ı
    """
    augmented = audio.copy()

    # Zaman tabanlı artırmalar
    if random.random() < config.AUGMENTATION_PROB:
        augmented = time_shift(augmented, config.TIME_SHIFT_MAX)

    if random.random() < config.AUGMENTATION_PROB:
        speed_factor = random.uniform(config.SPEED_CHANGE_MIN, config.SPEED_CHANGE_MAX)
        augmented = speed_change(augmented, speed_factor)

    # Frekans tabanlı artırmalar
    if random.random() < config.AUGMENTATION_PROB:
        n_steps = random.uniform(-config.PITCH_SHIFT_MAX, config.PITCH_SHIFT_MAX)
        augmented = pitch_shift(augmented, n_steps)

    # Gürültü tabanlı artırmalar
    if random.random() < config.AUGMENTATION_PROB:
        augmented = add_noise(augmented, config.NOISE_FACTOR)

    # Background mixing
    if random.random() < config.BACKGROUND_MIX_PROB:
        bg_audio = random.choice(background_cache)
        target_snr = random.uniform(config.SNR_MIN, config.SNR_MAX)
        augmented = mix_with_background(augmented, bg_audio, target_snr)

    return augmented
```

**Detaylı Açıklama:**
```
🎯 Artırma Stratejisi:
• Aşamalı: Farklı artırma türleri sırayla
• Olasılıksal: Her artırma için ayrı olasılık
• Bağımsız: Birbirini etkilemez
• Normalize: Son aşamada normalize etme

📊 Artırma Dağılımı:
• %85: En az bir artırma uygulanır
• %50: Birden fazla artırma uygulanır
• %15: Hiç artırma uygulanmaz (orijinal korunur)

⚖️ Denge Noktaları:
• Çok fazla artırma: Orijinali bozar
• Çok az artırma: Çeşitlilik yetersiz
• %85: Dengeli oran - OPTIMUM
```

---

## 📈 PERFORMANS OPTİMİZASYONU

### 6.1. GPU Optimizasyonu

#### GPU Memory Yönetimi
```python
# GPU memory monitoring
def monitor_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"GPU Memory: {allocated:.1f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")
        print(f"Reserved: {reserved:.1f}GB")

        return allocated, reserved, total
```

**GPU Optimizasyon Stratejileri:**
```
💾 Memory Optimizasyonu:
• Pin Memory: CPU-GPU transfer hızlandırma
• Gradient Accumulation: Büyük batch'ler için
• Mixed Precision: Hafıza ve hız artışı
• Gradient Checkpointing: Memory tasarrufu

⚡ Hız Optimizasyonu:
• CUDA Kernels: Optimize edilmiş GPU fonksiyonları
• Asynchronous Transfer: Paralel data transferi
• Tensor Cores: Modern GPU'lar için
• Memory Layout: Optimize edilmiş veri düzeni

🔧 Memory Ayarları:
• Batch Size: Memory sınırlarına göre
• Num Workers: Paralel data loading
• Pin Memory: True (GPU için)
• Non-blocking: True (asynchronous)
```

#### Mixed Precision Training
```python
# Mixed precision için scaler
scaler = torch.cuda.amp.GradScaler()

def train_epoch_mixed_precision(model, dataloader, optimizer, criterion):
    model.train()

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)

        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**Mixed Precision Avantajları:**
```
🚀 Mixed Precision Faydaları:
• Memory: %50 memory tasarrufu
• Hız: %2-3x hız artışı
• Kalite: Minimum accuracy kaybı
• Uyum: Modern GPU'lar ile çalışır

⚠️ Dikkat Edilmesi Gerekenler:
• Gradient scaling gerekli
• Numerical stability sorunları
• Tüm işlemler desteklemeyebilir
• Model architecture etkisi
```

### 6.2. Data Loading Optimizasyonu

#### Paralel Data Loading
```python
# Optimize edilmiş DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,        # Paralel worker sayısı
    pin_memory=True,     # GPU için pin memory
    persistent_workers=True,  # Worker'ları canlı tut
    prefetch_factor=2,   # Prefetch faktörü
    non_blocking=True    # Non-blocking transfer
)
```

**Optimizasyon Parametreleri:**
```
👥 Num Workers Seçimi:
• 0: Ana process'te loading (yavaş)
• 2-4: Optimum değer
• 8+: Çok fazla worker (overhead)
• CPU core sayısına göre ayarla

📌 Pin Memory:
• True: GPU memory'de önceden alan ayırır
• False: Normal memory allocation
• Speed: %10-20 hız artışı sağlar
• Memory: Biraz daha fazla memory kullanır

⏡ Persistent Workers:
• True: Worker'ları epoch'lar arasında canlı tutar
• False: Her epoch'da yeniden başlatır
• Speed: Epoch başına hızlandırma
• Memory: Sürekli memory kullanımı
```

### 6.3. Model Optimizasyonu

#### Model Pruning
```python
def prune_model(model, pruning_amount=0.2):
    """
    Model pruning ile optimizasyon
    """
    parameters_to_prune = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    # Global pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_amount
    )

    # Pruning'i kalıcı hale getirme
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    return model
```

**Pruning Stratejileri:**
```
✂️ Pruning Türleri:
• L1 Unstructured: En küçük ağırlıkları kaldırır
• Structured: Tüm filtreleri/katmanları kaldırır
• Global: Model genelinde pruning
• Local: Katman bazında pruning

📊 Pruning Miktarı:
• %10-20: Hafif pruning
• %30-50: Orta seviye pruning
• %60+: Ağır pruning
• Accuracy loss ile dengeli olmalı

⚖️ Pruning Trade-off'ları:
• Avantaj: Daha küçük model, daha hızlı inference
• Dezavantaj: Potansiyel accuracy kaybı
• Uygulama: Deployment için ideal
```

---

## 🚨 SORUN GİDERME

### 7.1. Yaygın Eğitim Sorunları

#### 7.1.1. Overfitting

**Belirtiler:**
```
📈 Overfitting Göstergeleri:
• Train accuracy sürekli artıyor (%95+)
• Validation accuracy düşüyor veya sabit
• Train loss sürekli azalıyor
• Validation loss artıyor
• Confusion matrix'de imbalance
```

**Çözümler:**
```python
# 1. Dropout artırma
model.dropout.p = 0.7  # 0.6'dan 0.7'a

# 2. Data augmentation güçlendirme
config.AUGMENTATION_PROB = 0.9  # 0.85'ten 0.9'a
config.NOISE_FACTOR = 0.2       # 0.15'ten 0.2'a

# 3. Early stopping uygulama
early_stopping = EarlyStopping(patience=8, min_delta=0.001)

# 4. Learning rate azaltma
optimizer = optim.Adam(model.parameters(), lr=0.00005)  # 0.0001'den 0.00005'e

# 5. Regularization ekleme
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
```

**Prevention (Önleme) Stratejileri:**
```
🛡️ Overfitting Önleme:
• Cross-validation kullanma
• Daha fazla veri toplama
• Augmentation çeşitliliği artırma
• Model karmaşıklığını azaltma
• Regularization teknikleri
• Early stopping implementasyonu
```

#### 7.1.2. Underfitting

**Belirtiler:**
```
📉 Underfitting Göstergeleri:
• Train accuracy düşük (<%70)
• Validation accuracy de düşük
• Her iki loss da yüksek
• Model öğrenemiyor
• Confusion matrix'de random dağılım
```

**Çözümler:**
```python
# 1. Model kapasitesini artırma
model_config.HIDDEN_SIZE = 512    # 256'dan 512'e
model_config.NUM_LAYERS = 3       # 2'den 3'e

# 2. Learning rate artırma
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 0.0001'den 0.001'e

# 3. Dropout azaltma
model.dropout.p = 0.3  # 0.6'dan 0.3'a

# 4. Epoch sayısını artırma
training_config.EPOCHS = 200  # 100'den 200'e

# 5. Augmentation azaltma
config.AUGMENTATION_PROB = 0.5  # 0.85'ten 0.5'e
```

#### 7.1.3. Gradient Explosion

**Belirtiler:**
```
💥 Gradient Explosion Belirtileri:
• Loss aniden çok büyük değerler alıyor
• Training NaN/Inf değerleri üretiyor
• Model ağırlıkları çok büyüyor
• Accuracy düştü (model çöktü)
• GPU memory hataları
```

**Çözümler:**
```python
# 1. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Learning rate azaltma
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 3. Batch normalization ekleme
model.conv1 = nn.Conv2d(1, 32, 3, padding=1)
model.bn1 = nn.BatchNorm2d(32)

# 4. Weight initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

model.apply(init_weights)
```

#### 7.1.4. Vanishing Gradient

**Belirtiler:**
```
🌀 Vanishing Gradient Belirtileri:
• Early katmanlar öğrenmiyor
• Loss çok yavaş azalıyor
• Accuracy sabit kalıyor
• Gradientler çok küçük
• Deep network'lerde sık görülür
```

**Çözümler:**
```python
# 1. Batch normalization ekleme
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU()
)

# 2. Residual connections
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

# 3. LSTM layer sayısını azaltma
model_config.NUM_LAYERS = 1  # 2'den 1'e
```

### 7.2. Memory ve Performance Sorunları

#### 7.2.1. GPU Memory Error

**Belirtiler:**
```
💾 GPU Memory Error Belirtileri:
• CUDA out of memory hatası
• Training başarısız oluyor
• Batch size küçültmek sorunu çözüyor
• GPU memory usage %90+
```

**Çözümler:**
```python
# 1. Batch size küçültme
training_config.BATCH_SIZE = 16  # 32'den 16'ya

# 2. Gradient accumulation
accumulation_steps = 2
optimizer.zero_grad()
for i, (data, target) in enumerate(dataloader):
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Mixed precision training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(data)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()

# 4. Data loading optimization
dataloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)
```

#### 7.2.2. Yavaş Training

**Belirtiler:**
```
🐌 Yavaş Training Belirtileri:
• Epoch başına çok zaman alıyor
• GPU kullanımı düşük
• CPU bottleneck var
• Data loading yavaş
```

**Çözümler:**
```python
# 1. Paralel data loading
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Worker sayısını artır
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

# 2. GPU memory optimization
torch.backends.cudnn.benchmark = True  # CNN için
torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32

# 3. Model optimization ile başlatma
model = model.to(device).train()  # Training modunda

# 4. Async data loading
for batch_idx, (data, target) in enumerate(dataloader):
    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
```

### 7.3. Veri Kalitesi Sorunları

#### 7.3.1. Düşük Accuracy

**Belirtiler:**
```
📊 Düşük Accuracy Belirtileri:
• Test accuracy < %70
• Validation accuracy < %75
• Model rastgele tahmin yapıyor
• Confusion matrix dengesiz
```

**Çözümler:**
```python
# 1. Veri kalitesini kontrol etme
def check_data_quality():
    # Ses dosyalarını kontrol et
    for file in wakeword_files[:10]:
        audio, sr = librosa.load(file, sr=16000)
        if len(audio) == 0:
            print(f"Boş dosya: {file}")
        if np.max(np.abs(audio)) > 1.0:
            print(f"Clipping: {file}")
        if len(audio) < sr * 0.5:
            print(f"Çok kısa: {file}")

# 2. Veri setini dengeleme
def balance_dataset():
    # Her kategoriden eşit sayıda sample
    min_samples = min(len(wakeword_files), len(negative_files))
    wakeword_balanced = random.sample(wakeword_files, min_samples)
    negative_balanced = random.sample(negative_files, min_samples)
    return wakeword_balanced + negative_balanced

# 3. Augmentation güçlendirme
config.AUGMENTATION_PROB = 0.95
config.NOISE_FACTOR = 0.25
config.TIME_SHIFT_MAX = 0.5
```

#### 7.3.2. Imbalanced Dataset

**Belirtiler:**
```
⚖️ Imbalanced Dataset Belirtileri:
• Bir sınıf diğerinden çok daha fazla
• Model çoğunluk sınıfını tahmin ediyor
• Precision/Recall dengesiz
• F1-score düşük
```

**Çözümler:**
```python
# 1. Class weights
class_counts = [len(negative_files), len(wakeword_files)]
class_weights = torch.tensor([
    len(wakeword_files) / len(negative_files),
    len(negative_files) / len(wakeword_files)
], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# 2. Oversampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# 3. Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# 4. SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

## 🏆 EN İYİ UYGULAMALAR

### 8.1. Veri Toplama ve Hazırlama

#### 8.1.1. Kaliteli Veri Toplama İpuçları
```
🎤 Kayıt Ortamı:
• Sessiz bir oda (SNR > 30dB)
• Yankısız ortam (halı, perde vb.)
• Sabit mesafe (15-30cm)
• Sabit ses seviyesi
• Arka plan gürültüsü minimum

👥 Konuşan Çeşitliliği:
• Farklı yaş grupları
• Farklı cinsiyetler
• Farklı aksanlar
• Farklı konuşma stilleri
• Farklı ses tonları

📞 Mikrofon Çeşitliliği:
• Smartphone mikrofonları
• USB mikrofonlar
• Bluetooth kulaklıklar
• Laptop dahili mikrofonlar
• Profesyonel kayıt ekipmanları

🔊 Teknik Kalite:
• Sample rate: 16kHz veya üzeri
• Bit depth: 16-bit veya üzeri
• Format: WAV (kayıpsız)
• Süre: 1-2 saniye
• Clipping olmamalı
```

#### 8.1.2. Veri Ön İşleme Best Practices
```python
def optimal_preprocessing_pipeline(audio_path):
    """
    Optimum veri ön işleme pipeline'ı
    """
    # 1. Yükleme
    audio, sr = librosa.load(audio_path, sr=16000)

    # 2. Kalite kontrolü
    if len(audio) == 0:
        return None
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    # 3. Gürültü azaltma (isteğe bağlı)
    if noisy_environment:
        audio = reduce_noise(audio, sr)

    # 4. Normalizasyon
    audio = audio / np.max(np.abs(audio))

    # 5. Boyut standardizasyonu
    target_length = int(sr * 1.7)
    if len(audio) > target_length:
        start_idx = random.randint(0, len(audio) - target_length)
        audio = audio[start_idx:start_idx + target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)))

    # 6. Feature extraction
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=80, n_fft=2048,
        hop_length=512, fmin=0, fmax=8000
    )

    # 7. Log scale dönüşümü
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel
```

### 8.2. Model Eğitimi Best Practices

#### 8.2.1. Hyperparameter Tuning
```python
def hyperparameter_tuning():
    """
    Sistematik hyperparameter tuning
    """
    param_grid = {
        'learning_rate': [0.001, 0.0005, 0.0001, 0.00005],
        'batch_size': [16, 32, 64],
        'hidden_size': [128, 256, 512],
        'dropout': [0.3, 0.5, 0.7],
        'augmentation_prob': [0.5, 0.7, 0.9]
    }

    best_params = None
    best_accuracy = 0

    for lr in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            for hidden_size in param_grid['hidden_size']:
                for dropout in param_grid['dropout']:
                    for aug_prob in param_grid['augmentation_prob']:
                        # Model oluştur ve eğit
                        model = create_model(hidden_size, dropout)
                        accuracy = train_and_evaluate(model, lr, batch_size, aug_prob)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'hidden_size': hidden_size,
                                'dropout': dropout,
                                'augmentation_prob': aug_prob
                            }

    return best_params, best_accuracy
```

#### 8.2.2. Cross-Validation Implementasyonu
```python
from sklearn.model_selection import StratifiedKFold

def cross_validation_training():
    """
    K-fold cross-validation
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold + 1}/{kfold.n_splits}")

        # Veriyi böl
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Dataset ve DataLoader oluştur
        train_dataset = WakewordDataset(X_train, y_train)
        val_dataset = WakewordDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Model eğitimi
        model = create_model()
        trainer = WakewordTrainer(model, device)
        accuracy = trainer.train(train_loader, val_loader, epochs=50)

        fold_accuracies.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}%")

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    print(f"Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")

    return mean_accuracy, std_accuracy
```

### 8.3. Deployment Best Practices

#### 8.3.1. Model Export ve Optimizasyon
```python
def optimize_model_for_deployment(model):
    """
    Deployment için model optimizasyonu
    """
    # 1. Evaluation moduna al
    model.eval()

    # 2. Gradient calculation'i kapat
    for param in model.parameters():
        param.requires_grad = False

    # 3. TorchScript'e dönüştür
    scripted_model = torch.jit.script(model)

    # 4. ONNX'e dönüştür (isteğe bağlı)
    dummy_input = torch.randn(1, 1, 80, 31)
    torch.onnx.export(model, dummy_input, "wakeword_model.onnx")

    # 5. Model bilgilerini kaydet
    model_info = {
        'input_shape': (1, 1, 80, 31),
        'output_shape': (1, 2),
        'sample_rate': 16000,
        'duration': 1.7,
        'n_mels': 80,
        'threshold': 0.8
    }

    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    return scripted_model
```

#### 8.3.2. Production Monitoring
```python
def monitor_model_performance():
    """
    Production'da model performansını izleme
    """
    metrics = {
        'accuracy': [],
        'latency': [],
        'memory_usage': [],
        'false_positives': [],
        'false_negatives': []
    }

    def log_prediction(prediction, ground_truth, latency):
        metrics['accuracy'].append(prediction == ground_truth)
        metrics['latency'].append(latency)

        if prediction == 1 and ground_truth == 0:
            metrics['false_positives'].append(1)
        if prediction == 0 and ground_truth == 1:
            metrics['false_negatives'].append(1)

        # Memory monitoring
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            metrics['memory_usage'].append(memory_used)

    def generate_report():
        report = {
            'average_accuracy': np.mean(metrics['accuracy']),
            'average_latency': np.mean(metrics['latency']),
            'average_memory': np.mean(metrics['memory_usage']),
            'false_positive_rate': np.mean(metrics['false_positives']),
            'false_negative_rate': np.mean(metrics['false_negatives'])
        }
        return report

    return log_prediction, generate_report
```

---

## 🎯 SONUÇ VE ÖNERİLER

### 9.1. Başarılı Bir Wakeword Modeli İçin Gereksinimler

#### Minimum Gereksinimler:
```
📊 Veri:
• 100+ temiz wakeword kaydı
• 450+ phonetically benzer negative
• 875+ random negative ses
• 66+ saat arka plan gürültüsü

⚙️ Model:
• CNN+LSTM mimarisi
• 882K+ parametre
• %85+ validation accuracy
• %80+ test accuracy

🔧 Konfigürasyon:
• Learning rate: 0.0001
• Batch size: 32
• Dropout: 0.6
• Augmentation: %85
```

#### İdeal Gereksinimler:
```
📊 Veri:
• 1000+ temiz wakeword kaydı
• 4500+ phonetically benzer negative
• 8750+ random negative ses
• 100+ saat arka plan gürültüsü

⚙️ Model:
• CNN+LSTM mimarisi
• 882K+ parametre
• %90+ validation accuracy
• %85+ test accuracy

🔧 Konfigürasyon:
• Learning rate: 0.0001 (with scheduling)
• Batch size: 32 (with gradient accumulation)
• Dropout: 0.6
• Augmentation: %85
• Cross-validation: 5-fold
```

### 9.2. Önerilen Workflow

#### Phase 1: Veri Hazırlama (1-2 Hafta)
```
1. Wakeword kayıtlarını topla (100+ sample)
2. Negative sample'leri topla (phonetically benzer)
3. Arka plan gürültülerini topla (66+ saat)
4. Veri kalitesini kontrol et
5. Veri setini dene
```

#### Phase 2: Model Geliştirme (2-3 Hafta)
```
1. Model mimarisini kur
2. Hyperparameter tuning yap
3. Cross-validation ile test et
4. Overfitting kontrolü
5. Model optimizasyonu
```

#### Phase 3: Training ve Evaluation (1-2 Hafta)
```
1. Full training yap
2. Model performansını değerlendir
3. Test seti ile validate et
4. Deployment için optimize et
5. Monitoring kurulumu yap
```

### 9.3. Sürekli İyileştirme Stratejileri

#### A/B Testing:
```
🧪 Test Edilecek Özellikler:
• Farklı augmentation teknikleri
• Farklı model mimarileri
• Farklı hyperparameter'lar
• Farklı veri setleri
```

#### Monitoring ve Alerting:
```
📈 İzlenecek Metrikler:
• Accuracy drift
• Latency değişimleri
• Memory usage
• False positive/negative oranları
• Kullanıcı feedback'i
```

---

## 📚 EK BİLGİLER

### A.1. Teknik Terimler Sözlüğü

#### Ses İşleme Terimleri:
```
🎵 Sample Rate: Sinyalin saniyedeki örnek sayısı
🎼 Mel Scale: İnsan işitmesine göre ölçeklenmiş frekans skalası
📊 Spectrogram: Zaman-frekans domeninde ses gösterimi
🔊 SNR (Signal-to-Noise Ratio): Sinyal/gürültü oranı
🎤 FFT (Fast Fourier Transform): Frekans domenine dönüşüm
```

#### Derin Öğrenme Terimleri:
```
🧠 CNN (Convolutional Neural Network): Evrişimli sinir ağı
🔄 LSTM (Long Short-Term Memory): Uzun kısa süreli hafıza
🎭 Dropout: Regularization tekniği
📈 Backpropagation: Geri yayılım algoritması
⚡ Gradient Descent: Gradient iniş optimizasyonu
```

### A.2. Yararlı Kütüphaneler ve Araçlar

#### Ses İşleme:
```
🎵 librosa: Ses analizi ve feature extraction
🔊 soundfile: Ses dosyası okuma/yazma
🎼 pydub: Ses manipülasyonu
📊 matplotlib: Ses görselleştirme
```

#### Derin Öğrenme:
```
🧠 PyTorch: Derin öğrenme framework'ü
⚡ CUDA: GPU hesaplama
📈 scikit-learn: Machine learning araçları
🎯 TensorFlow: Alternatif framework
```

#### Development:
```
🐍 Python: Ana programlama dili
🔧 NumPy: Sayısal hesaplama
📊 Pandas: Veri işleme
🎨 Gradio: Web arayüzü
```

### A.3. Referanslar ve Kaynaklar

#### Akademik Makaleler:
```
1. "Wake Word Detection using Deep Neural Networks" - Google Research
2. "End-to-End Wake Word Detection" - Amazon Alexa
3. "Small-Footprint Keyword Spotting" - Apple Siri
4. "Convolutional Neural Networks for Audio Classification" - Stanford
5. "LSTM Networks for Speech Recognition" - University of Toronto
```

#### Open Source Projeler:
```
1. Porcupine - Picovoice (Open source wake word engine)
2. Snowboy - KITT.AI (Wake word detection toolkit)
3. Mycroft - Open source voice assistant
4. Coqui TTS - Text-to-speech synthesis
5. Mozilla DeepSpeech - Speech-to-text engine
```

#### Online Kaynaklar:
```
1. PyTorch Audio Documentation
2. LibROSA Documentation
3. TensorFlow Audio Tutorials
4. Kaggle Audio Competition Kernels
5. GitHub Speech Recognition Projects
```

---

## 🎯 SON

Bu kapsamlı rehber, wakeword detection için gerekli tüm teknik detayları içerir. Başarılı bir model için:

1. **Kaliteli veri toplama** - Temiz, çeşitli, dengeli veri setleri
2. **Doğru model seçimi** - CNN+LSTM mimarisi
3. **Optimum konfigürasyon** - Hyperparameter tuning
4. **Sistem yaklaşımı** - Cross-validation, monitoring, A/B testing

Unutmayın, wakeword detection bir sanat bilimidir. Veri kalitesi en önemli faktördür. İyi eğlenceler! 🎉

---

*Bu rehber sürekli güncellenmektedir. Yeni özellikler ve optimizasyonlar için düzenli kontrol edin.*