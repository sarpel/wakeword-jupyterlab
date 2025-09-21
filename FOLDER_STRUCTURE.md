# 📁 Wakeword Projesi Klasör Yapısı

Bu dokümanda, wakeword detection projesinin veri organizasyonu için önerilen klasör yapısı tanımlanmıştır. Tüm öneriler, mevcut kod tabanına uygun olarak belirlenmiştir.

## 📊 Veri Tipleri ve Klasör Önerileri

| # | Veri Tipi | Önerilen Klasör | Kod Bağlantısı | Açıklama |
|---|-----------|-----------------|---------------|------------|
| 1 | **Pozitif Wakeword Sample** | `positive_dataset/` | ✅ Mevcut | Wakeword olarak kullanılacak ses kayıtları. Kod tarafından doğrudan yüklenir. |
| 2 | **Negatif Talking Sample** | `negative_dataset/` | ✅ Mevcut | Wakeword olmayan konuşma örnekleri. Kod tarafından negatif veri olarak yüklenir. |
| 3 | **Background Noise Sample** | `background_noise/` | ✅ Mevcut | Arka plan gürültüleri. Eğitim sırasında seslere karıştırılır. |
| 4 | **Hard Negative Samples** | `negative_dataset/hard_negative_wakewords/` | ⚠️ Alt klasör | Wakeword'e benzeyen ama wakeword olmayan kelimeler. Kodda özel işlenme yok, ancak ayrı klasörde organize edilmesi mantıklı. |
| 5 | **Negative Feature NPY Dosyası** | `features/train/negative/` | ✅ Mevcut | Pre-computed mel-spectrogram özellikleri. Enhanced training için kullanılır. |
| 6 | **Validation NPY Dosyası** | `features/validation/` | ✅ Mevcut | Validation set için pre-computed özellikler. |
| 7 | **MIT RIRS** | `datasets/mit_rirs/` | ✅ Mevcut | Room Impulse Response verileri. Akustik augmentation için kullanılır. |

## 🔍 Özel Notlar

### Background Noise Önem Derecelendirmesi
- **Ev sesleriniz** (`my_home_noises/` alt klasörü): Gerçek kullanım ortamını temsil ettiği için değerli
- **Diğer background'lar**: Genel dataset'lerden (audioset, fma, vb.)
- **Kod davranışı**: Tüm background dosyaları eşit ağırlıkta kullanılır, ancak ayrı klasörde tutmak analitik avantaj sağlar

### Hard Negative'ler
- Mevcut kodda özel işlenme yok, ancak `negative_dataset/` içinde ayrı klasörde organize edilmesi önerilir
- Eğitim kalitesini artırmak için ayrı takip faydalıdır

### Feature Dosyaları
- Kod `features/` klasörünü kullanır, `feature_datasets/` değil
- Eğer mevcut `feature_datasets/` içindeki dosyalar kullanılacaksa `features/` altına taşınmalı

## 📂 Tam Klasör Ağacı Örneği

```
wakeword-jupyterlab/
├── positive_dataset/           # 1. Pozitif wakeword sample'lar
│   ├── multi_voice_wakeword_outputs_1010/
│   ├── my_positive_samples_5283/
│   └── positive_wakewords_1100/
├── negative_dataset/           # 2. Negatif talking sample'lar
│   ├── hard_negative_wakewords/    # 4. Hard negative samples
│   ├── common_voice_en_16k_subset_24h/
│   └── speech-commands/
├── background_noise/           # 3. Background noise sample'lar
│   ├── my_home_noises/             # Ev sesleri (değerli)
│   ├── audioset_16k/
│   └── fma_16k/
├── features/                   # 5-6. Feature NPY dosyaları
│   ├── train/
│   │   └── negative/               # Negative feature NPY
│   └── validation/                 # Validation NPY
├── datasets/
│   └── mit_rirs/               # 7. MIT RIRS
└── config/
    └── feature_config.yaml
```

## ⚙️ Kod Uyumluluğu

- **✅ Mevcut**: Kod tarafından tanınan ve kullanılan klasörler
- **⚠️ Alt klasör**: Kodda doğrudan referans edilmeyen ancak mevcut yapıda olan klasörler
- Tüm öneriler mevcut kod tabanıyla uyumludur ve ek değişiklik gerektirmez

## 🔄 Güncelleme Notu

Bu yapı hem Gradio web UI hem Jupyter notebook için aynıdır. Kod, tüm klasörleri otomatik olarak tarar ve kullanır.</content>
<filePath>c:\Users\Sarpel\Desktop\wakeword-jupyterlab\FOLDER_STRUCTURE.md
