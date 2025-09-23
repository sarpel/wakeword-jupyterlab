#!/usr/bin/env python3
"""
Test script for the improved Load Datasets output format
"""

# Mock data for testing
wakeword_files = 150
negative_files = 1200
background_files = 800

hard_negative_train = 300
hard_negative_val = 75
random_negative_train = 600
random_negative_val = 150

train_dataset_size = 1250
val_dataset_size = 275

# Simulate the improved output format
data_info = f"""
# VERI YUKLEME BASARILI!

## VERI ISTATISTIKLERI

### Dosya Sayilari:
| Klasor | Toplam Dosya | Egitim Icin Ayrilan |
|--------|-------------|-------------------|
| **Wakeword (Pozitif)** | `{wakeword_files:,}` | `{hard_negative_train:,}` train + `{hard_negative_val:,}` val + `0` test |
| **Negative (Genel)** | `{negative_files:,}` | `{random_negative_train:,}` train + `{random_negative_val:,}` val + `0` test |
| **Background Gurultu** | `{background_files:,}` | `{background_files:,}` (hepsi kullanilir) |

### Negative Dagilimi:
| Tur | Egitim | Validation |
|----|--------|------------|
| **Hard Negative** (Fonetik Benzer) | `{hard_negative_train:,}` | `{hard_negative_val:,}` |
| **Random Negative** (Genel) | `{random_negative_train:,}` | `{random_negative_val:,}` |

### Final Dataset Boyutlari:
- **Egitim Seti:** `{train_dataset_size:,} ornek` ({hard_negative_train} wakeword + {hard_negative_train} hard_neg + {random_negative_train} random_neg + {background_files} bg)
- **Validation Seti:** `{val_dataset_size:,} ornek` ({hard_negative_val} wakeword + {hard_negative_val} hard_neg + {random_negative_val} random_neg + 50 bg)

### Teknik Detaylar:
- **Model Parametreleri:** `1,234,567`
- **Batch Size:** `32`
- **Cihaz:** `cuda`
- **Background Mix:** 30%
- **SNR Range:** -5dB - 15dB

### Kalite Degerlendirmesi:
"""

if wakeword_files >= 100 and negative_files >= 1000 and background_files >= 1000:
    kalite_text = f"""[SUCCESS] **MUKEMMEL** - Veri seti dengeli ve yeterli buyuklukte
- Wakeword: {wakeword_files} [OK] (minimum 100+)
- Negative: {negative_files} [OK] (minimum 1000+)
- Background: {background_files} [OK] (minimum 1000+)"""
elif wakeword_files >= 50 and negative_files >= 500:
    kalite_text = f"""[WARNING] **ORTA** - Veri seti kabul edilebilir ama gelistirilebilir
- Wakeword: {wakeword_files} [{'[OK]' if wakeword_files >= 50 else '[WARN]'}] (minimum 100+ onerilir)
- Negative: {negative_files} [{'[OK]' if negative_files >= 500 else '[WARN]'}] (minimum 1000+ onerilir)
- Background: {background_files} [{'[OK]' if background_files >= 500 else '[WARN]'}] (minimum 1000+ onerilir)"""
else:
    kalite_text = f"""[ERROR] **ZAYIF** - Veri seti yetersiz, egitim basarisi dusuk olacak
- Wakeword: {wakeword_files} [ERROR] (minimum 100+ gerekli)
- Negative: {negative_files} [ERROR] (minimum 1000+ gerekli)
- Background: {background_files} [ERROR] (minimum 1000+ gerekli)"""

data_info += f"""
{kalite_text}
"""

print("=== GELISTIRILMIS LOAD DATASETS CIKTISI ===")
print(data_info)
print("\n=== TEST BASARILI ===")
print("Yeni Load Datasets ciktisi Markdown tablolari, ayrintili istatistikler")
print("ve kalite degerlendirmesi ile insana daha okunabilir formatta!")
