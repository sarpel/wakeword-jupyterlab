# Uygulama Planı — PR Yorumlarından Çıkan Düzeltmeler (yalnızca plan)

Bu belge, aktif ve (erişilebilen) geçmiş PR’lerdeki yorumlardan türeyen, uygulanması gereken düzeltmelerin eylem planıdır. Bu aşamada hiçbir kod değişikliği yapmayacağız; yalnızca planı hazırlıyoruz.

Kaynaklar:
- Aktif PR: Feat/training live control and resume — PR #3
  - URL: https://github.com/sarpel/wakeword-jupyterlab/pull/3
  - Not: Bu PR üzerindeki pek çok yorum “unresolved” durumda ve uygulanması gereken düzeltmeleri listeliyor.
- Geçmiş PR’ler: Yerel `git log --merges` ile birleştirme commit’leri listelenemedi; GitHub araması depo-özel sonuç döndürmedi. Bu nedenle “geçmiş PR doğrulama” adımı plana dahil edildi (A2).

## A. Kapsam ve Önceliklendirme

Öncelik sırası, çalışma zamanı hatalarını ve mimari tutarsızlıkları önceleyen şekilde belirlenmiştir:
1) Çalışma zamanı kırılmaları ve veri boru hattı (dataset/feature) stabilizasyonu
2) Tekil model tanımı ve mimari konsolidasyon
3) Eğitim döngüsü/servis katmanı sağlamlaştırma (background training, durdur/yeniden başlat)
4) Kurulum ve indirme yardımcıları (setup skriptleri) sertleştirme
5) Belgeleme ve sürüm/doğruluk düzeltmeleri
6) Gereksinimler (dependencies) ve güvenlik düzeltmeleri
7) `.gitignore`/`.vscode` kuralları ve repo hijyeni
8) Notebook ve raporlama uyum düzeltmeleri
9) Test eklemeleri ve doğrulama

## B. Uygulanacak Değişiklikler (PR yorumlarına haritalı)

A1. Dataset ve Feature boru hattı stabilizasyonu
- B1.1 `training/feature_extractor.py`: “Line 190 duplicates line 189, overwriting n_features” — n_features üzerine yazma hatasını düzelt (Copilot review yorumu).
  - Kabul: Delta özellikler açıkken çıkış boyutu doğru hesaplanmalı; ilgili test (bkz. T1) geçmeli.
- B1.2 `gradio_app.py` → `EnhancedWakewordDataset.__getitem__`: `audio is None` dalında `mel_tensor` tanımlanmıyor (UnboundLocalError). Sıfır mel tensörü oluştur (N_MELS×frames), contiguous float32 → torch tensor + batch/channel boyutlarını ekle (CodeRabbit + ChatGPT/Coder yorumları).
  - Kabul: Kötü/okunamayan dosyada loader çökmeden örneği atlar veya sıfır tensör ile döner; unit test T1.2 geçer.
- B1.3 `training/enhanced_dataset.py`: Göreli import doğrudan çalıştırmada patlayabilir — `__main__` koruması veya mutlak import stratejisi ekle (Copilot review yorumu).
  - Kabul: `python training/enhanced_dataset.py` ve paket içinden import ikisi de çalışır.

A2. Geçmiş PR doğrulaması ve geriye dönük aksiyonlar
- B2.1 GitHub üzerinde “Closed & Merged” PR’leri gözden geçir: uygulanmamış yorum var mı? (Yerel `git log --merges` sinyal vermediği için GitHub UI üzerinden manuel tarama yapılacak.)
  - Kabul: Yeni aksiyon çıkarsa bu plana “B2.x” maddeleri olarak eklensin; mevcut maddeler güncellensin.

A3. Tekil model tanımı ve mimari konsolidasyon
- B3.1 Tek bir kanonik model modülü oluştur: `training/model.py` (örn. `EnhancedWakewordModel`) ve tüm kullanım yerlerinde (`gradio_app.py`, `training/enhanced_trainer.py`, notebook’lar) bunu import et (Gemini critical yorumları).
  - Kabul: Projede birden fazla model tanımı kalmasın; tüm eğitim/tahmin yolları aynı modülü kullansın.
- B3.2 LSTM zaman ekseni kaybı: `AdaptiveAvgPool2d((1,1))` sonrası tek zaman adımı oluşuyor — pooling/reshape akışını zaman eksenini koruyacak şekilde güncelle (CodeRabbit yorumu). CNN+pool → (B, C, H, W) → permute (B, W, C×H) → LSTM.
  - Kabul: Forward boyutları ve LSTM sıralı girişleri doğru; unit test T3 geçer.

A4. Eğitim denetimi ve iş parçacığı (background training)
- B4.1 `training/enhanced_trainer.py`: Placeholder dönüş kaldır; thread’li eğitim döngüsü veya mevcut training loop ile bütünleşik start/stop mekanizması uygula (Copilot review yorumu). `stop_training` bayrağı, durum güncellemeleri, hata yakalama eklensin.
  - Kabul: Gradio arayüzünden başlat/durdur çalışır; kısa duman testi (task: Headless 1-epoch smoke) geçer.

A5. Raporlama ve hata mesajları
- B5.1 `scripts/report_metrics.py`: Hata mesajı İngilizce standarda çekilsin (Copilot review yorumu).
  - Kabul: Mesajlar İngilizce; lint/test şikâyeti yok.

A6. Kurulum yardımcıları ve indirme sertleştirmeleri
- B6.1 `setup/setup_env.py`: Paket kurulumunu `requirements.txt`’den yap (Gemini yorumu); `run_command` güvenli (shell=False, `shlex.split`) hale getir (CodeRabbit yorumu); CUDA uygunluğu testinde `torch.cuda.is_available()` False ise testi fail et (CodeRabbit yorumu).
  - Kabul: Kurulum senaryosu `-r requirements.txt` kullanır; komut çalıştırıcı shell=False; CUDA testi başarısızsa toplam sonuç False döner; unit test T6.1 geçer.
- B6.2 `setup/setup_rirs.py`: `download_file` için URL şema doğrulaması (yalnız http/https), timeout ve streaming indirme; incomplete dosya temizliği; import’ların temizlenmesi (`urlparse`, kullanılmayanların kaldırılması) (CodeRabbit yorumu).
  - Kabul: Büyük dosyalarda ilerleme ve güvenli indirme çalışır; unit test T6.2 (mock’lu) geçer.

A7. Belge ve örnekler
- B7.1 `PROJECT_CLEANUP_SUMMARY.md`: Yanlış Gradio sürüm ifadesini düzelt (Gemini yorumu). Not: requirements pin’i ile uyumlu hale getir.
- B7.2 `docs/ENHANCED_FEATURES_README.md`: Yanlış script adı/path — `python setup/setup_rirs.py ...` olarak düzelt (Coderabbit yorumu).
- B7.3 `docs/COMPREHENSIVE_TRAINING_GUIDE.md`:
  - PSOLA yerine phase‑vocoder açıklaması (librosa time_stretch/pitch_shift) — iki bölüm (737–744, 774–778) (Coderabbit yorumu).
  - Mimari bölümü, gerçek modele göre güncelle (AdaptiveAvgPool→LSTM yerine 3×Conv+MaxPool→LSTM(+attention)) (Coderabbit yorumu).
  - Eksik fenced language uyarılarını gider (MD040).
  - Kabul: Belge‑özgü diff’ler uygulanır; lint uyarıları gider.

A8. Gereksinimler (dependencies) ve güvenlik
- B8.1 `requirements.txt`: `gradio>=5.0.0` (kritik SSRF yaması için). Gerekirse `>=4.44.0` fallback değerlendirilir. (Coderabbit güvenlik yorumu.)
- B8.2 `requirements.txt`: PyTorch hattı — güncel güvenli min. sürüme yükseltme (örn. `torch>=2.6,<2.9`) veya mevcut GPU/OS uyumluluğunu korumak için iki profil:
  - GPU profil: CUDA hattına uygun pins (ortama göre doğrulanacak)
  - CPU fallback profil: Workspace task “Install PyTorch CPU as fallback” ile hizalı
  - Kabul: Ortamda en az biri başarıyla kurulabilir; duman testi çalışır.

A9. Repo hijyeni
- B9.1 `.gitignore`: `*.sh`, `*.ps1`, `*.cmd` gibi küresel script ignore kaldır; gerekiyorsa klasör‑bazlı ignore’lara taşı (Coderabbit yorumu).
- B9.2 `.gitignore` ve `.vscode`: `.vscode/` dizinini ignore ettikten sonra yeniden dahil etme işe yaramaz; desenleri sıraya/görünüme göre düzelt (Coderabbit yorumu).
  - Kabul: `settings.json`, `tasks.json`, `launch.json`, `extensions.json` sürüm takibine düzgün döner.

A10. Notebook ve checkpoint tutarlılığı
- B10.1 `colab_wakeword_training.ipynb`: `save_ckpt` iki kez yazma yerine versiyonlu dosyayı yaz + `last_checkpoint.pth` için kopyala (Gemini yorumu).
- B10.2 Notebook değerlendirme ve deployment hücreleri: checkpoint anahtar adları (`model_state_dict` vs `model_state`) tutarlı hale getir; validasyon raporlaması ve çıktıları sadeleştir (PR diff’lerinde belirtilen alanlarla hizala).
  - Kabul: Notebook uyarısız çalışır; model yükleme anahtarları tek tip.

A11. Gradio katmanı sadeleştirme (thin UI)
- B11.1 `gradio_app.py` dosyasında core bileşen tanımlarını kaldırıp `training/` paketinden import et (Gemini critical yorumları). UI, orkestrasyon katmanı olarak kalmalı.
  - Kabul: `gradio_app.py` minimal kalır; training paketinde test edilen fonksiyonları çağırır.

A12. Raporlama (scripts/report_metrics) ve dil
- B12.1 Hata mesajlarını İngilizce standarda çek (Copilot review). (Bkz. B5.1 ile aynı; madde birleşebilir.)

## C. Test Planı (eklenecek yeni testler)
- T1: Feature/dataset boyut ve dayanıklılık testleri
  - T1.1 `training/feature_extractor.py` için delta açık/kapalı boyut kontrolü (n_features overwrite regression testi).
  - T1.2 Dataset `audio=None` dalında `mel_tensor` tanımlı ve şekli `(1, N_MELS, frames)` — yapay bozuk dosya ile DataLoader smoke.
- T2: `enhanced_dataset` import/çalıştırma testi (doğrudan çalıştırma + paket içinden import).
- T3: Model forward LSTM zaman ekseni testi — CNN çıkışı → (B, W, C×H) reshape doğrulaması, rastgele girişle bir forward.
- T4: `setup_env.py` komut çalıştırıcı (shell=False) ve CUDA availability testi (mock’lu), requirements üzerinden kurulum çağrısı üretimi.
- T5: `setup_rirs.py` indirme yardımcıları — http/https dışını reddetme, timeout ve streaming (mock’lu URL ile).
- T6: `.gitignore`/`.vscode` pattern testi — örnek dosyalarla git status simülasyonu (belgeleme amaçlı; otomasyon opsiyonel).
- T7: Notebook checkpoint anahtarları tutarlılık testi (en azından py betiği ile checkpoint yükleme validation’ı).

## D. Kabul Kriterleri (özet)
- Tüm “unresolved” PR #3 yorumları karşılık gelen değişikliklerle kapatılır (veya gerekçeyle reddedilir).
- Testler (T1–T5 zorunlu, T6–T7 önerilen) geçer.
- Gradio uygulaması başlar ve temel fonksiyonlar çalışır (Start/Stop training; inference demo).
- Task: “Headless 1-epoch smoke” başarılı tamamlanır.
- Belge güncellemeleri sürümler ve gerçek kodla uyumludur.

## E. Riskler ve Azaltmalar
- Gradio ≥5.0.0 ve PyTorch yükseltmeleri mevcut ortamda uyumsuzluk yaratabilir.
  - Azaltma: CPU fallback task; GPU profili için ayrı `requirements-gpu.txt` opsiyonu; kademeli yükseltme ve pin’ler.
- Model konsolidasyonu kısa vadede import kırılmalarına yol açabilir.
  - Azaltma: `training/model.py` eklenirken eski sınıf adları için geçici alias (deprecation uyarısıyla) sağla; CI/test ile doğrula.
- `setup_rirs` indirmelerinde ağ kısıtları.
  - Azaltma: Timeout ve tekrar deneme, kısmi dosya temizliği.

## F. Uygulama Sırası (milestones)
1) B1.1, B1.2, B1.3 — veri/feature stabilizasyonu ve import güvenliği
2) B3.1, B3.2 — model konsolidasyonu ve LSTM akışı
3) B4.1 — background training kontrolü (gradio entegrasyonu dahil)
4) B6.1, B6.2 — setup sertleştirmeleri
5) B7.* — dokümantasyon düzeltmeleri
6) B8.* — dependency yükseltmeleri (guarded rollout)
7) B9.* — `.gitignore`/`.vscode` temizlik
8) B10.* — notebook uyarlamaları
9) Testler (T1–T5 min.), smoke run ve PR yorumlarının “resolved” edilmesi

## G. İlgili Dosyalar (değişecek/eklenecek)
- `training/feature_extractor.py` (B1.1)
- `training/enhanced_dataset.py` (B1.3)
- `gradio_app.py` (B1.2, B11.1)
- `training/model.py` (yeni) (B3.1)
- `training/enhanced_trainer.py` (B3.2, B4.1)
- `scripts/report_metrics.py` (B5.1, B12.1)
- `setup/setup_env.py` (B6.1)
- `setup/setup_rirs.py` (B6.2)
- `docs/ENHANCED_FEATURES_README.md` (B7.2)
- `docs/COMPREHENSIVE_TRAINING_GUIDE.md` (B7.3)
- `PROJECT_CLEANUP_SUMMARY.md` (B7.1)
- `requirements.txt` (B8.1, B8.2)
- `.gitignore`, `.vscode/*` (B9.1, B9.2)
- `colab_wakeword_training.ipynb` (B10.1, B10.2)
- `tests/…` (yeni/ek testler: T1–T5)

## H. İzleme ve Doğrulama
- PR #3 içindeki her yorumun bir “B*.#” plan maddesine eşlemesi yapılmıştır; uygulama sonrası aynı yorum başlıklarında “Resolved with commit <sha> — see Bx.y” notu düşülecek.
- Geçmiş PR’ler (A2/B2.1) için GitHub UI üzerinden manuel tarama yapılacak; gerektiğinde bu plan yeni maddelerle güncellenecek.

---
## I. Hızlı Durum Değerlendirmesi (kodla karşılaştırma)

Aşağıdaki maddeler, plan (B*) ile mevcut kodun kısa taramasının eşleşmesini ve mevcut durumu özetler:

- B1.1 `training/feature_extractor.py` — n_features üzerine yazma:
  - Durum: Uygulanmamış. `except` bloğunda `n_features = self.feature_config.n_mels` satırı iki kez bulunuyor; delta/delta-delta ekleri aşağıda artıyor (beklenen düzeltme: mükerrer satırı kaldırıp toplamı doğru hesaplamak).
- B1.2 `gradio_app.py` — `EnhancedWakewordDataset.__getitem__` `audio is None` dalı:
  - Durum: Uygulanmamış. `audio is None` durumunda yalnızca `mel_spec = zeros(...)` atanıyor; `return` kısmında `mel_tensor` kullanıldığı için `UnboundLocalError` riski var. Sıfır mel tensörü `mel_tensor` olarak oluşturulmalı.
- B1.3 `training/enhanced_dataset.py` — göreli importun doğrudan çalıştırmada hatası:
  - Durum: Uygulanmamış. Dosya paket içi göreli import (`from .feature_extractor ...`) kullanıyor; doğrudan script çalıştırmada patlayabilir. Mutlak import veya `__main__` koruması eklenmeli.
- B3.1/B11.1 Model ve UI ayrımı (tekil model tanımı, ince UI):
  - Durum: Kısmen uyumsuz. `gradio_app.py` içinde dataset/model/trainer benzeri çekirdek bileşenler mevcut. `training/enhanced_trainer.py` içinde de Gradio/GUI karışımı var. Tekil model modülü ve ince UI hedeflenmeli.
- B3.2 LSTM zaman ekseni kaybı:
  - Durum: İncelenmeli. `enhanced_trainer.py` içindeki `EnhancedWakewordModel.forward` CNN→pool sonrası `x.view(batch_size, -1, x.size(2) * x.size(3))` kullanıyor; zaman ekseni muhafazası için permute/reshape sözleşmesi netleştirilmeli ve test (T3) eklenmeli.
- B6.1 `setup/setup_env.py` — güvenli komutlar ve requirements tabanlı kurulum:
  - Durum: Uygulanmamış. `run_command` `shell=True` çalışıyor ve paketler tek tek kuruluyor; `requirements.txt` kullanılmıyor. CUDA kontrolü `nvidia-smi` ile sınırlı; `torch.cuda.is_available()` doğrulaması test aşamasında var ancak “başarısızsa fail” politikası netleştirilmeli.
- B6.2 `setup/setup_rirs.py` — URL şeması, timeout, streaming, import temizliği:
  - Durum: Dosya bu taramada incelenmedi; plan kapsamında yapılacak.
- B7.* Belgeler — sürüm/isimlendirme/fenced language düzeltmeleri:
  - Durum: İncelenmedi; plan kapsamında güncellenecek.
- B8.1 `requirements.txt` — `gradio>=5.0.0`:
  - Durum: Uygulanmamış. Şu an `gradio>=4.0.0`. Güvenlik notuna göre 5.x’e yükseltilmeli (uyumluluk testleri ile).
- B8.2 PyTorch profilleri (GPU/CPU):
  - Durum: Kısmen uyumlu. Dosya `torch==2.1.2+cu118` hattına sabit; workspace’te CPU fallback task mevcut (2.3.1+cpu). Profil ayrımı ve uyumluluk testleri plan kapsamında yapılacak.
- B9.* `.gitignore`/`.vscode` desenleri:
  - Durum: Uygulanmamış. Hem `.vscode/` hem `.vscode/*` ignore edilip alt dosyalar `!` ile yeniden dahil edilmiş; üst dizin ignore edildiğinde yalnız dosyaları unignore etmek yeterli olmayabilir. Dizin seviyesinde de unignore stratejisi gözden geçirilmeli.

## J. Geçmiş PR’ler Bulguları (Yerel)

- Yerel `git log --merges` çıktısı birleşme commit’i göstermedi (boş). Bu nedenle geçmiş “Closed & Merged” PR’lerin yorumlarını doğrulamak için GitHub UI üzerinden manuel tarama adımı A2/B2.1 olarak planda tutuldu.
- Aksiyon: GitHub deposunda “Pull requests → Closed → Merged” filtreleriyle son 10–20 PR hızla taransın; uygulanmamış yorumlar bulunursa bu dosyada “B2.x” maddeleri olarak eklensin.

---
Not: Bu dosya yalnızca planı içerir; kod değişiklikleri bundan sonraki adımda ayrı commit’lerle gerçekleştirilecektir.
