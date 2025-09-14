#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paralel, RAM-dostu Ses Dosyaları Analiz Scripti
- 250k+ dosyada bile şişmeyen, akışkan toplama (streaming) mantığı
- Çok çekirdekli paralel süre okuma (process veya thread)
- İsteğe bağlı CSV çıkışı: detayları dosyaya yazar, RAM'i sabit tutar
"""

import os
import sys
import csv
import signal
from pathlib import Path
from datetime import timedelta
import mimetypes
import heapq
from typing import Optional, Tuple, Iterable

# =======================
# Konfigürasyon
# =======================
CONFIG = {
    # ---- Paralel ayarlar ----
    #    "WORKER_COUNT": max(1, (os.cpu_count() or 2) - 1),
    # Senin makinede (7950X) iyi başlangıç:
    "BACKEND": "process",          # "process" = çok çekirdek, "thread" = I/O ağırlık
    "WORKER_COUNT": 24,            # 16C/32T sistemde 20-28 arası genelde iyi
    "CHUNKSIZE": 512,              # Çok dosyada IPC overhead'i azaltır (256-1024 aralığı mantıklı)

    # ---- Tarama/filtreleme ----
    "USE_MIME_GUESS": False,       # Performans için kapalı. Sadece uzantıyla filtreler.
    "AUDIO_EXTS": {
        '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac',
        '.wma', '.opus'
    },

    # ---- Çıkış/raporlama ----
    "PROGRESS_EVERY": 2000,        # Her N dosyada bir ilerleme yazdır
    "DETAILS_CSV": "ses_dosyalari_detay.csv",  # Detay CSV'ye yaz (None yaparsan CSV yazmaz)
    "KEEP_IN_MEMORY_DETAILS": False,           # True yaparsan RAM'de de liste tutar (250k+ için önerilmez)
    "ERROR_EXAMPLES_LIMIT": 5,     # Örnek hata dosyası sayısı
    "TOP_N": 3,                    # En uzun/en kısa listesinde kaç öğe tutulsun
}

# ---- Mutagen kontrolü ----
try:
    from mutagen import File as MutagenFile
except ImportError:
    print("Mutagen kütüphanesi bulunamadı. Yüklemek için:\n  pip install mutagen")
    sys.exit(1)

# ---- İşçi fonksiyonu (multiprocessing için üst seviyede olmalı) ----
def probe_duration(file_path_str: str) -> Tuple[str, int, float, Optional[str]]:
    """
    Tek dosyanın (path, size, duration, error) bilgisini döndürür.
    Multiprocessing ile picklable olduğundan üst seviye fonksiyon.
    """
    try:
        p = Path(file_path_str)
        try:
            size = p.stat().st_size  # // Dosya boyutu
        except Exception:
            size = 0

        duration = 0.0
        try:
            af = MutagenFile(str(p))
            if af is not None and hasattr(af.info, "length"):
                duration = float(af.info.length)  # // Süre (sn)
            else:
                duration = 0.0
        except Exception as e:
            return (file_path_str, size, 0.0, str(e))

        return (file_path_str, size, duration, None)
    except Exception as e:
        return (file_path_str, 0, 0.0, str(e))

def iter_files_scandir(root: Path, exts: set, use_mime: bool) -> Iterable[str]:
    """
    os.scandir ile hızlı ve düşük overhead'li derin tarama (stack tabanlı, recursion yok).
    Yalnızca uzantı (ve istenirse MIME) ile filtre.
    yield: str path
    """
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            # // Uzantı kontrolü
                            _, ext = os.path.splitext(entry.name)
                            ext = ext.lower()
                            if ext in exts:
                                yield entry.path
                            elif use_mime:
                                # // MIME fallback (yavaş olabilir)
                                mime, _ = mimetypes.guess_type(entry.path)
                                if mime and mime.startswith("audio/"):
                                    yield entry.path
                    except Exception:
                        # Erişim/izin/symlink vb. lokal hata: geç
                        continue
        except Exception:
            # Klasör okunamıyorsa geç
            continue

class Stats:
    """
    RAM dostu istatistik toplayıcı.
    - Toplam süre/boyut/sayı
    - Uzantı dağılımı
    - En uzun / en kısa TOP_N dosya (heap ile)
    - Örnek hata kaydı
    - İsteğe bağlı CSV'ye satır satır yazım
    """
    def __init__(self, top_n: int, details_csv: Optional[str], keep_in_memory: bool, error_limit: int):
        self.count = 0
        self.total_duration = 0.0
        self.total_size = 0
        self.ext_counts = {}
        self.error_count = 0
        self.error_examples = []
        self.error_limit = error_limit

        # En uzun için min-heap (en küçük başta, büyük gelirse pushpop)
        self.top_longest = []  # list[(duration, name, path)]
        # En kısa için max-heap (negatif duration ile)
        self.top_shortest = [] # list[(-duration, name, path)]
        self.top_n = top_n

        self.details_list = [] if keep_in_memory else None
        self.csv_path = details_csv
        self.csv_file = None
        self.csv_writer = None
        if self.csv_path:
            # // CSV'yi aç ve header yaz
            self.csv_file = open(self.csv_path, "w", encoding="utf-8", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["name", "path", "size_bytes", "duration_seconds"])

    def add_record(self, path_str: str, size: int, duration: float):
        name = os.path.basename(path_str)
        _, ext = os.path.splitext(name)
        ext = ext.lower()

        self.count += 1
        self.total_duration += duration
        self.total_size += size
        self.ext_counts[ext] = self.ext_counts.get(ext, 0) + 1

        # Detay satırı -> CSV'ye akıt
        if self.csv_writer:
            self.csv_writer.writerow([name, path_str, size, f"{duration:.6f}"])

        # İstenirse RAM'de de tut
        if self.details_list is not None:
            self.details_list.append({"name": name, "path": path_str, "size": size, "duration": duration})

        # En uzun TOP_N (duration > 0 olmalı)
        if duration > 0:
            if len(self.top_longest) < self.top_n:
                heapq.heappush(self.top_longest, (duration, name, path_str))
            else:
                if duration > self.top_longest[0][0]:
                    heapq.heapreplace(self.top_longest, (duration, name, path_str))

            # En kısa TOP_N (negatif duration ile max-heap davranışı)
            if len(self.top_shortest) < self.top_n:
                heapq.heappush(self.top_shortest, (-duration, name, path_str))
            else:
                if duration < -self.top_shortest[0][0]:
                    heapq.heapreplace(self.top_shortest, (-duration, name, path_str))

    def add_error(self, path_str: str, err: str):
        self.error_count += 1
        if len(self.error_examples) < self.error_limit:
            self.error_examples.append((os.path.basename(path_str), err))

    def close(self):
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()

    # Yardımcı formatlayıcılar
    @staticmethod
    def fmt_duration(seconds: float) -> str:
        if seconds <= 0:
            return "Süre alınamadı"
        td = timedelta(seconds=int(seconds))
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        if td.days > 0:
            hours += td.days * 24
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def fmt_size(size_bytes: int) -> str:
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"

def analyze(folder_path: str):
    root = Path(folder_path)
    if not root.exists():
        print(f"❌ Hata: '{root}' klasörü bulunamadı!")
        return
    if not root.is_dir():
        print(f"❌ Hata: '{root}' bir klasör değil!")
        return

    print(f"\n📁 Taranan klasör: {root}")
    print("=" * 60)
    print("🔍 Ses dosyaları aranıyor...\n")

    stats = Stats(
        top_n=CONFIG["TOP_N"],
        details_csv=CONFIG["DETAILS_CSV"],
        keep_in_memory=CONFIG["KEEP_IN_MEMORY_DETAILS"],
        error_limit=CONFIG["ERROR_EXAMPLES_LIMIT"],
    )

    # Aday dosyaları LAZILY üret (RAM şişmez)
    candidates = iter_files_scandir(
        root=root,
        exts=CONFIG["AUDIO_EXTS"],
        use_mime=CONFIG["USE_MIME_GUESS"]
    )

    backend = CONFIG["BACKEND"].lower().strip()
    max_workers = int(CONFIG["WORKER_COUNT"])
    chunksize = int(CONFIG["CHUNKSIZE"]) if CONFIG["CHUNKSIZE"] else 64

    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    processed = 0
    print(f"⚙️  Paralel çalışma: {backend} (işçi sayısı: {max_workers}, chunksize: {chunksize})")

    # Ctrl+C ile kibar çıkış
    interrupted = {"flag": False}
    def _handle_sigint(sig, frame):
        interrupted["flag"] = True
        print("\n🛑 Kesildi, mevcut sonuçlar yazdırılıyor...")
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        with Executor(max_workers=max_workers) as ex:
            for (path_str, size, duration, err) in ex.map(probe_duration, candidates, chunksize=chunksize):
                processed += 1

                if err:
                    stats.add_error(path_str, err)
                else:
                    stats.add_record(path_str, size, duration)

                if processed % CONFIG["PROGRESS_EVERY"] == 0:
                    print(f"… {processed} dosya işlendi")

                if interrupted["flag"]:
                    break
    finally:
        stats.close()

    print(f"✅ Tamamlandı: {processed} dosya işlendi.")
    print_summary(stats, root)

def print_summary(stats: Stats, root: Path):
    print("\n" + "=" * 60)
    print("📊 ANALIZ SONUÇLARI")
    print("=" * 60)
    print(f"\n📁 Taranan klasör: {root}")
    print(f"🎵 Toplam ses dosyası sayısı: {stats.count}")

    if stats.count > 0:
        total_hours = stats.total_duration / 3600.0
        print(f"⏱️  Toplam süre: {Stats.fmt_duration(stats.total_duration)} ({total_hours:.2f} saat)")
        avg = stats.total_duration / stats.count
        print(f"📊 Ortalama dosya süresi: {Stats.fmt_duration(avg)}")
        print(f"💾 Toplam boyut: {Stats.fmt_size(stats.total_size)}")

        print("\n📂 Dosya türlerine göre dağılım:")
        for ext, cnt in sorted(stats.ext_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (cnt / stats.count) * 100
            print(f"   {ext or '(uzantısız)'}: {cnt} dosya ({percentage:.1f}%)")

        # En uzun
        longest = sorted(stats.top_longest)  # min-heap -> artan
        if longest:
            print("\n⏱️  En uzun dosyalar:")
            for dur, name, path_str in longest[::-1]:
                print(f"   • {name}: {Stats.fmt_duration(dur)}  | {path_str}")

        # En kısa
        shortest = sorted(stats.top_shortest)  # max-heap (negatif) -> artan
        if shortest:
            print("\n⏱️  En kısa dosyalar:")
            for neg_dur, name, path_str in shortest:
                dur = -neg_dur
                print(f"   • {name}: {Stats.fmt_duration(dur)}  | {path_str}")

    if stats.error_count:
        print(f"\n⚠️  Süre bilgisi alınamayan dosya sayısı: {stats.error_count}")
        if stats.error_examples:
            print("   Örnekler:")
            for name, err in stats.error_examples:
                print(f"   • {name}  ({err})")

    if CONFIG["DETAILS_CSV"]:
        print(f"\n📄 Detay CSV: {CONFIG['DETAILS_CSV']}")

    print("\n" + "=" * 60)

def main():
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("🎵 Analiz edilecek klasörün yolunu girin: ").strip()
        folder_path = folder_path.strip('"').strip("'")
    if not folder_path:
        folder_path = "."
        print("ℹ️  Klasör belirtilmedi, mevcut klasör kullanılıyor.")

    analyze(folder_path)
    print("\n✅ Analiz tamamlandı!")

if __name__ == "__main__":
    main()
