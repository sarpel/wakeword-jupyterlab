import os
import random
import shutil
import sys

def collect_random_samples(src_root, sample_count, dest_root):
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    subfolders = [os.path.join(src_root, d) for d in os.listdir(src_root)
                  if os.path.isdir(os.path.join(src_root, d))]

    for subfolder in subfolders:
        files = [os.path.join(subfolder, f) for f in os.listdir(subfolder)
                 if os.path.isfile(os.path.join(subfolder, f))]

        if len(files) == 0:
            continue

        if len(files) < sample_count:
            print(f"⚠️ {subfolder} içinde sadece {len(files)} dosya var.")
            chosen = files
        else:
            chosen = random.sample(files, sample_count)

        for f in chosen:
            dest_path = os.path.join(dest_root, os.path.basename(f))

            # Çakışma olursa isim değiştir
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(os.path.basename(f))
                dest_path = os.path.join(dest_root, f"{base}_{random.randint(1000,9999)}{ext}")

            shutil.move(f, dest_path)
            print(f"Taşındı: {f} → {dest_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Kullanım: python sample.py <kaynak_klasör> <örnek_sayısı> <hedef_klasör>")
        sys.exit(1)

    src_root = sys.argv[1]
    sample_count = int(sys.argv[2])
    dest_root = sys.argv[3]

    collect_random_samples(src_root, sample_count, dest_root)
