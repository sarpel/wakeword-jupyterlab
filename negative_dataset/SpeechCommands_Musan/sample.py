# sample_recursive.py
import os, sys, argparse, random, shutil, time

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def is_under(path, parent):
    try:
        return os.path.commonpath([os.path.abspath(path)]) == os.path.commonpath([os.path.abspath(parent), os.path.abspath(path)])
    except ValueError:
        return False

def iter_nonempty_dirs(root, exclude=None):
    root = os.path.abspath(root)
    exclude = os.path.abspath(exclude) if exclude else None
    for dirpath, _, filenames in os.walk(root):
        if exclude and os.path.commonpath([exclude, os.path.abspath(dirpath)]) == exclude:
            continue
        if filenames:
            yield dirpath

def list_files(folder, exts=None):
    out = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            if not exts:
                out.append(p)
            else:
                ext = os.path.splitext(name)[1].lower()
                if ext in exts:
                    out.append(p)
    return out

def parse_count(token, total):
    token = str(token).strip()
    if token.endswith("%"):
        pct = int(token[:-1])
        return max(1, int(total * pct / 100))
    return int(token)

def collect(src_root, count_token, dest_root, exts=None, dry_run=False, debug=False):
    src_root = os.path.abspath(src_root)
    dest_root = os.path.abspath(dest_root)

    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Kaynak klasör yok: {src_root}")
    ensure_dir(dest_root)

    if exts:
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

    dirs = list(iter_nonempty_dirs(src_root, exclude=dest_root))
    if debug:
        print(f"🗂️  Dosya içeren klasör sayısı (tüm seviye): {len(dirs)}")
        for d in dirs[:20]:
            print(f"  • {d}")
        if len(dirs) > 20:
            print("  …")

    moved_total = 0
    for i, d in enumerate(dirs, 1):
        files = list_files(d, exts)
        if not files:
            if debug: print(f"[{i}/{len(dirs)}] {d}: uygun dosya yok.")
            continue

        n = parse_count(count_token, len(files))
        chosen = files if n >= len(files) else random.sample(files, n)

        if debug:
            print(f"[{i}/{len(dirs)}] {d}: {len(files)} dosya, seçilecek {len(chosen)}")

        for src in chosen:
            dst = os.path.join(dest_root, os.path.basename(src))
            if os.path.abspath(src) == os.path.abspath(dst):
                if debug: print(f"  ↷ Atlandı (aynı yol): {src}")
                continue
            if os.path.exists(dst):
                base, ext = os.path.splitext(os.path.basename(src))
                dst = os.path.join(dest_root, f"{base}_{int(time.time()*1000)}_{random.randint(1000,9999)}{ext}")
            if dry_run:
                print(f"  ▶ DRY-RUN: {src} → {dst}")
            else:
                shutil.move(src, dst)
                moved_total += 1
                if debug:
                    print(f"  ✓ Taşındı: {src} → {dst}")

    print(f"\nBitti. Toplam taşınan: {moved_total}")

def main():
    ap = argparse.ArgumentParser(description="Tüm seviye subfolder'lardan rastgele dosya taşıma")
    ap.add_argument("src_root", help="Kaynak kök klasör")
    ap.add_argument("count", help="Adet ya da yüzde, örn: 1000 veya 30%")
    ap.add_argument("dest_root", help="Hedef klasör")
    ap.add_argument("--ext", nargs="*", default=None, help="Uzantı filtresi, örn: wav mp3 flac")
    ap.add_argument("--dry-run", action="store_true", help="Taşımadan önce sadece listele")
    ap.add_argument("--debug", action="store_true", help="Detaylı çıktı")
    args = ap.parse_args()
    collect(args.src_root, args.count, args.dest_root, args.ext, args.dry_run, args.debug)

if __name__ == "__main__":
    main()
