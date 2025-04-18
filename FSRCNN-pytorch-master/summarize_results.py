import os
import re
import pandas as pd
import torch
from models import FSRCNN

RESULTS_DIR = "results"
OUTPUT_CSV = "summary_results.csv"
BEST_TXT = os.path.join(RESULTS_DIR, "final", "best_psnr.txt")

def extract_dsm(folder_name):
    match = re.match(r'd(\d+)_s(\d+)_m(\d+)', folder_name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None

def read_psnr(psnr_path):
    try:
        with open(psnr_path, "r") as f:
            line = f.readline().strip()
            return float(line.split(":")[-1])
    except:
        return None

def get_param_count(d, s, m, scale=4):
    try:
        model = FSRCNN(scale_factor=scale, d=d, s=s, m=m, num_channels=1).to("cpu")
        return sum(p.numel() for p in model.parameters())
    except:
        return -1

def main():
    data = []

    for folder in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder)
        if not os.path.isdir(folder_path) or folder == "final":
            continue

        d, s, m = extract_dsm(folder)
        if d is None:
            print(f"[SKIP] Invalid folder name: {folder}")
            continue

        psnr_path = os.path.join(folder_path, "psnr.txt")
        psnr = read_psnr(psnr_path)
        param_count = get_param_count(d, s, m)

        if psnr is not None:
            data.append({
                "d": d,
                "s": s,
                "m": m,
                "psnr": psnr,
                "param_count": param_count
            })
        else:
            print(f"[WARN] psnr.txt missing or unreadable in: {folder}")

    if not data:
        print("[WARN] No valid data found. Check results/ for psnr.txt files.")
        return

    df = pd.DataFrame(data)
    df = df.sort_values(by="param_count", ascending=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Summary saved to {OUTPUT_CSV}")
    print(df.head())

    # 최고 PSNR 모델 저장
    os.makedirs(os.path.dirname(BEST_TXT), exist_ok=True)
    best = df.loc[df['psnr'].idxmax()]
    with open(BEST_TXT, "w") as f:
        f.write(f"d={best.d}, s={best.s}, m={best.m}, psnr={best.psnr:.4f}, param_count={best.param_count}\n")
    print(f"[INFO] Best PSNR saved to {BEST_TXT}")

if __name__ == "__main__":
    main()
