import os
import time
import pandas as pd
import shutil

train_file = "91-image_x4.h5"
eval_file = "Set5_x4.h5"
output_root = "auto_search"
scale = 4
epochs = 10
sample_image = "test_img.bmp"

df = pd.read_csv("auto_search/search_part_0.csv")

gpu_id = 0

log_path = os.path.join(output_root, f"log_gpu{gpu_id}.csv")
os.makedirs(output_root, exist_ok=True)

with open(log_path, "w") as logf:
    logf.write("d,s,m,param_count,psnr,train_time_s,status\n") # 헤더 작성


# 조합별 학습 및 평가
for index, row in df.iterrows():
    d, s, m, param_count = int(row.d), int(row.s), int(row.m), int(row.param_count)

    # 조합별 결과 디렉토리 설정
    output_dir = os.path.join(output_root, f"d{d}_s{s}_m{m}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[GPU{gpu_id}] Training d={d}, s={s}, m={m}...")

    start = time.time()

    # modified 04.15.18:08
    # train.py 실행
    cmd = (
        f"python train.py "
        f"--train-file {train_file} "
        f"--eval-file {eval_file} "
        f"--outputs-dir {output_dir} "
        f"--scale {scale} "
        f"--num-epochs {epochs} "
        f"--d {d} --s {s} --m {m} "
        f"--device cuda:{gpu_id} "
    )
    os.system(cmd)
    end = time.time()

    best_model = os.path.join(output_dir, "x4", "best.pth")
    psnr_val = -1.0
    status = "fail"


    # PSNR 측정 (test.py 사용)
    if os.path.exists(best_model) and os.path.exists(sample_image):
        psnr_txt = os.path.join("results", f"d{d}_s{s}_m{m}", "psnr.txt")

        # modified 04.16.10:07
        psnr_cmd = (
            f"python test.py "
            f"--weights-file {best_model} "
            f"--image-file {sample_image} "
            f"--scale {scale} "
            f"--d {d} --s {s} --m {m}"
        )
        
        os.system(psnr_cmd)

        try:
            with open(psnr_txt, "r") as f:
                lines = f.readlines()
                psnr_line = [l for l in lines if "PSNR" in l][-1]
                psnr_val = float(psnr_line.strip().split(":")[-1])
                if psnr_val >= 30.0:
                    status = "pass"
        
        except:
            psnr_val = -1.0
            status = "fail"

    print(f"[GPU{gpu_id}] DONE d={d}, s={s}, m={m} → PSNR={psnr_val:.2f}, time={end-start:.1f}s, status={status}")

    with open(log_path, "a") as logf:
        logf.write(f"{d},{s},{m},{param_count},{psnr_val},{round(end-start, 2)},{status}\n")

    
# 04.17 10:27 update
# 모든 조합 평가 완료 후
# 1. results/d_s_m/psnr.txt 읽어서 PSNR 비교
# 2. 가장 높은 PSNR의 best.pth 파일을 results/final 폴더에 복사

best_model_path = None
best_psnr_value = -1.0
best_model_name = ""

for index, row in df.iterrows():
    d, s, m = int(row.d), int(row.s), int(row.m)
    model_dir = os.path.join("results", f"d{d}_s{s}_m{m}")
    psnr_file = os.path.join(model_dir, "psnr.txt")
    weight_file = os.path.join("auto_search", f"d{d}_s{s}_m{m}", "x4", "best.pth")

    if os.path.exists(psnr_file):
        with open(psnr_file, "r") as f:
            psnr_line = f.read().strip()
            if "PSNR" in psnr_line:
                try:
                    psnr_value = float(psnr_line.split(":")[-1].strip())
                    if psnr_value > best_psnr_value:
                        best_psnr_value = psnr_value
                        best_model_path = weight_file
                        best_model_name = f"d{d}_s{s}_m{m}"
                except:
                    pass

# ✅ 최종 best 모델 저장
if best_model_path and os.path.exists(best_model_path):
    os.makedirs("results/final", exist_ok=True)
    shutil.copy(best_model_path, "results/final/final_best.pth")

    with open("results/final/final_best_model.txt", "w") as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"PSNR: {best_psnr_value:.2f}\n")

    print(f"[SUMMARY] Best model: {best_model_name}, PSNR: {best_psnr_value:.2f}")
    print("[SUMMARY] Saved to results/final/final_best.pth + final_best_model.txt")
