import argparse
import os
import itertools
import pandas as pd
import numpy as np
import torch
from models import FSRCNN

def get_model_params(d, s, m, scale, input_size=(1, 33, 33), device='cpu'):
    print(f"[DEBUG] Attempting d={d}, s={s}, m={m}")
    try:
        model = FSRCNN(scale_factor=scale, d=d, s=s, m=m, num_channels=1).to(device)

        # 실제 파라미터 수 계산
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[DEBUG] SUCCESS d={d}, s={s}, m={m} -> param_count={param_count}")
        return param_count

    except Exception as e:
        print(f"[ERROR] d={d}, s={s}, m={m} -> {e}")
        return -1



# 전체 조합 생성 및 조건 필터링
def generate_candidates(d_max, s_max, m_max, param_limit, scale, num_splits, output_dir):
    results = []

    total_combinations = d_max * s_max * m_max
    print(f"[INFO] Total combinations to check: {total_combinations}")
    #2025.04.17 21:00 수정
    for d in range(4, d_max + 1, 4):
        for s in range(3, s_max + 1, 3):
            if s >= d:
                continue
            for m in range(2, m_max + 1, 2):
                if m >= s:
                    continue

                param_count = get_model_params(d, s, m, scale=scale)
                print(f"[INFO] Trying d={d}, s={s}, m={m} -> param_count={param_count}")
    
                if 0 < param_count <= param_limit:
                    results.append({
                        "d": d,
                        "s": s,
                        "m": m,
                        "param_count": param_count
                    })


    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "search_candidates.csv"), index=False)

    splits = np.array_split(df, num_splits)
    for i, split_df in enumerate(splits):
        split_df.to_csv(os.path.join(output_dir, f"search_part_{i}.csv"), index=False)

    return df, splits

# ✅ 메인 실행부
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d-max', type=int, required=True)
    parser.add_argument('--s-max', type=int, required=True)
    parser.add_argument('--m-max', type=int, required=True)
    parser.add_argument('--param-limit', type=int, required=True)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--num-splits', type=int, default=3)
    parser.add_argument('--output-dir', type=str, default="auto_search")
    args = parser.parse_args()

    df, splits = generate_candidates(
        d_max=args.d_max,
        s_max=args.s_max,
        m_max=args.m_max,
        param_limit=args.param_limit,
        scale=args.scale,
        num_splits=args.num_splits,
        output_dir=args.output_dir
    )

    print(f"\n[INFO] Valid combinations found: {len(df)}")
    for i, part in enumerate(splits):
        print(f"[INFO] search_part_{i}.csv: {len(part)} rows")
    print(f"[INFO] All saved in: {args.output_dir}/search_candidates.csv")
