import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # 🔥 추가: x축 포맷 바꾸려고

# 데이터 불러오기
df = pd.read_csv("summary_results.csv")

# 플롯 시작
plt.figure(figsize=(10, 6))
plt.scatter(df["param_count"], df["psnr"], c="royalblue", edgecolors="black", alpha=0.75)

# 제목과 레이블
plt.title("PSNR vs Number of Parameters (FSRCNN)", fontsize=14)
plt.xlabel("Number of Parameters", fontsize=12)
plt.ylabel("PSNR", fontsize=12)

# 🔥 여기 추가: x축을 K단위로 포맷팅
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}K'))

plt.grid(True)
plt.tight_layout()
plt.savefig("results/summary_plot.png")
plt.show()
