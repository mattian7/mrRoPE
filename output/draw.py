import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file1 = pd.read_csv("./output/proofpile-long-small-olmo-test2.csv", index_col=0)
file2 = pd.read_csv("./output/proofpile-long-small-olmo-yarn.csv", index_col=0)

tf1= file1.transpose()
tf2 = file2.transpose()



df1 = tf1.iloc[:,0]  # 第一列作为横坐标
x1 = df1.index.to_numpy() # 第一行作为横坐标s
y1 = df1.values

df2 = tf2.iloc[:,0]  # 第一列作为横坐标
x2 = df2.index.to_numpy() # 第一行作为横坐标
y2 = df2.values


# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x1, y1, label="Yarn-NTK", color='blue')
plt.plot(x2, y2, label="Yarn", color='orange')

plt.xlabel("Context Window")
plt.ylabel("Perplexity")
plt.title("Perplexity on Context Window")
plt.legend()
plt.grid(True)
plt.tight_layout()


# 保存图像
plt.savefig("./output/perplexity_olmo.png", dpi=300)  # 可调分辨率
plt.close()  # 关闭绘图以释放内存
