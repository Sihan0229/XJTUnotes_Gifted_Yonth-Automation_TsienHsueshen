---
layout: post  
title: "XJTU-AUTO300527 notes"  
date: 2024-05-29 00:10 +0800  
last_modified_at: 2024-06-02 14:00 +0800  
tags: [Course Note]  
math: true  r
toc: true  
excerpt: "机器学习课程一些疑难点的解决（TBC）"
---

# 求解线性回归模型（p173-178）
训练数据为𝑁个输入数据
$$\mathbf{X} =\begin{pmatrix}\mathbf{x_1},\mathbf{x_2},\cdots\mathbf{x_N}
\end{pmatrix}$$
及对应函数值
$$\mathbf{t} =\begin{pmatrix}{t_1},{t_2},\cdots{t_N}
\end{pmatrix}$$
模型为线性回归模型 
$$y(\mathbf{x},\mathbf{w}) =\mathbf{w}^T\mathbf{\phi(x)}$$

$$
  \mathbf{t} =\begin{pmatrix}
  t_{0}\\
  t_{1} \\
  \vdots \\  
  t_{N}  \\
  \end{pmatrix}_{N \times 1}
$$

$$
  \mathbf{w} =\begin{pmatrix}
  w_{0}\\
  w_{1} \\
  \vdots \\  
  w_{M-1}  \\
  \end{pmatrix}_{(M-1) \times 1}
$$


$$
  \mathbf{\phi}=
  \begin{pmatrix}
  \phi^T(x_1)\\
  \phi^T(x_2)\\
  \vdots \\  
  \phi^T(x_N)\\
  
  \end{pmatrix}=
  \begin{pmatrix}
  \phi_{0}(x_1) & \phi_{1}(x_1) & \phi_{2}(x_1) & \cdots & \phi_{M-1}(x_1)\\
  \phi_{0}(x_2) & \phi_{1}(x_2) & \phi_{2}(x_2) & \cdots & \phi_{M-1}(x_2)\\
  \vdots & \vdots & \ddots & \vdots \\  
  \phi_{0}(x_N) & \phi_{1}(x_N) & \phi_{2}(x_N) & \cdots & \phi_{M-1}(x_N)\\
  \end{pmatrix}_{N\times M}
$$

因此平方和误差函数

$$
 E_D(\mathbf{w})=\frac{1}{2}  \sum_{n=1}^N (t_n-\mathbf{w}^T \mathbf{\phi}(\mathbf{x}_n))^2 $$

 求导得

 $$\nabla E_D(\mathbf{w})=\sum_{n=1}^N (t_n-\mathbf{w}^T \mathbf{\phi}(\mathbf{x}_n))\mathbf{\phi}(\mathbf{x}_n)^T 
$$

合并可得

$$
\nabla E_D(\mathbf{w}) = \left(
\begin{pmatrix}
t_{0} &t_{1}&\cdots &t_{N}
\end{pmatrix} - \begin{pmatrix}
w_{0} &  w_{1}& \cdots & w_{M-1}
\end{pmatrix} \begin{pmatrix}
\phi(x_1) &  \phi(x_2)& \cdots & \phi(x_N)
\end{pmatrix}\right)

\begin{pmatrix}
\phi^T(x_1)\\
\phi^T(x_2)\\
\vdots \\  
\phi^T(x_N)\\

\end{pmatrix}
$$

即

$$\nabla E_D(\mathbf{w})= \left(\begin{pmatrix}
t_{0}\\
t_{1} \\
\vdots \\  
t_{N}  \\
\end{pmatrix}^T - \begin{pmatrix}
w_{0}\\
w_{1} \\
\vdots \\  
w_{M-1}  \\
\end{pmatrix}^T \begin{pmatrix}
\phi_{0}(x_1) & \phi_{1}(x_1) & \phi_{2}(x_1) & \cdots & \phi_{M-1}(x_1)\\
\phi_{0}(x_2) & \phi_{1}(x_2) & \phi_{2}(x_2) & \cdots & \phi_{M-1}(x_2)\\
\vdots & \vdots & \ddots & \vdots \\  
\phi_{0}(x_N) & \phi_{1}(x_N) & \phi_{2}(x_N) & \cdots & \phi_{M-1}(x_N)\\
\end{pmatrix}^T\right)\begin{pmatrix}
\phi_{0}(x_1) & \phi_{1}(x_1) & \phi_{2}(x_1) & \cdots & \phi_{M-1}(x_1)\\
\phi_{0}(x_2) & \phi_{1}(x_2) & \phi_{2}(x_2) & \cdots & \phi_{M-1}(x_2)\\
\vdots & \vdots & \ddots & \vdots \\  
\phi_{0}(x_N) & \phi_{1}(x_N) & \phi_{2}(x_N) & \cdots & \phi_{M-1}(x_N)\\
\end{pmatrix}
$$

即

$$
\nabla E_D(\mathbf{w})=(\mathbf{t}^T-\mathbf{w}^T\mathbf{\Phi}^T)\mathbf{\Phi}
$$

令其为0可得

$$
\mathbf{t}^T\mathbf{\Phi}=\mathbf{w}^T\mathbf{\Phi}^T\mathbf{\Phi}
$$

$$
(\mathbf{t}^T\mathbf{\Phi})^T=(\mathbf{w}^T\mathbf{\Phi}^T\mathbf{\Phi})^T
$$

$$
\mathbf{\Phi}^T\mathbf{t}=\mathbf{\Phi}^T\mathbf{\Phi}\mathbf{w}
$$

$$
\mathbf{w}_{ML}=(\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}
$$

# 逻辑回归



# 吉布斯分布（p269-p270）

$$p(x)=\frac{1}{Z}e^{-\frac{1}{T}U(x)}$$

- 能量函数：
$$U(x)=\Sigma_{c \in \mathbf{C}} V_c(\mathbf{x})$$
- 配分函数：
$$Z=\Sigma_{x}e^{-\frac{1}{T}U(x)}$$
- 温度：
$$T$$

- 定义在团c上的吉布斯势函数:
$$V_c(\mathbf{x})$$

任意单位点s是一个团；元素数目大于1的子集
$$c \in \mathbf{C}$$
，若C中任意两个不同位点都相邻，则C是一个团。所有团的集合用
$$\mathbf{C}$$
表示

## 吉布斯与马尔可夫的等价
参考
[The Hammersley-Clifford
Theorem and its Impact on
Modern Statistics
Helge Langseth](https://originalstatic.aminer.cn/misc/billboard/aml/Hammersley-Clifford\%20Theorem.pdf)

**Hammersley-Clifford’s theorem**的结论为

一个无向图模型的概率可以表示为定义在图上所有最大团上的势函数的乘积

The following are equivalent (given the positivity
condition):

- Local Markov property : 
$$ p(x_i | x \backslash \{x_i\}) = p(xi | N (xi)) $$

- Factorization property : The probability factorizes according
to the cliques of the graph

- Global Markov property : 
$$p(x_A | x_B , x_S ) = p(x_A | x_S )$$
whenever xA and xB are separated by xS in G

具体证明可以看原文或者[yizt-Hammersley-Clifford定理证明](https://www.jianshu.com/p/dd27249b8c4a)

# 隐马尔可夫模型三个基本问题

HMM由五个要素决定：状态空间S，输出集Y，状态转移矩阵P，输出概率矩阵Q，初始状态分布π

- 模型结构：S、Y
- 模型参数：
$$\lambda = (P,Q,\pi)$$

## 问题一 评估问题：计算给定模型生成某观测输出序列的概率
给定模型参数λ和观测序列y，计算给定模型生成某观测输出序列的概率，通过对比各个模型输出概率，概率最大的模型为识别结果。

解决：加法原理

## 问题二 解码问题
给定模型λ和观测序列y，寻找最有可能产生观察序列的状态序列s，可以最好地解释观测序列。
## 问题三 学习问题
给定多个观测序列y，找出最优模型参数集λ。


# 重要采样（p519-523）

基于假设：无法直接从p(z)中采样，但对于任一给定的z值，可以很容易地计算p(z)的值。仿照拒绝采样的思路，可以利用提议分布;与拒绝采样不同的是：提议分布要使用**最优**的采样函数，**不用一定全部覆盖**原分布函数；并且所有生成的样本都会被保留。

如果直接从p(z)中采样，如果有服从p(z)的L个独立样本，就可以从

$$
E(f)=\int f(\mathbf{z})p(\mathbf{z})d\mathbf{z}
$$

过渡到

$$
f\approx\frac{1}{L}  \sum_{l=1}^L f(\mathbf{z}^{(l)})
$$

因此我们可以从q(z)中采样，乘一个系数，将q(z)当作p(z)的作用，从

$$
E(f)=\int f(\mathbf{z})\frac{p(\mathbf{z})}{q(\mathbf{z})} q(\mathbf{z})d\mathbf{z}
$$

过渡到

$$
\hat f\approx\frac{1}{L}  \sum_{l=1}^L \frac{p(\mathbf{z}^{(l)})}{q(\mathbf{z}^{(l)})}f(\mathbf{z}^{(l)})
$$

乘上的这个系数是**重要性权重**

$$
r_l=\frac{p(\mathbf{z}^{(l)})}{q(\mathbf{z}^{(l)})}
$$

此外采样得到的需要归一化，因此引入

$$
p(\mathbf{z})= \frac{1}{Z_p} \tilde p(\mathbf{z})
$$

$$
q(\mathbf{z})= \frac{1}{Z_q} \tilde q(\mathbf{z})
$$

这样期望就变成了

$$
E(f)=\int f(\mathbf{z})p(\mathbf{z})d\mathbf{z}
=\int f(\mathbf{z})\frac{p(\mathbf{z})}{q(\mathbf{z})} q(\mathbf{z})d\mathbf{z}
=\frac{Z_q}{Z_p} \int f(\mathbf{z})\frac{\tilde p(\mathbf{z})}{ \tilde q(\mathbf{z})}  q(\mathbf{z})d\mathbf{z}
$$

$$
\approx\frac{Z_q}{Z_p} \frac{1}{L}  \sum_{l=1}^L \frac{\tilde p(\mathbf{z}^{(l)})}{\tilde q(\mathbf{z}^{(l)})}f(\mathbf{z}^{(l)})
=\frac{Z_q}{Z_p} \frac{1}{L}  \sum_{l=1}^L \tilde r_l f(\mathbf{z}^{(l)})
=\frac{Z_q}{\int \tilde p(\mathbf{z})d\mathbf{z}} \frac{1}{L}  \sum_{l=1}^L \tilde r_l f(\mathbf{z}^{(l)})
$$

$$
=\frac{Z_q}{\int \frac {\tilde p(\mathbf{z})}{\frac{\tilde q(\mathbf{z})}{Z_q}} q(\mathbf{z}) d\mathbf{z}} \frac{1}{L}  \sum_{l=1}^L \tilde r_l f(\mathbf{z}^{(l)})
=\frac{\frac{1}{L}  \sum_{l=1}^L \tilde r_l f(\mathbf{z}^{(l)})}{\int \frac {\tilde p(\mathbf{z})}{\tilde q(\mathbf{z})} q(\mathbf{z}) d\mathbf{z}} 
=\frac{\frac{1}{L}  \sum_{l=1}^L \tilde r_l f(\mathbf{z}^{(l)})}{\frac{1}{L}  \sum_{l=1}^L \tilde r_l } 
$$

$$
=\frac{  \sum_{l=1}^L \tilde r_l f(\mathbf{z}^{(l)})}{  \sum_{l=1}^L \tilde r_l } 
=\sum_{l=1}^L\frac{   \tilde r_l }{  \sum_{m=1}^M \tilde r_m } f(\mathbf{z}^{(l)})
=\sum_{l=1}^L\frac{   \frac{\tilde p(\mathbf{z}^{(l)})}{\tilde q(\mathbf{z}^{(l)})} }{  \sum_{m=1}^M \frac{\tilde p(\mathbf{z}^{(m)})}{\tilde q(\mathbf{z}^{(m)})} } f(\mathbf{z}^{(l)})
=\sum_{l=1}^Lw_l f(\mathbf{z}^{(l)})
$$

# Metropolis-Hasting方法（p525-526）

在基本Metropolis算法中，假设提议分布（转移核函数）是**对称**的，但Metropolis-Hastings算法无此限制

**算法流程**：

初始化：最大迭代次数T，需要样本数目n，初始化样本 $$\mathbf{z^{(0)}} \sim q(\mathbf{z})$$

循环迭代：

- 从提议分布q中采样得到样本值 $$ \mathbf{z}^* $$

- 从均匀分布中采样$$u \sim (0,1)$$

- 如果$$A(\mathbf{z}^* , \mathbf{z}^{(\tau)})= min{\left( 1,\frac{\tilde p(\mathbf{z}^*) q(\mathbf{z}^{(\tau)}|\mathbf{z}^* )}{\tilde p(\mathbf{z}^{(\tau)}) q(\mathbf{z}^*| \mathbf{z}^{(\tau)})}\right) }>u$$，
则接受$$\mathbf{z}^*$$，即$$\mathbf{z}^{(\tau+1)}=\mathbf{z}^*$$；
否则$$\mathbf{z}^*$$被舍弃，且$$\mathbf{z}^{(\tau+1)}=\mathbf{z}^{(\tau)}$$；

- 如果未达到最大迭代次数，循环上述过程；否则停止

输出：根据需要截取尾部n个样本

如果只是基本Metropolis算法，由于q是对称的，所以$$A(\mathbf{z}^* , \mathbf{z}^{(\tau)})=min{ \left( 1,\frac{\tilde p(\mathbf{z}^*) }{\tilde p(\mathbf{z}^{(\tau)}) }\right) }$$

此处参考了[Persist_Zhang](https://blog.csdn.net/weixin_39753819/article/details/136620415)的博客内容
```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 我们想要采样的目标分布p，以双峰高斯混合分布为例
def p(x):
    return 0.3 * stats.norm.pdf(x, loc=2, scale=0.5) + 0.8 * stats.norm.pdf(x, loc=6, scale=0.5)

# 转移核函数,即提议分布q，以非对称的对称随机游走为例
def q(x):
    if np.random.rand() < 0.5:
        return stats.norm.rvs(loc=x + 0.5, scale=2)  # 向右移动
    else:
        return stats.norm.rvs(loc=x - 0.5, scale=3)  # 向左移动

# Metropolis-Hastings算法
def metropolis_hastings(target_dist, trans_kernel, n_samples):
    samples = []
    # 初始状态
    current_state = 0 
    for _ in range(n_samples):
        # 从条件概率分布q中采样得到新的样本值
        candidate_state = trans_kernel(current_state)  
        # 计算接受概率
        acceptance_prob = min(1, target_dist(candidate_state) / target_dist(current_state))  
        # 从均匀分布中采样，决定是否接受候选状态
        if np.random.rand() < acceptance_prob:
            # 若接受，则转移采样点
            current_state = candidate_state
        samples.append(current_state)
    return samples

# 采样过程
samples = metropolis_hastings(p, q, n_samples=5000)

# 绘制样本分布图像
plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 1000)
plt.plot(x, p(x), label='p Distribution', color='blue')
plt.hist(samples, bins=50, density=True, label='M-H Distribution', color='skyblue', alpha=0.7)
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

```

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/Metropolis-Hastings.png?raw=true" width="100%">