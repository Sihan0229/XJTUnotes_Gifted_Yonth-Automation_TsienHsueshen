建议在[个人博客](https://sihan0229.github.io/2024/06/04/XJTU-ELEC427304-Big-Data-Notes.html)中查看

---
layout: post  
title: "XJTU-ELEC427304 Big Data Notes"  
date: 2024-06-04 19:10 +0800  
last_modified_at: 2024-06-19 19:18 +0800  
tags: [Course Note]  
math: true  r
toc: true 
excerpt: "Course notes of XJTU-ELEC427304 (TBC)."
---
<style>
        h1 { font: 26pt Times !important; }
        h2 { font: 20pt Times !important; }
        h3 { font: 16pt Times !important; }
</style>

| 评分项目 | 详细描述 | 权重 |
| --- | --- | --- |
| 闭卷考试 | 由每人出20分左右的题目组成 | 40-45% |
| 课堂测试|选择题 | 5-10% |
| 个人大作业 | 采用了什么模型<br>选择了哪些特征作为input，为什么<br>数据预处理<br>output是什么<br>对输出的分析<br>和别人的比较<br>未来展望改进方向<br>training的过程<br>test的过程<br>超参数是如何调整的 | 50% |


# Introduction

**PAC - probably Approximately Correct learning model**

$$
P(|f(x)-y|\le \epsilon)\ge 1-\delta
$$

**Data Types**

+ Continuous, Binary
+ Discrete, String
+ Symbolic

**Big Data** is high-volume, high-velocity, high-variety, demanding cost-effective, innovative forms of imformation processing, whose size is beyongd the ability of typical database software tools to capture, store, manage, and analyze.

**Data Mining** is the process of discovering patterns in large data set, including intersection of ML, statistic and database systems.

习题1：大数据的特点包括类型多、对处理实时性要求高、容量大

习题2：理想的数据挖掘算法得到的结果应该是：Useful, Hidden, Interesting

**Source & Materials**
[KDnuggets](https://www.kdnuggets.com/), [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php)

**Framework of ML**

+ Step 1: function with unknown

$$
y=f_\theta(x)
$$

+ Step 2: define loss from training data

$$
L(\theta)
$$

+ Step 3: optimization

$$
\theta^* = \arg \min_{\theta} \mathcal{L}
$$

**DM Techniques - Classification**
Decision Trees, K-Nearest Neighbours, Neural Networks, Support Vector Machines

# Evaluation method of model

## Data Segmentation

**hold-out 留出法** 保持数据分布一致性，多次重复随机划分，测试集不能太大、不能太小

**cross-validation 交叉验证法** k折交叉验证

**boostrapping 自助法** 有放回/可重复采样，数据分布有所改变，训练集与原样本集同规模
（包外估计 out-of-bag estimation）

## Model performance

**Error rate 错误率**

$$
E(𝑓;𝐷)=\frac{1}{m}\sum_{i=1}^{m}I(f(𝒙_i)≠𝑦)
$$

**Accuracy 精度**

$$
Acc(𝑓;𝐷)=\frac{1}{m}\sum_{i=1}^{m}I(f(𝒙_i)=𝑦)=1-E(𝑓;𝐷)
$$

**Precision 查准率**

$$
P=\frac{TP}{TP+FP}
$$

**Recall 查全率**

$$
R=\frac{TP}{TP+FN}
$$

**F1**

$$
\frac{1}{F_1}=\frac{1}{2}(\frac{1}{R}+\frac{1}{P})
$$

**Fβ**

$$
\frac{1}{F_\beta}=\frac{1}{1+\beta^2}(\frac{\beta^2}{R}+\frac{1}{P})
$$

**Confusion Matrix 混淆矩阵**

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/ConfusionMatrix.png?raw=true" width="100%">

**P-R曲线**

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/pr.png?raw=true" width="60%">

**BEP 平衡点 Break-Even Point** 查准率 = 查全率的取值

**ROC 受试者工作特征曲线**

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/roc.png?raw=true" width="100%">

**AUC**：ROC曲线下面积

How to Construct an ROC curve

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/rocCurve.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/rocCurve2.png?raw=true" width="100%">

**Cost Matrix**
Cost-sensitive error rate and cost curve

C(i,j)：Cost of misclassifying class j example as class i

**Cost VS Accuracy**

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/cost_vs_accuracy.png?raw=true" width="100%">

**Lift Analysis**

[[模型解釋策略]Lift Chart, Permutation Importance, LIME](https://yulongtsai.medium.com/lift-chart-permutation-importance-lime-c22be8bdaf48)

例题：假设目标客户占人群的5%，现根据用户模型进行打分排序，
取1000名潜在客户中排名前10%的客户，发现其中包含25名目
标客户，问此模型在10%处的提升度是多少?

答案：5

$$
Lift = \frac {P(A \mid B)}{P(A)}=\frac {\frac{25}{1000 \times 10\% }}{5\%}=\frac{25\%}{5\%}=5
$$

例题：我们通常将数据集划分为训练集，验证集和测试集进行模型的训
练，参数的验证需要在**验证集**上进行，参数确定后**需要**重新训练模型。

例题：当西瓜收购公司去瓜摊收购西瓜时既希望把好瓜都收走又保证收
到的瓜中坏瓜尽可能的少，请问他应该考虑什么评价指标？

正确：**F1调和平均**与**BEP**

例题：假设我们已经建立好了一个二分类模型,输出是0或1,初始阈值设
置为05 超过0.5概率估计就判别为1,否则就判别为0;如果我们现
在用另一个大于0.5的阈值，一般来说，下列说法正确的是：**查准率会上升或不变，查全率会下降或不变**

**DM Techniques - Classification**: K-Means, Sequential Leader, Affinity Propagation

**DM Techniques – Association Rule**

**DM Techniques – Regression**


**Typical lssues** : 缺少属性值Missing Attribute Values, 不同的编码/命名方案Different Coding/Naming Schemes, 不可行的值Infeasible Values, 不一致的数据InconsistentData, 异常值Outliers

**Data Quality** : Accuracy （准确性）, Completeness（完整性）, Consistency （一致性）, Interpretability（可解释性）, Credibility（可信性）, Timeliness（时效性）

**数据集成** : 组合来自不同来源的数据。

**数据缩减** : 特征选择、抽样

**Privacy**

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/privacy.png?raw=true" width="100%">

**No Free Lunch**

Why bother so many different algorithms?

+ No algorithm is always superior to others.
+ No parameter settingis optimal over all problems.

Look for the best match between problem and algorithm.

+ Experience
+ Trial and Error

Factors to consider:
+ Applicability
+ Computational Complexity
+ Interpretability

Always start with simple ones.

# Data Preprocessing

## Data Cleaning 数据清洗
填充缺失值、更正不一致的数据、识别异常值和噪声数据。
### 缺失值

数据缺失类型分为三种：完全随机缺失、随机缺失、非随机缺失。
参考[数据缺失类型](https://blog.csdn.net/jwtning/article/details/116125819)

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/missing.png?raw=true" width="100%">

如何处理缺失数据？
+ 忽略：删除有缺失值的样本/属性，最简单、最直接的方法，低缺失率时效果很好
+ 手动填写缺失值：重新收集数据或领域知识，繁琐/不可行
+ 自动填写缺失值：全局常数/平均值或中位数/最可能的值

例题：学生小明在调查问卷中没有回答下述问题:“你去年的工资收入和前年相比是否有所增加?”对这种情况最恰当的描述是: **N/A**而不是“数据未提供”

以下参考[劉智皓 (Chih-Hao Liu) 機器學習_學習筆記系列(96)：區域性異常因子(Local Outlier Factor)](https://tomohiroliu22.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98%E7%B3%BB%E5%88%97-96-%E5%8D%80%E5%9F%9F%E6%80%A7%E7%95%B0%E5%B8%B8%E5%9B%A0%E5%AD%90-local-outlier-factor-a141c2450d4a)

**Outliers离群点** : Outliers≠Anomaly

**Local Outliner Factor**
关于LOF算法，它是基于空间密度来寻找异常值的，这里我们定义**可达距离reachability distance** = max(B点到离B第k近的点的距离, A和B的距离)

$$
reachability_{-}distance_{k}(A,B)=m a x\Big[k_{-}distance(B),distance(A,B)\Big]
$$ 
假设有两个点A和B，`k_distance(B)`代表的就是B点到离B第k近的点的距离，`distance(A,B)`则就是A和B的距离。所以这里的意思是：如果点和点之间相距够近，就将他们一视同仁，视为密度较高的区域。

而接下來我們會計算**local reachability density: (平均距离)**

$$
IRD_{k}(A)=\frac{1}{\left( \frac{\sum_{B\in{\cal N}_{k}(A)}reachability_{-}distance_{k}(A,B)}{|{N}_{k}(A)|}\right)}
$$

其中N_k為A點的neighbor。所以這個式子代表的就是，我們A點neighbor其reachability distance**平均**的**倒數**，所以我們可以說，**如果IRD很大，代表以A點為中心的區域很密集**，反之則是很疏鬆。
而當我們求得了IRD之後，我們最後就會計算

**Local Outlier Factor**:

$$L O F_{k}(A)=\frac{\sum_{B\in N_{k}(A)}I R D_{k}(B)/I R D_{k}(A)}{\left|N_{k}(A)\right|}=\frac{1}{I R D_{k}(A)}\frac{\sum_{B\in N_{k}(A)}I R D_{k}(B)}{\left|N_{k}(A)\right|}$$



我們可以看到LOF，他做的事情就是計算A所有**neighbor的**IRD值並且將其平均除以IRD(A)。而LOF在意義上來說，**如果接近1代表，A和其Neighbor的空間密度都非常接近，如果小於1非常多，代表A的密度大於他的neighbor，也就是密度較高的區域，若大於1非常多，則代表A的密度小於他的neighbor。**

例题：关于离群点的判定需要考虑**相对距离因素**，主要看其`与近邻的平均距离`与主要看其与`近邻的最大距离`均为错误

**名义数据 (Nominal data)** 与 **序数数据 (Ordinal data)** 对比：
+ 名义数据：国家、颜色等等没有顺序
+ 序数数据：ABCD、非常不、稍不、中等、稍优、非常优

**相似度与不相似度** 
+ 相似度: 两个数据对象相似程度的数值测量,对象越相似，相似度越高,通常在 [0,1] 范围内
+ 相异度: 两个数据对象差异程度的数值测量,对象越相似，相似度越低,最小不相似度通常为 0,上限各不相同

简单属性的相似性/不相似性

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/simidissimi.png?raw=true" width="100%">

**distance**
	
+ Euclidean Distance
$$dist={\sum^{n}_{k-1}}(p_{k}-q_{k})^{2}$$
+ Minkowski Distance
$$dist=(\sum_{k=1}^{n}|p_{k}-q_{k}|^r)^{1/r}$$

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/Minkowski.png?raw=true" width="100%">

+ Mahalanobis Distance

$$mahalanobis(p,q)=\sqrt{(p-q)\Sigma^{-1}\left(p-q\right)^{T}}$$

$$
\Sigma_{j,k}=\frac{1}{n-1} \sum_{i=1}^{n} (X_{ij}-\overline{X}_{j})(X_{ik}-\overline{X}_{k})
$$

**Duplicate Data重复数据处理方法**

Large Data : create keys -> sort -> merge

## Data Transformation 数据转换
现在我们有了一个无错误的数据集，还需要标准化 standardized。类型转换、规范化Normalization、采样、 聚合Aggregation

例如：选取变换$$\phi(x)$$将非线性变为线性、使用独热编码标记类别(不能因为编码而产生新的参数影响，如因为编码1 2 3而导致距离不相等)等

**采样**：什么是采样？

有效抽样的关键原则如下：
+ 如果样本具有代表性，则使用样本的效果几乎与使用整个数据集一样好 
+ 如果样本具有与原始数据集大致相同的属性（感兴趣的），则该样本具有代表性

习题：在大数据分析中，利用采样技术可以:
+ 降低获取数据的成本（错误）
+ 减少需要处理的数据量
+ 有助于处理不平衡数据
+ 提高数据的稳定性



**Imbalanced Datasets不平衡的数据集**

**G-means**

$${G-m e a n}=(A c c^{+}\times A c c^{-})^{1/2}$$

$$w h e r e\ A c c^{+}={\frac{T P}{T P+F N}};\ \ \ A c c^{-}={\frac{T N}{T N+F P}}$$

**F-measure**

$$F-m e a s u r e={\frac{2\times P r e c i s i o n\times R e c a l l}{P r e c i s i o n+R e c a l l}}$$

$$w h e r e\;\;\;P r e c i s i o n=\frac{T P}{T P+F P};\;\;\;\;{R e c a l l}=\frac{T P}{T P+F N}=A c c^{+}$$

**Over-Sampling**

可以参考[机器学习之类别不平衡问题 (3) —— 采样方法
](https://www.cnblogs.com/massquantity/p/9382710.html),这里面的Border-line SMOTE个人认为比较符合ppt上的下一个要点**Boundary Sampling**

不平衡的数据集要采样、扩增、调整loss function

**Normalization**

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/Normalization.png?raw=true" width="100%">


## Data Description

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/description.png?raw=true" width="80%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/description2.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/description3.png?raw=true" width="100%">

## Feature Selection

属性和特征的选择

**Class Distributions** ：不平衡的class样本数量/分布

**Entropy 熵**与**Information Gain 信息增益**

信息熵
$$\operatorname{Ent}(D)=-\sum_{k=1}^{ \|y\|}p_{k}\log_{2}p_{k}
$$

$$\operatorname{Ent}(D)=-\Sigma_{k=1}^{ \|y\|}p_{k}\log_{2}p_{k}
$$

信息熵值越小，D的纯度越高

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/gain.png?raw=true" width="100%">

$$
{\mathrm{Gain}}(D,a)={\mathrm{Ent}}(D)-\sum_{v=1}^{V}{\frac{|D^{v}|}{|D|}}{\mathrm{Ent}}(D^{v})
$$

信息增益越大，说明使用属性a划分的纯度提升越大，在决策树中被选为划分属性

## Feature Extraction

+ PCA会投影到保留信息最好的方向
+ LDA关注哪个方向可以更好地保持原始的分类信息（保留？类别信息）在尽可能保留类别区分信息的同时进行降维。

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/pcdlda.png?raw=true" width="100%">

### 主成分分析PCA 数据规约

使所有样本的投影尽可能分开(如图中红线所示)，则需最大化投影点的方差，将原始数据投影到具有最大特征值的 S 的特征向量

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/pca.png?raw=true" width="60%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/pca_label.png?raw=true" width="60%">

### 线性判别分析LDA 降维

设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离

**Fisher Criterion**

$$J={\frac{\left|\mu_{1}-\mu_{2}\right|^{2}}{S_{1}^{2}+S_{2}^{2}}}={\frac{w^{T}{S}_{B}w}{w^{T}{S}_{w}w}}$$

详细解释：
要让类间距离尽可能大，类内距离尽量小
$$J=\frac {\Vert w^t\mu _0 -w^Tmu _1\Vert^2_2}{w^T \Sigma_0 w+w^T \Sigma_1w}$$
类内散度矩阵
$$\mathrm{S_w} =\Sigma_0+\Sigma_1 =  \sum_{\alpha \in X_0} (x - \mu_0)(x - \mu_0)^T + \sum_{\alpha \in X_1} (x - \mu_1)(x - \mu_1)^T$$
类间散度矩阵
$$S_{b}=\left(\mu_{0}-\mu_{1}\right)\left(\mu_{0}-\mu_{1}\right)^{\mathrm{T}}$$

Measure of Separability
+ LDA produces at most C-1 projections
+ SB类间散度矩阵 is a matrix with rank C-1 or less.
+ SW类内散度矩阵 may be singular.
+ LDA does not work well when...?

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/lda.png?raw=true" width="60%">

(because the denominator of fisher critertion = 0)

例题：PCA中包含**去均值**、**坐标变换**、**矩阵特征值分解**，不包含属性值标准化

例题：样本个数小于数据维数时，LDA不能正常工作的原因是**类内散布矩阵不满秩**

例题：当类中心重合时，LDA不能正常工作的原因是：Fisher准则函数恒等于0

（
类间散度矩阵
$$
S_{b}=\left(\mu_{0}-\mu_{1}\right)\left(\mu_{0}-\mu_{1}\right)^{\mathrm{T}}
$$
）

# Naïve Bayes Classifier

$$
P(A \mid B)=\frac{P(B\mid A) P(A)}{P(B)}
$$

**Naïve Bayes Classifier**

$$\omega_{MAP}=\arg\max_{\omega_i \in \omega} P\bigl(\omega_{i}\mid a_{1},a_{2},...,a_{n}\bigr)$$

$$\omega_{MAP}=\arg\max_{\omega_i \in \omega} \frac {P\bigl (a_{1},a_{2},...,a_{n}\mid\omega_{i}\bigr)P\bigl (\omega _{i} \bigr)}{P\bigl (a_{1},a_{2},...,a_{n}\bigr)}$$

$$\omega_{MAP}=\arg\max_{\omega_i \in \omega} {P\bigl (a_{1},a_{2},...,a_{n}\mid\omega_{i}\bigr)P\bigl (\omega _{i} \bigr)}$$

由于条件独立性

$$\omega_{M d P}=\arg\max_{\omega_i \in \omega} P{\big(}\omega_{i})\prod_{j}P{\big(}a_{j}\mid\omega_{i}{\big)}$$



## Conditional Independence   

$$P(A,B\mid G)=P(A\mid G)P(B\mid G)\;\longleftrightarrow P(A\mid G,B)=P(A\mid G)$$

**Independent** ≠ **Uncorrelated**

e.g. When $$X \in [-1, 1]$$, then $$Y = x^2$$.

$$Cov(X, Y) = 0$$
indicating that$$X$$and $$Y$$ are **uncorrelated**, even though$$ Y$$is **completely determined** by $$X$$.

**拉普拉斯平滑**

参考[拉普拉斯平滑（Laplacian smoothing）](https://www.cnblogs.com/BlairGrowing/p/15803361.html)

零概率问题：在计算事件的概率时，如果某个事件在训练集中没有出现过，会导致该事件的概率结果是0。这是不合理的，不能因为一个事件没有观察到，就被认为该事件一定不可能发生（即该事件的概率为0）

原本极大似然估计公式为

$$
\varphi_{j}=\frac{\sum_{i=1}^{m}I\{z^{(i)}=j\}}{m}$$

拉普拉斯平滑后(在分母上加上随机变量取值范围的大小k, 在分子加1)

$$
\varphi_{j}=\frac{\sum_{i=1}^{m}I\{z^{(i)}=j\}+1}{m+k}$$

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/Laplacian smoothing.png?raw=true" width="100%">


例题：以下关于拉普拉斯平滑说法正确的是:

+ 防止计算条件概率时分母为零
+ 防止计算条件概率时分子为零（正确）
+ 用于解决训练集中的噪声
+ 用于解决训练集中的异常值

# Decision Tree Model
## ID3

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/Dataset.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/Attribute Selection.png?raw=true" width="100%">


**奥卡姆剃刀**：倾向于选择更简单的模型

**过拟合over fitting**：泛化误差小，测试误差大（额外划出Validation Set来判定是否过拟合）

定义为：给定一个假设空间 H，如果存在某个备择假设 h' ∈ H，比如 h 在训练样本上的误差比 h' 小，但是 h' 在整个实例分布上的误差比 h 小，则称假设 h ∈ H 与训练数据过度拟合。

解决：早停，Loss加入正则项，利用新模型

决策树中解决过拟合的方法：**剪枝**

## ID3 Framework

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/ID3 Framework.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/ID3Framework.png?raw=true" width="100%">

+ 叶节点为纯节点，stop.
+ 叶节点为空节点，如何决定该节点的分类？根据父节点的比例
+ 属性用完了也无法分类：树无法生长，根据占优比例决定类别。

### 剪枝方法

如何判断剪枝泛化后性能是否提升？ 留出法

+ 预剪枝：生成决策树时，对每个节点在划分前先进行估计，若不能带来泛化性能的提升，则停止划分。（主要用到的标准是**验证集/测试集**的精度提升情况）

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/precut.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/precut2.png?raw=true" width="100%">

|保留分支数 | 训练时间 | 泛化性能 | 欠拟合风险 |
| --- | --- | --- | --- |
|少| 小 |强| 大|
很多分支未展开|||学到的特点不足以对样本进行正确分类|

+ 后剪枝:生成完整的决策树（也就是说，分类决策是由**训练集**给出的），自底向上向叶节点进行考察，若该节点对应的子树替换成叶节点（叶节点正负取值仍然由**训练集**样本占比决定）能带来泛化性能（在**验证集**上分类的准确性）的提升，则替换成叶节点。

**Entropy Bias**
熵作为决策标准会偏向于具有更多类值的属性，例如• 考虑属性“出生日期”，将训练数据分成非常小的子集，信息增益非常高，对看不见的实例的目标函数预测非常差，这样的属性需要penalized！

### 信息增益率

$$Split Information(S,A)=-\sum_{i=1}^{C}{\frac{\mid S_{i}\mid}{\mid S\mid}}log_{2}\,{\frac{\mid S_{i}\mid}{\mid S\mid}}$$

$$G a i n R a t i o(S,A)=\frac{G a i n(S,A)}{Split Information(S,A)}$$

### 基尼指数 Gini index

当前样本集合 D 中第 k 类样本所占的比例为 Pk ,数据集 D 的纯度可用基尼值来度量

$$Gini(D)=1-\sum_{k=1}^{|y|}p_{k}^{2}$$

Gini(D) 反映了从数据集 D 中随机抽取两个样本，其类别标记不一致的概率.Gini(D)越小，说明纯度越高。
属性 α 的基尼指数定义为(D指的是划分处的数据集)

$${\mathrm{Gini\_ index}}(D,a)=\sum_{v=1}^{V}{\frac{|D^{v}|}{|D|}}{\mathrm{Gini}}(D^{v})$$

基尼指数最小的是最优划分属性

$$a_{*}=\arg\min_{a\in A}{\mathrm{Gini\_ index}}(D,a)$$

### 分类错误

$$
\text{Error} = 1 - \max(P(C1), P(C2), \ldots, P(Cn))
$$

+ 最大值：当记录在所有类中均匀分布时，错误最大。这意味着节点包含的信息最少。
+ 最小值：当所有记录都属于同一类时，错误最小。这意味着节点包含的信息最多。

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/classification_error.png?raw=true" width="100%">

熵、Gini与分类error的比较

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/compare.png?raw=true" width="70%">

### 连续值处理:设置threshold

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/continous_id3.png?raw=true" width="100%">

### 缺失值处理

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/na.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/na2.png?raw=true" width="100%">

**解决问题1：如何选择属性**

**解决问题2： 如何划分样本集合**

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/lost1.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/lost2.png?raw=true" width="100%">

### 多变量决策树

实现这样的"斜划分"甚至更复杂划分的决策树，采用属性的组合

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/multi_tree.png?raw=true" width="100%">


# Convolutional Neural Networks

**Optimization**

$$w^{1}\leftarrow w^{0}-\eta\frac{\partial L}{\partial w}|_{w=w^{0},b=b^{0}}$$

$$b^{1}\leftarrow b^{0}-\eta\frac{\partial L}{\partial b}|_{w=w^{0},b=b^{0}}$$

**Small Batch v.s. Large Batch**

Small Batch具有更好的性能

## Step1: function with unknown
**Sigmoid Function**

$$y=c{\frac{1}{1+e^{-(b+w x_{1})}}}=c sigmoid (b+wx_1)$$

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/sigmoid_change.png?raw=true" width="100%">

j为特征数量，i为sigmoid数量
$$y=b+\sum_{i} c_{i} sigmoid \big( b_i + \sum_{j}w_{i j} x_{j}\Big)$$

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/sigmoid_net.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/sigmoid_mat.png?raw=true" width="60%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/sigmoid_net2.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/sigmoid_a.png?raw=true" width="100%">

## Step2: define loss from training data

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/loss_def.png?raw=true" width="100%">

## Step 3: optimization

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/update.png?raw=true" width="100%">

例题：

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/batch_eg.png?raw=true" width="60%">

Sigmoid to **ReLU** 激励函数变化，哪一种更好？

$$y=b+\sum_{2i}c_{i}\max\biggl({0},b_{i}+\sum_{j}w_{i j}x_{j}\biggr)$$

例题：以下关于感知机说法正确的是:
+ 在batchlearning模式下，权重调整出现在学习每个样本之后
+ 只要参数设置得当，感知机理论上可以解决各种分类
问题
+ 感知机的训练过程可以看成是在误差空间进行梯度下降(正确)
+ 感知机的激励函数必须采用门限函数

例题：采用sigmoid函数作为激励函数的主要原因是:
+ 有固定的输出上下界(正确)
+ 处处可导(正确)
+ 计算复杂度较低(错误)
+ 导数存在解析解(正确)

例题：以下关于感知机说法正确的是:
+ 多层感知机比感知机只多了一个隐含层
+ 感知机只能形成线性判决平面，无法解决异或问题（正确）
+ 多层感知机可以有多个隐含层，但是只能有一个输出单元
+ 隐含层神经元的个数应当小于输入层神经元的个数

多层感知机解决线性不可分问题的原理是:
+ 分而治之，对原始问题空间进行划分
+ 将原始问题向更高维空间映射
+ 在输出层和隐含层之间形成非线性的分界面
+ 将原始问题在隐含层映射成线性可分问题(正确)

## Backpropagation 反向传播

### 前向传播

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/forward.png?raw=true" width="100%">

### 后向传播

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/backward1.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/backward2.png?raw=true" width="100%">

**Case 1. Output Layer**

$$\frac{\partial C}{\partial z^{\prime}}=\frac{\partial y_{1}}{\partial z^{\prime}}\frac{\partial C}{\partial y_{1}}\;\;\;\;\;\frac{\partial C}{\partial z^{\prime\prime}}=\frac{\partial y_{2}}{\partial z^{\prime\prime}}\frac{\partial C}{\partial y_{2}}$$

**Case 2. Not Output Layer**：递归计算

前后向传播总结

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/forback.png?raw=true" width="100%">

例题：在误差逆传播算法中，输出层神经元权重的调整机制和感知机的学习规则相比:
+ 考虑到线性不可分问题，学习规则更为复杂
+ 一模一样，等价于多个感知机
+ 遵循相同的原理，激励函数可能有所不同（正确）
+ 所有输出层神经元的权重需要同步调整

例题：在权重更新公式中引入冲量的主要目的是:
+ 提高算法的收敛精度
+ 提高算法的稳健性
+ 提高算法的全局优化能力
+ 有助于摆脱误差平缓区域 （正确）

例题：为了克服学习空间中存在的局部最优点应当:
+ 尝试从不同的初始点开始训练（正确）
+ 将权重初始化为接近于0的值
+ 采用较小的学习率
+ 增加隐含层神经元个数

## Softmax

Classification：one hot 编码

$$y_{i}^{\prime}={\frac{e x p(y_{i})}{\sum_{j}e x p(y_{j})}}$$

满足

$$1>y_{i}^{\prime}>0$$

$$\sum_{i} y_{i}^{\prime}=1$$

Loss of Classification:最小化**交叉熵**等同于最大化似然。

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/classloss.png?raw=true" width="100%">

MSE与交叉熵损失对比

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/mse_cross.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/mse_cross2.png?raw=true" width="100%">

## Fully connected layers

参数量

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/bd_fc_num.png?raw=true" width="100%">

## Convolution arithmetic 卷积

+ 局部感知、参数共享、激励求和
+ 有些图案比整个图像小得多。
+ 相同的图案出现在不同的区域

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/story2.png?raw=true" width="100%">

## Pooling arithmetic 池化

对像素进行Subsampling不会改变对象

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/wholecnn.png?raw=true" width="100%">

CNN 对缩放和旋转不具有不变性（我们需要数据增强）

# Classifier

## Linear Classifier

$$
{{f(x,w,b)=s i g n(g(x))}}=s i g n(w\cdot x+b)
$$

$$
M={\frac{\left|g(x)\right|}{\left\|w\right\|}}={\frac{\left|w\cdot x+b\right|}{\left\|w\right\|}}
$$

## SVM 支持向量机

只有支持向量在起作用
$$
\alpha_{i}\ge0
$$
的点是支持向量

正确分类样本：
$$ y_i(w\cdot x_i+b)-1\ge 0 $$

Margins宽度：
$$
M={\frac{2}{\lVert w\rVert}}
$$

而 
$$\max M={\frac{\left|g(x)\right|}{\left\|w\right\|}} \rightarrow\min \frac{1}{2}w^Tw
$$

在以下约束下用拉格朗日乘子法

$$
\begin{cases}
  y_i(w \cdot x_i + b) - 1 \ge 0 \\
  a_i \ge 0 \\
  a_i [y_i (w \cdot x_i + b) - 1] = 0
\end{cases}
$$

$$
L_{D}\equiv\sum_{i}\alpha_{i}-{\frac{1}{2}\alpha ^T H \alpha} 
$$

where 

$$
H_{ij}=x_ix_jy_iy_j 
$$

$$s u b j e c t\;t o:\sum_{i}\alpha_{i}y_{i}=0\;\;\&\;\;\alpha_{i}\ge0
$$

$$
w=\sum_{i=1}^{l}\alpha_{i}y_{i}x_{i}
$$

$$b={\frac{1}{N_{s}}}\sum_{s\in S}(y_{s}-\sum_{m\in S}\alpha_{m}y_{m}x_{m}\cdot x_{s})$$


<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/svm_eg.png?raw=true" width="100%">

### Soft Margin 软间隔

$$
y_{i}(w x_{i}+\partial)-1+\xi_{i}\geq0
$$

$$\Phi(w)=\frac{1}{2}\,w^{t}w+C\sum_{i}\xi_{i}$$

$$\xi_{r}\geq0$$

$${\cal L}_{P} \equiv\frac{1}{2}\Bigl|w\Bigr|^{2}+C\sum_{i=1}^{l}\xi_{i}-\sum_{i=1}^{l}\alpha_{i}[y_{i}(w\cdot x_{i}+b)-1+\xi_{i}]-\sum_{i=1}^{l}\mu_{i}\xi_{i}
$$

$$L_{\scriptscriptstyle D}\equiv\sum_{i}\alpha_{i}-\frac{1}{2}\alpha^{T}H\alpha\quad s.t.0\le\alpha_{i}\le C\quad a n d\ \sum_{i}\alpha_{i}y_{i}=0
$$

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/soft.png?raw=true" width="100%">

### 非线性SVMs （Non-linear SVMs）


**Feature Space**：
向高阶空间进行映射，可以把很多不能用linear SVM解决的问题使用linear SVM解决

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/nonlinearSVMs.png?raw=true" width="100%">

Which kind of $$\varphi(x)$$ can solve this problem?

## Kernel Trick 核方法

$$K(x_{i}x_{j})=\varphi(x_{i})*\varphi(x_{j})$$

每个数据点都通过某种变换
$$\mathcal{\Phi}\colon\,x\rightarrow\mathcal{\varphi}(x)$$
映射到高维空间

$$x_{i}*x_{j}\rightarrow\varphi(x_{i})*\varphi(x_{j})$$

常见核函数

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/kernel.png?raw=true" width="100%">

问题：在SVM当中进行空间映射的主要目的是:
- A 降低计算复杂度 （提高）
- B 提取较为重要的特征
- C 对原始数据进行标准化
- D 提高原始问题的可分性 √

问题：对于SVM，在映射后的高维空间直接进行计算的主要问题是:
- A 模型可解释性差
- B 计算复杂度高 √
- C 容易出现奇异矩阵
- D 容易出现稀疏矩阵

问题：所谓kernel trick，指的是:
- A 利用在原始空间定义的函数替代高维空间的向量内积操作 √
- B 利用在高维空间定义的函数替代原始空间的向量内积操作
- C 核函数的导数具有简单的解析解，简化了运算
- D 核函数具有固定的上下界，可以输出$$(-1,+1)$$区间中的连续值

问题：通过运用核函数，我们可以:
- 提高算法的可解释性
- 生成数量较少的支持向量
- 生成数量较多的支持向量
- 避免高维空间运算，降低算法复杂度（正确）


# 集成学习 

组合模型的不同方法：
+ 平均法
+ 多数投票法
+ 加权多数投票法

## bagging

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/bootstrap.png?raw=true" width="100%">

## Random Forests 随机森林
**主要特征**

生成截然不同的树
+ 使用训练数据的随机引导样本。
+ 对每个节点使用随机变量子集。

变量数量
+ 平方根 (K)
+ K：可用变量总数
+ 可以显著加快树构建过程。

树数量
+ 500 棵或更多

自我测试
+ 使用大约三分之二的原始数据。
+ **大约三分之一的原始数据被遗漏**。
+ 包外 (OOB)
+ 类似于交叉验证

**RF 优势**
所有数据均可用于训练过程。
+ 无需留下一些数据用于测试。
+ 无需进行常规交叉验证。
+ OOB 中的数据用于评估当前树。

整个 RF 的性能
+ 每个数据点都在树的子集上进行测试。
+ 取决于它是否在 OOB 中。

高水平的预测准确性
+ 仅需尝试几个参数。
+ 适用于分类和回归。

抗过度训练（过度拟合）。

无需事先进行特征选择。

## 有放回采样
有多少样本没有被用到但是被以为是可以test的 OOB

好处：可以帮我们构建具有分散性的基础分类器

充分利用所有样本

例题：自助法（Bootstrap）是有放回采样

## 保证基础分类器多样性的方法
+ 算法多样性
+ 训练集（随机又放回采样）
+ 选择不同的属性，决策树中用$$\sqrt{K}$$个属性构造500-5000棵树
+ 超参数


## Stack

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/stack.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/stacking.png?raw=true" width="100%">

例题：Bagging的主要特点有:
+ 各基础分类器并行生成（正确）
+ 各基础分类器权重相同（认为正确）
+ 只需要较少的基础分类器
+ 基于Bootstrap采样生成训练集（正确）


## Boosting

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/Boosting.png?raw=true" width="100%">

+ Bagging 旨在减少方差，而不是偏差。
+ 在 Boosting 中，分类器是按顺序生成的
+ 关注最具信息量的数据点。训练样本是加权的,通过加权投票合并输出。
+ 可以创建任意强的分类器。基础学习者可以任意弱, 只要它们比随机猜测更好！

## Adaboost

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/AdaBoost.png?raw=true" width="100%">

alpha的选择

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/Choice1.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/Choice2.png?raw=true" width="100%">

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/Choice3.png?raw=true" width="100%">

$$
a=\frac{1}{2} \ln(\frac {1-\varepsilon}{\varepsilon})
$$

误差界的计算

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/Error Bounds.png?raw=true" width="100%">

$$r=\sum_{i}D_{i}y_{i}h(x_{i}) \rightarrow \varepsilon = \frac{1-r}{2} \rightarrow a=\frac{1}{2} \ln(\frac {1+r}{1-r})
$$

$$
Z = \sqrt{1-r^2}
$$

例题：在AdaBoost算法中，Z的作用是:
+ 确保在t+1代所有样本权重之和为1
+ 一个用于标准化的变量，可有可无（错误）
+ 可以用来描述算法的训练误差上界
+ 较小的Z值说明当前分类器的效果较好

例题：AdaBoost中基础分类器的权重设置策略存在的问题有:
计算复杂
不能保证是最优解（正确）
需要用户进行手工设置
不能根据测试样本进行自适应调整（正确）

**AdaBoost总结**

优点
+ 简单易行
+ 几乎没有需要调整的参数
+ 训练集上有证明的上限+
+ 不易过拟合
 
缺点
+ 次优的 𝛼 值
+ 最速下降
+ 对噪音敏感

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/dy_w.png?raw=true" width="100%">

## RegionBoost

不同的模型强调不同的区域,模型的权重应该依赖于输入,给定输入，仅调用适当的模型。

问题：以下对RegionBoost算法描述正确的是:
+ 基础分类器权重根据当前输入样本计算得出
+ 每个基础分类器需要一个额外的可信度模型
+ 每个基础分类器的权重针对不同输入样本有所区别
+ 可信度模型用于估计基础分类器对特定输入的准确度

问题：RegionBoost与AdaBoost相比:
+ 训练误差通常降低较慢（错误，初期下降较快）
+ 训练误差能够趋近于0（错误，无法达到0）
+ 测试误差可能优于AdaBoost（正确）
+ 有较多的参数需要设置（正确）

主要对比下面两张图，左侧为训练误差，右边为测试误差

<img src="https://github.com/Sihan0229/Sihan0229.github.io/blob/master/assets/ada_region.png?raw=true" width="100%">


