<!-- KaTeX -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.6.0/katex.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.6.0/katex.min.css" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.6.0/contrib/auto-render.min.js"></script>

<!-- highlight.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.4.0/highlight.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.4.0/styles/github.min.css">

# 有监督学习（分类）

## 有监督学习(分类)
在有监督学习中，主要的工作是通过由训练样本组层 $\{(x_n, y_n)\}$的已标记数据(labeled data)推测出隐藏结构。分类的意思是输出值$y$是离散的值。

这里是一个Edward的样例，基于Jupyter notebook的互动版本在[这里](http://nbviewer.jupyter.org/github/blei-lab/edward/blob/master/notebooks/supervised_classification.ipynb)

##　数据
我们使用用形态测量(morphological measurements)不同蟹种类的[蟹类数据集](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/crabs.html).我们感兴趣的是预测一只螃蟹是蓝色的或者是橙色的。

在这里使用了25个蟹类数据的样本点：
` ` ` Python
df = np.loadtxt('data/crabs_train.txt', delimiter=',')
df[df[:, 0] == -1, 0] = 0  # replace -1 label with 0 label

N = 25  # number of data points
D = df.shape[1] - 1  # number of features

subset = np.random.choice(df.shape[0], N, replace=False)
X_train = df[subset, 1:]
y_train = df[subset, 0]

print("数据点个数: {}".format(N))
print("特征个数: {}".format(D))
` ` `
数据点个数：100
特征个数：5

## 模式
高斯过程(Gaussian process)是模拟随机变量对非线性(nonlinear)关系的有强大工具。它定义了一个分布（可能是非线性）函数，它可以应用于表示我们的不确定性围绕真实函数关系。我们现在定义一个用于分类的高斯过程(Rasmussen & Williams, 2006)。

一个分布函数$f:\mathbb{R}^D\to\mathbb{R}$可以由高斯过程表示。
$$
  p(f)
  &=
  \mathcal{GP}(f\mid \mathbf{0}, k(\mathbf{x}, \mathbf{x}^\prime)),
$$

它的平均函数(mean function)是零函数，其协方差函数(covariance function)是描述函数的任意一组输入之间的依赖关系的一些核。

给出的一系列输入－输出对$\{\mathbf{x}_n\in\mathbb{R}^D,y_n\in\mathbb{R}\}$，它们的似然(likehood)可以写成多元常态(multivariate normal).
$$
  p(\mathbf{y})
  &=
  \text{Normal}(\mathbf{y} \mid \mathbf{0}, \mathbf{K})
$$


$\mathbf{K}$ 是一个协方差矩阵(covariance matrix)这个矩阵是通过每一对数据集的输入$k(\mathbf{x}_n, \mathbf{x}_m)$得到的。

上面的方法直接试用于回归那些在$\{0,1\}$上的 $\mathbb{y}$真值响应，但不是作为(二)分类。为了处理这些分类，我们把响应解释为压缩在$[0,1]$上的潜变量(latent variables)。然后，我们通过伯努利(Bernoulli)决定标签，概率由被压缩的值给定。

从观测值$(\mathbf{x}_n, y_n)$定义似然：
$$
  p(y_n \mid \mathbf{z}, x_n)
  &=
  \text{Bernoulli}(y_n \mid \text{logit}^{-1}(\mathbf{x}_n^\top \mathbf{z})).
$$

定义先验是一个多变量的协方差矩阵：
$$
  p(\mathbf{z})
  &=
  \text{Normal}(\mathbf{z} \mid \mathbf{0}, \mathbf{K}),
$$

现在通过Edward简历模式。我们用径向基函数RBF(radial basis function)核,也可以被称之为平方指数(squared exponential)或者(取幂两次)exponentiated quadratic。它返回由所有数据点评估出的核矩阵；用Cholesky分解矩阵使参数呈多元正态分布。
` ` ` Python
from edward.models import Bernoulli, MultivariateNormalCholesky
from edward.util import rbf

X = tf.placeholder(tf.float32, [N, D])
f = MultivariateNormalCholesky(mu=tf.zeros(N), chol=tf.cholesky(rbf(X)))
y = Bernoulli(logits=f)
` ` `

然后我们定义占位符(placeholder)\texttt{X}。我们按照数据将数值存入占位符。

## inference

执行变分推断。
将变分模型定义为一个完全分解的正态分布。
` ` ` Python
qf = Normal(mu=tf.Variable(tf.random_normal([N])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([N]))))
` ` `

运行变分推断\texttt{500}代。
` ` ` Python
inference = ed.KLqp({f: qf}, data={X: X_train, y: y_train})
inference.run(n_iter=500)
` ` `

在这个例子中\texttt{KLqp}默认使用再参数化梯度(reparameterization gradient)最小化$\text{KL}(q\|p)$散度测量(divergence measure)

对于更详细的推测过程，可以参考\href{/tutorials/klqp}{$\text{KL}(q\|p)$教程。(这个例子的过程很缓慢，因为在高斯过程中对完全协方差的求值和逆过程都很慢。)

## 参考和引用

**Rasmussen, C. E., & Williams, C. (2006). Gaussian processes for machine learning. MIT Press.**












　　　　　　　　　　　
