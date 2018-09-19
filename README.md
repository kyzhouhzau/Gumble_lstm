# Towards Binary-Valued Gates for Robust LSTM Training

### 一、首先给出LSTM公式

$i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i)$
$f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f)$
$o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o)$
$z_i = tanh(W_{xz}x_t + W_{hz}h_{t-1} + b_z))$
$c_t = c_{t-1}*f_i+z_i*i_t$
$o_t = o_{t}*tanh(c_t)$

在输入门，与遗忘门通常使用的是$\sigma$函数，公式是：$\sigma(y) = \frac{e^{yi}}{\sum^N_{j=1}{e^{yj}}}$是一个0-1的值。我们选用该函数的目的就是为了用这些门来控制信息的流动。在此文中作者提出一种新的方法，即用 Gumbel 函数来更好的逼近0，和1从而使得更好的达到我们预期的目标。因为：此时函数大多时候处于0，和1附近，而不是出于0-1的某些中间值，使得结果解释性更好。

通过实验作者发现。
1、尽管看上去限制取值的柔度会影响模型效果，但实际情况并不是如此。
2、每一个门的输出对他们的输入不会特别敏感。

### 二、证明Gumbel的可用性
定义$\sigma(.)$是是一个sigmoid()函数。给定$\alpha\in R$ 并且$\tau>0$.定义随机变量$D_\alpha \sim B(\sigma(\alpha)) $ 这里$B(\sigma(\alpha))$是伯努利分布，定义$G(\alpha,\tau) = \sigma(\frac{\alpha+logU-log(1-U)}{\tau})$,这里$U \sim Uniform(0,1)$ 那门下面的不等式在当$ \varepsilon \in (0,1/2)$时，成立。
$P(D_\alpha = 1)-(\tau/4)log(1/\varepsilon)\leqslant P(G(\alpha,\tau) \geqslant1-\varepsilon)\leqslant P(D_\alpha = 1)$                   (1)
$P(D_\alpha = 0)-(\tau/4)log(1/\varepsilon)\leqslant P(G(\alpha,\tau) \geqslant1-\varepsilon)\leqslant P(D_\alpha = 0)$ (2)

###### 证明：
$\sigma(x) = \frac{1}{1+e^{-x}}$
则$\sigma^{-1}(x) = log\frac{x}{1-x}$

$P(G(\alpha,\tau)\geqslant1-\varepsilon)$
$=P( \sigma(\frac{\alpha+logU-log(1-U)}{\tau})\geqslant 1-\varepsilon)$

###### 这里用的是逆哈函数，直接两边取逆函数，根据性质当原函数递增的时候，逆函数也递增。


$=P(\frac{\alpha+logU-log(1-U)}{\tau}\geqslant log(\frac{1}{\varepsilon}-1))$
$=P(e^{\alpha - \tau log(\frac{1}{\varepsilon}-1)}\geqslant(1-U)U)$
$=P(U \geqslant \frac {1}{1+e^{\alpha-\tau log(\frac{1}{\varepsilon}-1)}}$
$=\sigma(\alpha-\tau log(\frac{1}{\varepsilon}-1))$

考虑到sigmoid function是四分之一的Lipschitz 连续并且单点递增，于是有
$P(D_\alpha = 1)-P(G(\alpha,\tau)\geqslant1-\varepsilon)$
$=\sigma(\alpha)-\sigma(\alpha-\tau log(\frac{1}{\varepsilon}-1))$
$\geqslant (\frac{\tau}{4}) log(\frac{1}{\varepsilon}-1)\geqslant (\frac{\tau}{4}) log(\frac{1}{\varepsilon})$

并且$P(D_\alpha=1)-P(G(\alpha,\tau)\geqslant1-\varepsilon) \geqslant 0$

当$\tau$ 逼近0时候公式一，公式二再根据夹逼准则于,
$P(lim_{\tau-0^+} G(\alpha,\tau)=1)=P(D_\alpha =1)$
$P(lim_{\tau-0^+} G(\alpha,\tau)=0)=P(D_\alpha =0)$
于是Gumble函数近似二项分布

最终LSTM公式被修改成
$i_t = G(W_{xi}x_t + W_{hi}h_{t-1} + b_i,\tau),$
$f_t = G(W_{xf}x_t + W_{hf}h_{t-1} + b_f,\tau)$
$o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o)$
$z_i = tanh(W_{xz}x_t + W_{hz}h_{t-1} + b_z))$
$c_t = c_{t-1}*f_i+z_i*i_t$
$o_t = o_{t}*tanh(c_t)$


论文：Towards Binary-Valued Gates for Robust LSTM Training
<br>
博客：https://blog.csdn.net/zhoukaiyin_hzau/article/details/82347700
<br>
代码参考：https://github.com/zhuohan123/g2-lstm/blob/master/language-modeling/g2_lstm.py
