
# Reviewer WmS2

  

Dear Reviewer mQKF, we sincerely appreciate your informative feedback and helpful suggestions that helped enhance the integrity of our evaluations. We have added the new experiments following your suggestions, provided more discussions, and modified the paper and appendix accordingly. Please see our point-to-point response below. (need to motify)

  

>W1: "Theorem 3.3 is very similar to Theorem 1 in Liu et al. (2024). This should be properly acknowledged, and hence claims on innovation are weaker than presented." and ""

  

**A1**: Thank you for your valuable comment, which helps us clearly distinguish our contributions from those in Li et al. (2024). Although Theorem 3.3 has similarities with Theorem 1 in Liu et al. (2024) from the perspective of proof techniques we would like to highlight that there are important **distinctions**, which are shown as follows:

-  **Component-wise Identification vs. Block-wise Identification**: In our work, we prove that latent variables at each layer is block-wise identifiable, i.e., $z_t^l=h(\hat{z}_t^l)$. In the meanwhile, Li et al. (2024) does not consider the hierarchical latent architecture and show the component-wise identifiable, i.e., $z_{t,i}=h(\hat{z}_{t,j})$.

  

-  **Block-wise Identification without Permutation**: IDOL shows the

  
  

Unlike IDOL, our method does not involve permutations between layers. This is a key difference, which is explicitly addressed in Lemma 3 of our paper. The lack of permutation between layers ensures that the structure remains more interpretable and stable.

  

-  **Use of Prior Structure** vs. Sparsity Constraint: Our method incorporates a prior structure that guides the learning process, whereas IDOL relies on a sparsity constraint. This difference in approach is significant as the prior structure in our method provides additional flexibility and guidance that is not present in IDOL.

  

We believe these differences warrant a distinction between our work and that of Li et al. (2024), and we hope this clarification resolves any concerns about the innovation in our approach.

  

<!--

三个不同：

1.idol是compoent，我们是layerwise-block

2. 我们没有层之间permutation，IDOL有，具体不同在lemma3

3. 我们利用了先验结构，而且idol需要sparsity constraint

-->

  
  

>W2: "heorem 3.3 is very similar to Theorem 1 in Liu et al. (2024). This should be properly acknowledged, and hence claims on innovation are weaker than presented." and

  

**A2**:

<!-- 提一下hu的工作，IDOL只能处理可逆，单层的 -->

  
  

>W3: "The experimental results indicate that the proposed approach outperforms several baselines. However, it is noted that the IDOL model matches the performance of the proposed approach in a majority of cases. This observation suggests that the empirical advantage may not be as robust as claimed"

  
  

**A3**:

  

<!--

1. 对于模拟实验，我们提升了xxx%

2. 对于真实实验，我们平均提升了xxx%，我们发现在某些真实数据集上提升看起来不明显，但是由于指标本身很低，所以xxxx

  

-->

  
  
  

>W4: "The empirical validation would be improved by running additional model/dataset seeds to rule out the possibility that the observed superiority is due to a favorable random seed."

  
  
  

<!--

  

我们所有的数据集都在三个随机种子上跑，重新公布方差

  

考虑了三个real-world数据集

  

-->

**A4**

  
  

>W5: "However, the prior design shows significant overlap with IDOL, particularly in Lines 294-318, which are highly similar to Section 4.2 of IDOL (Liu et al. (2024)). Given that the hierarchical prior is a central component of the proposed model, this similarity should be acknowledged, and the authors should clarify whether their approach introduces any substantive improvements over IDOL’s prior mechanism."

  

**A5**:

  
  

<!--

  

使用了相同的prior（这个prior widely used in TDRL，我们已经引用了），但是输入不同，在这儿我们利用了先验结构信息（indutive bias），但是IDOL因为没有先验结构，xxxx

  

-->

  
  

>W6: "A key limitation is that the real-world datasets do not inherently follow a hierarchical latent variable process. This weakens the motivation for the proposed approach because it is unclear whether the improvements stem from the explicit hierarchical modeling or simply from better parameterization. Evaluating on a dataset where hierarchical latent structures are explicitly known (or can be inferred) would strengthen the empirical justification."

  
  

<!--

  

天气一年四季

  

-->

  

**A6**:

  
  

>W7: So far, the dimensionality of each latent $z_t^l$ is n. However, in some places we can see $z_t \in \mathbb{R}^n$ instead of $\mathbb{R}^{n\times L}$

  

>W7.1: Definition 2.2.

  

<!-- 这个没错的 -->

  

>W7.2: Line 167 in Theroem 3.3. In this case the dimensionality might be correct, but you would probably were considering the first layer of estimated $z_t$

  

<!-- 承认错误 -->

  

>W7.3: I would suggest to revise this in the main text and the proofs. Specially because the proof for Theorem B.2 results confusing as it is unclear whether the above refers to the first layer of an estimated latent or the whole latents.

  

<!-- 承认错误，告诉他我们怎么改 -->

  

>W8: What is the dimensionality of the noise? Could you clarify the input/output domain of $g$ in Eq. (1)? If g is invertible and both x and z are n-dimensional, does this mean the noise is 0-dimensional or $g$ is non-injective?

  
  

<!-- 一维的 g: R^n+1->R^n, 噪声为0的时候就是可逆的 -->

  
  

>W9 "t seems the paper assumes first-order temporal and hierarchical connections. If this is the case, could we replace Eq. (2) for $z_{t,i}^l=f_i^l(z_{t-1,i}^l, z_{t,i}^{l+1})$"? Same reasoning for Eq. (3).

  

<!-- 没有问题，改正就是 -->

  
  

>W10: Definition 2.1 needs specific refinement for the presented problem

>W10.1: Given we are dealing with a hierarchical latent process and we have $n\times L$ latents, what is the support of the permutation $\pi$ and component-wise transformation?

  

<!--

  

我们这儿的pi是层内的permutation，层之间是没有permutation的

  

其实pi应该是有一个pi^l, 他的support是 (L x n) x (L x n) 的矩阵, block diagnose, transformation 和之前的没变都是z_i 到\hat{z}_j

  

-->

  

>W10.2: I am not sure Eq. (4) or (5) are correct here. The inverse of $g$ does not give us $z$, but a tuple $(z, \epsilon)$

. Furthermore, depending on the dimensionality of the noise, an inverse of $g$ is not directly possible.

  
  

<!--

承认写错了，但是不知道正确怎么写好

-->

  
  

>W10.3: Definition 3.1 requires connection to the linear operators presented in (ii). 1) Given we use linear operators in terms of probability densities, would it be possible to introduce the definition directly for probability distributions? 2) This would help clarify the notation introduced for $h_{b|a}(\cdot|a)$. Otherwise, this expression needs to be clarified.

  
  

<!--

1) 可以用概率表示，但是我们遵循了hu这篇的定义，使得他更加general（h是一个general function包括概率），感谢您的意见，我们补充说明可以用概率表示

-->

  
  

>W11.1 The paper assumes the linear operators are injective, but it does not provide conditions under which these operators satisfy injectivity.

  

<!--

参考minghao的论文，P22

-->

  
  

>W11.2: The Discussion of Assumptions paragraph connects assumptions to real-world cases, but it does not help verify whether the proposed generative model satisfies the assumptions.

  

<!--

real-world里面怎么去验证

-->

  
  

>W11.3: For example, what are the assumptions that we need on $g$ and $f$ such that the linear operators are injective?

  
  

<!--

Minghao写，additive noise, general nonliner, post-nonlinear

-->

  

Thank you for raising this point. The explicit form of functions f and g is still under discussion in functional analysis. We consider several common forms:

1. g(x) = h(z) + e, where h is nonlinear and invertible, and the distribution of e must not vanish everywhere under the Fourier transform.

2. p(x, z) follows a joint exponential distribution, which is a widely used assumption in many literatures.

3. g(x) = h_1(h_2(x) + e), a post-nonlinear form with invertible nonlinear h_1, h_2.

4. g(x) = h(x, e), a general nonlinear form. Small deviations from the additive model (e.g., polynomials) are often still manageable.

  

We will include a detailed discussion of these forms in the paper.

  

>W11.4: If these conditions are not guaranteed in practice, this could weaken the theoretical validity of Theorem 3.2.

  

<!--

问Kun，最后回答

可以参考：https://openreview.net/forum?id=nzgvkQM3EH&noteId=vCyG5SaUW9

-->

  
  

>W12.1: Given the proof is build from previous work, could you clarify what are theoretical innovations on this point?

<!--

是一个extension，我们进一步利用了temporal structural中multi measurement的属性，还有injective的假设，是的injective 假设更加容易满足，创新在于2L+1

-->

  
  

>W12.2: "The statement requires further acknowledgement from Li et al. (2024)." and "Both statement and proof are very similar to this work, and the similarities between both approaches should be carefully addressed."

  

<!--

和上面一样

-->

  
  

>W13: To improve soundness, more dataset or model seeds should be run.

  

<!-- zijian 放 -->

  
  

>W14: Section 5.2.2 and 5.2.3 names might be swapped by mistake.

  

<!-- 承认错误 -->

  

>W15: The number of latent variables and/or layers are assumed known. What happens empirically if either of these values are misspecified?

  

The dimension of latent space is previously considered arbitrary in previous CRL works. However, here we provide a brief proof that the invertible function $h_z$ preserves the dimensionality, that is $d_{\hat{z}} = d_z$. We analyze two scenarios:

\begin{enumerate}[label=\roman*),leftmargin=*]

\item $d_{\hat{z}} > d_z$: This implies that only $d_z$ components in $\hat{\mathbf{z}}_t$ are required to reconstruct the observations $\mathbf{x}_t$. Any variation in the remaining $d_{\hat{z}} - d_z$ components would not affect $\mathbf{x}_t$. Let $\hat{\mathbf{z}}_t, \hat{\mathbf{z}}_t$ then we can always find

\begin{equation}

p(\mathbf{x}_t \mid \mathbf{z}_{t,:d_{\hat{z}} - d_z}, \mathbf{z}_{t,d_{\hat{z}} - d_z:}) = p(\mathbf{x}_t \mid \mathbf{z}_{t,:d_{\hat{z}} - d_z}, \mathbf{z}'_{t,d_{\hat{z}} - d_z:}),

\end{equation}

which contradicts the Assumption~\ref{asp:nonredundant}.

\item $d_{\hat{z}} < d_z$: This suggests that only $d_{\hat{z}}$ dimensions are sufficient to describe $\mathbf{x}_t$, leaving $d_z - d_{\hat{z}}$ components constant, which violates that there are $d_z$ latent \textit{variables}.

\end{enumerate}

In summary, if dimensionality is not preserved, it contradicts the assumptions or the sufficiency of the latent representation.

  

<!--

  

维度：Minghao，维度可以设大一点，多出来的是noise

  

层数：退化成IDOL

  

-->

  
  
  

# Reviewer oTjA

  

Dear Review oTjA, we highly appreciate the valuable commands and helpful suggestions on our paper and the time dedicated to reviewing it. Below please see our point-to-point responses to your comments and suggestions.

  
  

>W1: However, the main theoretical contribution (Thm 3.2) is not presented clearly.

  
  

>W2: however, they are not motivated well in the paper. The physical meanings of the learned variables are not highlighted well, e.g., in the real-world data sets, what do these variables correspond to? How can can we further utilize them in downstream tasks?

  
  

<!-- -->

  

>W3.1: The linear operators $L_{x_{t+1,\cdots,x_{t+L}}|z_t}$ and $L_{\cdot|\cdot} are not defined before they are used in ln. 145-146 column 2.

  

<!-- 我们在definition3.1已经定义了，根据他的定义这两个算子展开是怎样xxxx，将 t+1,\cdots,x_{t+L}看作一个vector-->

  
  

>W3.2: Thm 3.2 has a condition to achieve identifiability: “Suppose that we have learned $(\hat{g}, \hat{f}, P(\hat{z}_t))$ to achieve Equations (1) and (2), …‘. What does this mean? It should only be possible to match the marginal distributions of the observations. Where is this condition used in the proof in App. B.1?

  

<!--

问Kun

-->

  
  

>W4.1 Definition A.1. is not clear. As a result, Eq. 22 (ln. 764-765) is not clear.

  

>W4.2 Can you define the mathematical object $D_{x_t|z_t}=P(x_t|z_t)$ formally? From the definition, what is $f(\cdot)$ here? What is $g_{b|a}$?

  

<!-- 全部用概率表示 -->

  
  

>W4.3 The notation is hard to follow. Can you explain what the left- and right- inverse objects are, and their definitions? For the right-inverse definition in ln. 684, $L_{b|a}\circ L_{b|a}^{-1}\circ p_a=p_a, \forall a\in A$, it is not clear how $L_{b|a}^{-1}$ is defined. Can you write their domains/codomains explicitly? What are the domains/codomains for the operators in Eq. 28, 29?

  
  

<!-- 立near operator 左右逆的定义（范函的内容）

找定义，提出帮助理解的情况，将函数看成矩阵，例如概率离散情况

-->

  
  

>W4.4 Why does the unique eigenvalues of Eq. 29 imply identifiability? There is no predicted $\hat{z}_t$ among the equations until this point. I do not see how (Eq. 32) $z_t=H(\hat{z}_t)$ follows since we do not have access to the true conditional densities $p(x_t|z_t)$. Only the marginal data distributions can be matched.

  

<!--

  

Minghao：告诉他的intuition，然后在indeterminacy2提到，再说怎么推

  

-->

  

The intuition is that, the eigenvalues are $p(x_t|z_t)$, if the eigenvalues is identifiable, i.e., unique eigendecomposition, the truth and estimated eigenvalues are totally same, except for their labels --- $\dot{z}_t$ and $\dot{\hat{z}}_t$ are exchanged, which is a bijective operation.

  
  

>W5: What does it mean for these linear operators to be injective? The terminology makes the idea hard to understand. For example, what do these injective operators imply for the density functions considered in the proof? Or, is it possible to translate these assumptions into function space assumptions for the functions in Eqs. 1-2?

  

<!--

Minghao：

1. linear operators 本质上是一个函数，

2. injective 有什么用？intuitively就是x分布的变化可以表示z分布的变化，也就是说需要充足的x观测来刻画z的变化，再说proof哪儿用到了（公式27）

3. 问Kun

-->

  

Thank you for the comment. A linear operator can indeed be viewed as a transformation on the density function, mapping $p(z) \rightarrow p(x)$. Injectivity of this operator implies that changes in $p(z)$ are reflected in changes in $p(x)$.

  

Eq. (1)–(2) can be interpreted at the density level as $p(x) = \mathcal{L} p(z)$, where $\mathcal{L}$ denotes the transformation operator.

  

>W6: The paper does not provide a clear justification or motivation for why the real-world task (time-series generation) is well-suited to the proposed method. Specifically, it fails to explain why hierarchical latent variables should lead to improved time-series samples.

  

<!--

天气的例子，

hierarchical latent variables 可以有助于细粒度控制

-->

  
  

>W7: The paper does not provide a clear justification or motivation for why the real-world task (time-series generation) is well-suited to the proposed method. Specifically, it fails to explain why hierarchical latent variables should lead to improved time-series samples.

>W7.1: The data generating process provided in Eqs 57, 58 in Appendix E.1 is too simplistic. The dependency between the hierarchical latent variables is linear, while the dependency of the observations $x_t$ on $z_t$ is an identity function followed by a leaky ReLU activation.

  

<!--

  

首先我们不是identity,补了更加复杂的实验

  

-->

  

>W7.2: The theoretical framework assumes that the latent variables at different layers have the same dimensionality, while in the synthetic dataset, they have different dimensions.

  

<!--

  

观测多维

  

-->

  
  

>W7.3: The experiments lack generalization to different time-lags and number of hierarchical layers.

  
  

<!--

  

lag=2

  

-->

  
  

>W7.4: The results are provided for a single seed, which limits the ability to assess the statistical significance.

  
  

<!--

  
  
  

-->

  
  

>W8: Does the practical algorithm generalize to more dimensions ($n>8$), more layers ($L>2$), and more complex mixing functions $f()$ and $g()$?

  
  
  
  
  

# Review JbzF

  

Dear Reviewer JbzF, we are very grateful for your valuable comments, helpful suggestions, and encouragement. We provide the point-to-point response to your comments below and have updated the paper and appendix accordingly.

  

>W1: The Clockwork VAE would make a good additional point of comparison alongside the Koopman VAE and TimeVAE.

  

A1: Thanks for your valuable suggestions, which improve the completeness of our experiment results. In light of your suggestions, we have considered Clockwork VAE as baseline, experiment results are show as follows:

  

| | | ETTh | Stocks | MuJoco | fmri | Box | Gesture | Throwcatch | Discussion | Purchases | WalkDog |

|:-------------------:|:-------------:|:--------------:|:--------------:|:------------:|:---------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|

| Context-FID Score | Clockwork VAE | 0.892（0.059） | 0.807（0.252） | | 0.896（0.103） | 0.215(0.026) | 0.329(0.051) | 0.284(0.025) | 0.919(0.132) | 1.147(0.146) | 0.796(0.274) |

| | ChILD | 0.05(0.000) | 0.017(0.026) | 0.026(0.004) | 0.027(0.004) | 0.070(0.002) | 0.017(0.015) | 0.045(0.041) | 0.018(0.004) | 0.018(0.000) | 0.008(0.000) |

| Correlational Score | Clockwork VAE | 0.070（0.001） | 0.053（0.001） | | 18.434（0.030） | 0.215(0.026) | 0.329(0.051) | 0.284(0.025) | 8.624(0.032) | 7.741(0.055) | 7.691(0.047) |

| | ChILD | 0.002(0.000) | 0.005(0.001) | 0.002(0.000) | 0.069(0.003) | 0.021(0.002) | 0.04(0.011) | 0.146(0.015) | 0.180(0.000) | 0.127(0.001) | 0.345(0.001) |

  

>W2: "Why previous theoretical results can hardly identify the hierarchical latent variables?" should read "Why can't previous theoretical results identify the hierarchical latent variables?"

  

A2: Thanks for your reminder. We have modified it in the updated version.

  
  
  

# Reviewer aV78

  
  

>W1: It would be good if the authors could provide the code in a later version. The authors should also mention the size of each dataset in the appendix and how many latent variables the authors use for each real-world dataset in the experiments.

  
  

A1: Thanks for your comments. We have provided the source code in the

  

>W2: Lemma 3.4 claims that we can identify the component-wise latent variables under the conditional independence assumption. Does this claim mean that we can recover the whole causal graph from the learned model?

  
  

>W3: What is the implication of the conditional independence assumption on the underlying Structural Causal Model? I guess for the conditional independence to hold, the graph has to have some structure.