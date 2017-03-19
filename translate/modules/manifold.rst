
.. currentmodule:: sklearn.manifold

.. _manifold:

=========================
流形学习Manifold learning
=========================

.. rst-class:: quote

                 | Look for the bare necessities
                 | The simple bare necessities
                 | Forget about your worries and your strife
                 | I mean the bare necessities
                 | Old Mother Nature's recipes
                 | That bring the bare necessities of life
                 |
                 |             -- Baloo's song [The Jungle Book]



.. figure:: ../auto_examples/manifold/images/plot_compare_methods_001.png
   :target: ../auto_examples/manifold/plot_compare_methods.html
   :align: center
   :scale: 60

流形学习是一种非线性的降维方法，其各种算法致力于处理数据中高而无用的维度。

简介
====

高维数据集通常难以可视化——二维或者三维的数据可以被绘制出来以展现其内部结构，但是更高维度的数据展示却太过于抽象。为了对数据集的结构进行可视化，数据的维度必须用某种方法降低。

最简单的降维方法是对数据随机投影。虽然这种方法使数据结构某种程度上实现了可视化，但是选择的随机性过于偏离我们的期望。在随机投影中，数据中更有意义的结构可能丢失。 

.. |digits_img| image:: ../auto_examples/manifold/images/plot_lle_digits_001.png
    :target: ../auto_examples/manifold/plot_lle_digits.html
    :scale: 50

.. |projected_img| image::  ../auto_examples/manifold/images/plot_lle_digits_002.png
    :target: ../auto_examples/manifold/plot_lle_digits.html
    :scale: 50

.. centered:: |digits_img| |projected_img|


为了完成这一目标，一系列监督性或者非监督性的线性降维框架已经问世。比如说主成分分析(Principal Component Analysis,PCA)，独立成分分析(Independent Component Analysis)还有线性判别分析(Linear Discriminant Analysis)等等。这些算法定义了具体的规则以选择数据中“有意义”的线性投影。这些方法卓有成效，但是经常会错失数据中重要的非线性结构。

.. |PCA_img| image:: ../auto_examples/manifold/images/plot_lle_digits_003.png
    :target: ../auto_examples/manifold/plot_lle_digits.html
    :scale: 50

.. |LDA_img| image::  ../auto_examples/manifold/images/plot_lle_digits_004.png
    :target: ../auto_examples/manifold/plot_lle_digits.html
    :scale: 50

.. centered:: |PCA_img| |LDA_img|

流形学习可被看作线性框架（例如PCA）的一种泛化尝试，使这些框架可以适用于数据中非线性的结构。虽然流形学习中存在监督性的变量，但是典型的流形学习却是非监督的。它没有使用预先决定的分类，而是自学数据中的高维结构。

.. topic:: Examples:

    * See :ref:`example_manifold_plot_lle_digits.py` for an example of
      dimensionality reduction on handwritten digits.

    * See :ref:`example_manifold_plot_compare_methods.py` for an example of
      dimensionality reduction on a toy "S-curve" dataset.

sklearn中可供使用的流形学习实现如下。

.. _isomap:

等距特征映射 Isometric Feature Mapping
=====================================

Isomap是最易习得的流形学习算法之一。Isomap可以视为多维标度分析（multidimensional scaling ,MDS）或者核PCA在高维上的扩展。Isomap寻找一个更低维度的嵌入，这个嵌入保持了所有点之间的几何距离。Isomap可见于 :class:`Isomap` 。

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_005.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

复杂度
-----
Isomap算法由一下3步骤组成：

1. **最近邻搜索.**  Isomap使用 :class:`sklearn.neighbors.BallTree` 完成高效率的近邻搜索，对于在 :math:`D` 维空间 :math:`N` 个数据点中的 :math: `k` 个最近邻而言，其开销约为 :math:`O[D \log(k) N \log(N)]` 。

2. **最短路搜索.**  完成最短路搜索最有效的算法是 *Dijkstra 算法*，其复杂度约为 :math:`O[N^2(k + \log(N))]` ，或者是 *Floyd-Warshall算法* ,其复杂度为 :math:`O[N^3]` 。用户可以用 ``Isomap`` 的 ``path_method`` 关键字选择合适的算法。若无特指，程序则会自动选择最合适者。

3. **部分特征值分解.**  The embedding is encoded in the 
   eigenvectors corresponding to the :math:`d` largest eigenvalues of the
   :math:`N \times N` isomap kernel.  For a dense solver, the cost is
   approximately :math:`O[d N^2]`.  This cost can often be improved using
   the ``ARPACK`` solver.  用户可以用 ``Isomap`` 的 ``path_method`` 关键字选择合适的特征值算法。若无特指，程序则会自动选择最合适者。

整个Isomap算法的复杂度是 :math:`O[D \log(k) N \log(N)] + O[N^2(k + \log(N))] + O[d N^2]`.

* :math:`N` : 训练集数据个数
* :math:`D` : 输入数据维度
* :math:`k` : 最近邻数量
* :math:`d` : 输出数据维度

.. topic:: References:

   * `"A global geometric framework for nonlinear dimensionality reduction"
     <http://www.sciencemag.org/content/290/5500/2319.full>`_
     Tenenbaum, J.B.; De Silva, V.; & Langford, J.C.  Science 290 (5500)

.. _locally_linear_embedding:

局部线性嵌入 Locally Linear Embedding
=====================================

局部线性嵌入 (LLE)寻找一个能使数据保持与局部近邻点距离不变的低维投影。它可以看作一系列寻找全局最优非线性嵌入的局部PCA的组合。

LLE可以通过函数 :func:`locally_linear_embedding` 或者类 :class:`LocallyLinearEmbedding` 实现。

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_006.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

复杂度
------

标准的LLE算法由以下3部分组成：

1. **最近邻搜索**. 见上问对Isomap的讨论。

2. **权重矩阵构造**. :math:`O[D N k^3]`.LLE权重矩阵包含着对 :math:`N` 个局部近邻的 :math:`k \times k` 线性方程的解。

3. **部分特征值分解**. 见上文对Isomap的讨论。

整个标准LLE的复杂度为 :math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[d N^2]`.

* :math:`N` : 训练集数据个数
* :math:`D` : 输入数据维度
* :math:`k` : 最近邻数量
* :math:`d` : 输出数据维度

.. topic:: References:
   
   * `"Nonlinear dimensionality reduction by locally linear embedding"
     <http://www.sciencemag.org/content/290/5500/2323.full>`_
     Roweis, S. & Saul, L.  Science 290:2323 (2000)

改良局部线性嵌入Modified Locally Linear Embedding
=================================================

正则化是LLE的一个明显问题。当近邻数量大于输入数据的维数时，用于定义近邻的矩阵将秩亏。为此，标准LLE采用了一个任意的正则化参数 :math:`r` ,该值与局部权重矩阵的迹有关。虽然在形式上当math:`r \to 0` 时，该解会收敛于期望的嵌入，但并没有保证当 :math:`r > 0` 时有最优解。这个问题在一个扭曲流形底层几何的嵌入中体现。

一个处理正则化问题的方法是在每个近邻中使用多个权重向量。这就是*改良局部线性嵌入* (MLLE)的本质。MLLE可以在函数 :func:`locally_linear_embedding` 或者类 :class:`LocallyLinearEmbedding` 中使用，其中关键字 ``method = 'modified'`` 。它要求 ``n_neighbors > n_components`` 。

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_007.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50
   
复杂度
------

改良局部线性嵌入由以下3个部分组成：

1. **最近邻搜寻**.  同标准LLE算法。

2. **权重矩阵构造**.约为 :math:`O[D N k^3] + O[N (k-D) k^2]` 。其中第一项和标准的LLE相同，第二项从多个权重值中构造权重矩阵。在实践中，因构造MLLE权重矩阵而增加的开销和步骤1和3相比显得非常小。

3. **部分特征分解**. 同标准LLE算法。

整个MLLE模型的复杂度为 :math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[N (k-D) k^2] + O[d N^2]`.

* :math:`N` : 训练集数据个数
* :math:`D` : 输入数据维度
* :math:`k` : 最近邻数量
* :math:`d` : 输出数据维度

.. topic:: References:
     
   * `"MLLE: Modified Locally Linear Embedding Using Multiple Weights"
     <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382>`_
     Zhang, Z. & Wang, J.


Hessian Eigenmapping
====================

Hessian Eigenmapping (also known as Hessian-based LLE: HLLE) is another method
of solving the regularization problem of LLE.  It revolves around a
hessian-based quadratic form at each neighborhood which is used to recover
the locally linear structure.  Though other implementations note its poor
scaling with data size, ``sklearn`` implements some algorithmic
improvements which make its cost comparable to that of other LLE variants
for small output dimension.  HLLE can be  performed with function
:func:`locally_linear_embedding` or its object-oriented counterpart
:class:`LocallyLinearEmbedding`, with the keyword ``method = 'hessian'``.
It requires ``n_neighbors > n_components * (n_components + 3) / 2``.

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_008.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50
   
复杂度
-----

The HLLE algorithm comprises three stages:

1. **Nearest Neighbors Search**.  Same as standard LLE

2. **Weight Matrix Construction**. Approximately
   :math:`O[D N k^3] + O[N d^6]`.  The first term reflects a similar
   cost to that of standard LLE.  The second term comes from a QR
   decomposition of the local hessian estimator.

3. **Partial Eigenvalue Decomposition**. Same as standard LLE

The overall 复杂度 of standard HLLE is
:math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[N d^6] + O[d N^2]`.

* :math:`N` : 训练集数据个数
* :math:`D` : 输入数据维度
* :math:`k` : 最近邻数量
* :math:`d` : 输出数据维度

.. topic:: References:

   * `"Hessian Eigenmaps: Locally linear embedding techniques for
     high-dimensional data" <http://www.pnas.org/content/100/10/5591>`_
     Donoho, D. & Grimes, C. Proc Natl Acad Sci USA. 100:5591 (2003)

.. _spectral_embedding:

Spectral Embedding
====================

Spectral Embedding (also known as Laplacian Eigenmaps) is one method
to calculate non-linear embedding. It finds a low dimensional representation
of the data using a spectral decomposition of the graph Laplacian.
The graph generated can be considered as a discrete approximation of the 
low dimensional manifold in the high dimensional space. Minimization of a 
cost function based on the graph ensures that points close to each other on 
the manifold are mapped close to each other in the low dimensional space, 
preserving local distances. Spectral embedding can be  performed with the
function :func:`spectral_embedding` or its object-oriented counterpart
:class:`SpectralEmbedding`.

复杂度
-----

The Spectral Embedding algorithm comprises three stages:

1. **Weighted Graph Construction**. Transform the raw input data into
   graph representation using affinity (adjacency) matrix representation.

2. **Graph Laplacian Construction**. unnormalized Graph Laplacian
   is constructed as :math:`L = D - A` for and normalized one as
   :math:`L = D^{-\frac{1}{2}} (D - A) D^{-\frac{1}{2}}`.  

3. **Partial Eigenvalue Decomposition**. Eigenvalue decomposition is 
   done on graph Laplacian

The overall 复杂度 of spectral embedding is
:math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[d N^2]`.

* :math:`N` : 训练集数据个数
* :math:`D` : 输入数据维度
* :math:`k` : 最近邻数量
* :math:`d` : 输出数据维度

.. topic:: References:

   * `"Laplacian Eigenmaps for Dimensionality Reduction
     and Data Representation" 
     <http://www.cse.ohio-state.edu/~mbelkin/papers/LEM_NC_03.pdf>`_
     M. Belkin, P. Niyogi, Neural Computation, June 2003; 15 (6):1373-1396


Local Tangent Space Alignment
=============================

Though not technically a variant of LLE, Local tangent space alignment (LTSA)
is algorithmically similar enough to LLE that it can be put in this category.
Rather than focusing on preserving neighborhood distances as in LLE, LTSA
seeks to characterize the local geometry at each neighborhood via its
tangent space, and performs a global optimization to align these local 
tangent spaces to learn the embedding.  LTSA can be performed with function
:func:`locally_linear_embedding` or its object-oriented counterpart
:class:`LocallyLinearEmbedding`, with the keyword ``method = 'ltsa'``.

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_009.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

复杂度
-----

The LTSA algorithm comprises three stages:

1. **Nearest Neighbors Search**.  Same as standard LLE

2. **Weight Matrix Construction**. Approximately
   :math:`O[D N k^3] + O[k^2 d]`.  The first term reflects a similar
   cost to that of standard LLE.

3. **Partial Eigenvalue Decomposition**. Same as standard LLE

The overall 复杂度 of standard LTSA is
:math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[k^2 d] + O[d N^2]`.

* :math:`N` : 训练集数据个数
* :math:`D` : 输入数据维度
* :math:`k` : 最近邻数量
* :math:`d` : 输出数据维度

.. topic:: References:

   * `"Principal manifolds and nonlinear dimensionality reduction via
     tangent space alignment"
     <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.4.3693>`_
     Zhang, Z. & Zha, H. Journal of Shanghai Univ. 8:406 (2004)

.. _multidimensional_scaling:

Multi-dimensional Scaling (MDS)
===============================

`Multidimensional scaling <http://en.wikipedia.org/wiki/Multidimensional_scaling>`_
(:class:`MDS`) seeks a low-dimensional
representation of the data in which the distances respect well the
distances in the original high-dimensional space.

In general, is a technique used for analyzing similarity or
dissimilarity data. :class:`MDS` attempts to model similarity or dissimilarity data as
distances in a geometric spaces. The data can be ratings of similarity between
objects, interaction frequencies of molecules, or trade indices between
countries.

There exists two types of MDS algorithm: metric and non metric. In the
scikit-learn, the class :class:`MDS` implements both. In Metric MDS, the input
similarity matrix arises from a metric (and thus respects the triangular
inequality), the distances between output two points are then set to be as
close as possible to the similarity or dissimilarity data. In the non-metric
version, the algorithms will try to preserve the order of the distances, and
hence seek for a monotonic relationship between the distances in the embedded
space and the similarities/dissimilarities.

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_010.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50
 

Let :math:`S` be the similarity matrix, and :math:`X` the coordinates of the
:math:`n` input points. Disparities :math:`\hat{d}_{ij}` are transformation of
the similarities chosen in some optimal ways. The objective, called the
stress, is then defined by :math:`sum_{i < j} d_{ij}(X) - \hat{d}_{ij}(X)`


Metric MDS
----------

The simplest metric :class:`MDS` model, called *absolute MDS*, disparities are defined by
:math:`\hat{d}_{ij} = S_{ij}`. With absolute MDS, the value :math:`S_{ij}`
should then correspond exactly to the distance between point :math:`i` and
:math:`j` in the embedding point.

Most commonly, disparities are set to :math:`\hat{d}_{ij} = b S_{ij}`.

Nonmetric MDS
-------------

Non metric :class:`MDS` focuses on the ordination of the data. If
:math:`S_{ij} < S_{kl}`, then the embedding should enforce :math:`d_{ij} <
d_{jk}`. A simple algorithm to enforce that is to use a monotonic regression
of :math:`d_{ij}` on :math:`S_{ij}`, yielding disparities :math:`\hat{d}_{ij}`
in the same order as :math:`S_{ij}`.

A trivial solution to this problem is to set all the points on the origin. In
order to avoid that, the disparities :math:`\hat{d}_{ij}` are normalized.


.. figure:: ../auto_examples/manifold/images/plot_mds_001.png
   :target: ../auto_examples/manifold/plot_mds.html
   :align: center
   :scale: 60
  

.. topic:: References:

  * `"Modern Multidimensional Scaling - Theory and Applications"
    <http://www.springer.com/statistics/social+sciences+%26+law/book/978-0-387-25150-9>`_
    Borg, I.; Groenen P. Springer Series in Statistics (1997)

  * `"Nonmetric multidimensional scaling: a numerical method"
    <http://www.springerlink.com/content/tj18655313945114/>`_
    Kruskal, J. Psychometrika, 29 (1964)

  * `"Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis"
    <http://www.springerlink.com/content/010q1x323915712x/>`_
    Kruskal, J. Psychometrika, 29, (1964)

.. _t_sne:

t-distributed Stochastic Neighbor Embedding (t-SNE)
===================================================

t-SNE (:class:`TSNE`) converts affinities of data points to probabilities.
The affinities in the original space are represented by Gaussian joint
probabilities and the affinities in the embedded space are represented by
Student's t-distributions. This allows t-SNE to be particularly sensitive
to local structure and has a few other advantages over existing techniques:

* Revealing the structure at many scales on a single map
* Revealing data that lie in multiple, different, manifolds or clusters
* Reducing the tendency to crowd points together at the center

While Isomap, LLE and variants are best suited to unfold a single continuous
low dimensional manifold, t-SNE will focus on the local structure of the data
and will tend to extract clustered local groups of samples as highlighted on
the S-curve example. This ability to group samples based on the local structure
might be beneficial to visually disentangle a dataset that comprises several
manifolds at once as is the case in the digits dataset.

The Kullback-Leibler (KL) divergence of the joint
probabilities in the original space and the embedded space will be minimized
by gradient descent. Note that the KL divergence is not convex, i.e.
multiple restarts with different initializations will end up in local minima
of the KL divergence. Hence, it is sometimes useful to try different seeds
and select the embedding with the lowest KL divergence. 

The disadvantages to using t-SNE are roughly:

* t-SNE is computationally expensive, and can take several hours on million-sample
  datasets where PCA will finish in seconds or minutes
* The Barnes-Hut t-SNE method is limited to two or three dimensional embeddings.
* The algorithm is stochastic and multiple restarts with different seeds can
  yield different embeddings. However, it is perfectly legitimate to pick the the
  embedding with the least error.
* Global structure is not explicitly preserved. This is problem is mitigated by
  initializing points with PCA (using `init='pca'`).


.. figure:: ../auto_examples/manifold/images/plot_lle_digits_013.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

Optimizing t-SNE
----------------
The main purpose of t-SNE is visualization of high-dimensional data. Hence,
it works best when the data will be embedded on two or three dimensions.

Optimizing the KL divergence can be a little bit tricky sometimes. There are
five parameters that control the optimization of t-SNE and therefore possibly
the quality of the resulting embedding:

* perplexity
* early exaggeration factor
* learning rate
* maximum number of iterations
* angle (not used in the exact method)

The perplexity is defined as :math:`k=2^(S)` where :math:`S` is the Shannon
entropy of the conditional probability distribution. The perplexity of a
:math:`k`-sided die is :math:`k`, so that :math:`k` is effectively the number of
nearest neighbors t-SNE considers when generating the conditional probabilities.
Larger perplexities lead to more nearest neighbors and less sensitive to small
structure. Larger datasets tend to require larger perplexities.
The maximum number of iterations is usually high enough and does not need
any tuning. The optimization consists of two phases: the early exaggeration
phase and the final optimization. During early exaggeration the joint
probabilities in the original space will be artificially increased by
multiplication with a given factor. Larger factors result in larger gaps
between natural clusters in the data. If the factor is too high, the KL
divergence could increase during this phase. Usually it does not have to be
tuned. A critical parameter is the learning rate. If it is too low gradient
descent will get stuck in a bad local minimum. If it is too high the KL
divergence will increase during optimization. More tips can be found in
Laurens van der Maaten's FAQ (see references). The last parameter, angle,
is a tradeoff between performance and accuracy. Larger angles imply that we
can approximate larger regions by a single point,leading to better speed
but less accurate results. 

Barnes-Hut t-SNE
----------------

The Barnes-Hut t-SNE that has been implemented here is usually much slower than
other manifold learning algorithms. The optimization is quite difficult
and the computation of the gradient is :math:`O[d N log(N)]`, where :math:`d`
is the number of output dimensions and :math:`N` is the number of samples. The 
Barnes-Hut method improves on the exact method where t-SNE 复杂度 is 
:math:`O[d N^2]`, but has several other notable differences:

* The Barnes-Hut implementation only works when the target dimensionality is 3
  or less. The 2D case is typical when building visualizations.
* Barnes-Hut only works with dense input data. Sparse data matrices can only be
  embedded with the exact method or can be approximated by a dense low rank
  projection for instance using :class:`sklearn.decomposition.TruncatedSVD`
* Barnes-Hut is an approximation of the exact method. The approximation is
  parameterized with the angle parameter, therefore the angle parameter is
  unused when method="exact"
* Barnes-Hut is significantly more scalable. Barnes-Hut can be used to embed
  hundred of thousands of data points while the exact method can handle
  thousands of samples before becoming computationally intractable

For visualization purpose (which is the main use case of t-SNE), using the
Barnes-Hut method is strongly recommended. The exact t-SNE method is useful
for checking the theoretically properties of the embedding possibly in higher
dimensional space but limit to small datasets due to computational constraints.

Also note that the digits labels roughly match the natural grouping found by
t-SNE while the linear 2D projection of the PCA model yields a representation
where label regions largely overlap. This is a strong clue that this data can
be well separated by non linear methods that focus on the local structure (e.g.
an SVM with a Gaussian RBF kernel). However, failing to visualize well
separated homogeneously labeled groups with t-SNE in 2D does not necessarily
implie that the data cannot be correctly classified by a supervised model. It
might be the case that 2 dimensions are not enough low to accurately represents
the internal structure of the data.


.. topic:: References:

  * `"Visualizing High-Dimensional Data Using t-SNE"
    <http://jmlr.org/papers/v9/vandermaaten08a.html>`_
    van der Maaten, L.J.P.; Hinton, G. Journal of Machine Learning Research
    (2008)

  * `"t-Distributed Stochastic Neighbor Embedding"
    <http://lvdmaaten.github.io/tsne/>`_
    van der Maaten, L.J.P.

  * `"Accelerating t-SNE using Tree-Based Algorithms."
    <http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf>`_
    L.J.P. van der Maaten.  Journal of Machine Learning Research 15(Oct):3221-3245, 2014.

实践技巧
=====================

* 确保对全部特征使用了相同的度量指标，因为流形学习是基于最近邻搜索的，所以如果没有统一的度量标准，算法的表现会非常的不理想。对不同的数据进行统一度量指标的方法亲详见 :ref:`StandardScaler <preprocessing_scaler>` 。

* 由每一个步骤算出的重构误差，可用于选择最优输出维数。对于嵌入在D-维空间的d-维流形而言，重构误差会随着 ``n_components`` 的增加而减小，直至 ``n_components == d`` 。

* 请注意噪声数据，特别是在流形的不同部分构成桥状相连的噪声数据，因为它会使流形学习出现短路。目前对嘈杂或者不完整的数据进行的流形学习是研究的热门。

* Certain input configurations can lead to singular weight matrices, for
  example when more than two points in the dataset are identical, or when
  the data is split into disjointed groups.  In this case, ``solver='arpack'``
  will fail to find the null space.  The easiest way to address this is to
  use ``solver='dense'`` which will work on a singular matrix, though it may
  be very slow depending on the number of input points.  Alternatively, one
  can attempt to understand the source of the singularity: if it is due to
  disjoint sets, increasing ``n_neighbors`` may help.  If it is due to
  identical points in the dataset, removing these points may help.

.. seealso::

   :ref:`random_trees_embedding` can also be useful to derive non-linear
   representations of feature space, also it does not perform
   dimensionality reduction.

