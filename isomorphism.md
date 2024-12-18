##### LDF:

label and degree filtering $$C(u)=\{v\in V(G)|L(v)=L(u)\wedge d(v)\geqslant d(u)\}$$

##### NLF:

neighbor label frequency filtering $$|N(u,l)|>|N(\upsilon,l)|\mathrm{~where~}L(N(u))=\{L(u^{\prime})|u^{\prime}\in N(u)\}\mathrm{~and~}N(u,l)=\{u^{\prime}\in N(u)|L(u^{\prime})=l\}$$

#### Filtering Methods:

##### CFL:

1. BFS树$$q_t$$根据LDF和NLF生成C(u)
2. 按照顺序自下而上根据邻居节点candidate细化C(u)

空间复杂度O(|V(q)|$$\times$$|E(G)|)

时间复杂度O(|E(q)|$$\times$$|E(G)|)

##### CECI:

1. 按照顺序根据LDF构造C(u)
2. 根据邻居节点candidate逆序提炼C(u)

时间复杂度和空间复杂度O(|E(q)|$$\times$$|E(G)|)

##### DP-iso:

1. BFS q 根据LDF生成C(u)
2. 按顺序根据邻居节点candidate提炼C(u)(重复k次交替反向顺序)

时间复杂度和空间复杂度O(|E(q)|$$\times$$|E(G)|)

#### Ordering Methods:

##### QuickSI:

infrequent-edge first ordering method

##### GraphQL:

left-deep join based method

##### CFL:

path-based ordering method

##### CECI:

C(u) is generated by NLF

$$u_r=arg \min_{u\in V(q)}\frac{|C(u)|}{d(u)}$$ as the start vertex of $$\varphi$$

performs a BFS on q started from $$u_r$$ to obtain the matching order

##### DP-iso:

C(u) is generated by LDF

$$u_r=arg\min_{u\in V(q)}\frac{|C(u)|}{d(u)}$$ as the start vertex of

DP-iso generates a collection of tree-like paths P in q according to δ

##### RI:

$$u^{*}=arg\operatorname*{max}_{u\in V(q)}d(u)$$ as the start vertex of $$\varphi$$

iteratively select $$u^*=arg\max_{u\in N(\varphi)-\varphi}|N(u)\cap\varphi|$$

breaks ties by the maxinum value of

1.  $$|\{u^{\prime}\in\varphi|\exists u^{\prime\prime}\in V(q)-\varphi,e(u^{\prime},u^{\prime\prime})\in E(q)\wedge e(u,u^{\prime\prime})\in E(q)\}|$$
2. $$|\{u^{\prime}\in N(u)-\varphi|\forall u^{\prime\prime}\in\varphi,e(u^{\prime},u^{\prime\prime})\notin E(q)\}|$$

##### VF2++:

first picks the vertex $$u \in V(q)$$ the label of which is least frequently appeared in G but with the largest degree as the root vertex $$u_r$$

iteratively select $$u^*=arg\max_{u\in V_i(q_t)-\varphi}|N(u)\cap\varphi|$$

breaks ties by

1. the largest degree value
2. the minimum value of $$|\{\upsilon\in V(G)|L(u)=L(\upsilon)\}|$$

#### Optimization Methods:

##### Graph Compression:

1. 数据图压缩技术，只适用于数据图密集时
2. 查询图压缩方法，只有少数查询顶点能被压缩
3. CFL表现的比压缩技术更好

##### Failing Sets Pruning:

失败集记录搜索子树中导致失败的部分映射或条件，利用这些失败集的信息提前排除某些部分映射

在探索以部分结果 M 为根的子树时，如果确定了某些映射组合无效：

- 记录这些映射组合或条件为失败集。

在搜索树的同一层（即 M 的兄弟节点）或更深层次时：

- 检查当前的部分结果是否满足失败集的条件。
- 如果满足，直接剪枝，避免无效计算。

#### Glasgow Algorithm:

基于 **约束编程（Constraint Programming, CP）** 思想实现。它将查询图 q 和数据图 G 的匹配问题转化为变量和约束的组合，通过高效的剪枝和搜索策略解决子图匹配问题。搜索过程中维护了许多状态，消耗了大量内存。

### Experiments:

#### Evaluating Filtering Methods:

##### Preprocessing Time:

![image-20241213191357203](isomorphism.assets/image-20241213191357203.png)

##### Number of Candidate Vertices:

![image-20241213194635920](isomorphism.assets/image-20241213194635920.png)

#### Evaluating Enumeration Methods:

![image-20241213200859797](isomorphism.assets/image-20241213200859797.png)

![image-20241213202526768](isomorphism.assets/image-20241213202526768.png)

#### Evaluating Ordering Methods:

##### Enumeration Time:

![image-20241214131657025](isomorphism.assets/image-20241214131657025.png)

![image-20241214132100775](isomorphism.assets/image-20241214132100775.png)

RI completes more than 95% queries within 1 second on $$Q_{32D}$$and $$Q_{32S}$$ , which significantly outperforms other algorithms.

##### Number of Unsolved Queries:

![image-20241215173127367](isomorphism.assets/image-20241215173127367.png)

If a query cannot be completed within the time limit by any competing algorithms, it is called as a fail-all query.

##### Spectrum Analysis:

![image-20241215184819988](isomorphism.assets/image-20241215184819988.png)

The blue point is the enumeration time with our generated orders.

![image-20241215191435548](isomorphism.assets/image-20241215191435548.png)

For comparison, we also measure the performance of the other algorithms under study as well as the performance of the 1000 randomly sampled matching orders.

It also lists the number of queries with a speedup of more than 10 times, denoted as ">10".

##### Discussion:

因为当数据集稀疏时，给定两个顶点，它们拥有更少的公共邻居，所以优先考虑非树边更有效，所以RI在稀疏图上表现非常好，因为RI的排序方法在每一步选择具有更多后向邻居的查询顶点。GQL和RI表现得比其他方法好。

#### Evaluating Optimization Methods:

失败集剪枝操作在$$Q_4$$和$$Q_8$$表现比没有差，在大的查询图中优化效果好，因为可以提前终止大型查询。

#### Overall Performance:

![image-20241215195543606](isomorphism.assets/image-20241215195543606.png)

#### Scalability Evaluation:

![image-20241215200902582](isomorphism.assets/image-20241215200902582.png)

### Conclusions:

#### Comparison with results in previous research:

##### previous studies:

1. 没有算法能在所有情况比其他算法都好，排序方法会生成一些无效匹配顺序
2. 不同算法在同一个查询集中得差异很大，不同的查询在同一查询集中的运行时间也可能差异很大

##### latest algorithms:

例如CFL、CECI和DP-iso因为有效的过滤方法和排序方法的整体性能取得了很大的改进

但实验结果表明GraphQL 的过滤方法与最新算法具有竞争力，并且最新算法的排序方法比 QuickSI 和 GraphQL 等早期算法表现更差。

新算法整体表现比GraphQL好因为用来辅助数据结构来保存候选之间的边

##### preprocessing-enumeration algorithms:

表现优于直接枚举算法

原因：

1. 过滤方法生成的候选顶点集可以为排序方法提供更准确的信息
2. 辅助数据结构显着提高了局部候选计算的效率

#### Effectiveness of techniques in each category:

1. GraphQL、CFL和DP-iso的滤波方法在pruning能力方面优于CECI，三种方法通常相互竞争
2. GraphQL 和 RI 的排序方法通常在竞争排序方法中最有效，因为它们倾向于将非树边放在匹配顺序前面。CFL 和 DP-iso 中的基于路径的排序方法可能会导致大量未解决的查询，并且自适应排序在我们的实验中不会主导静态排序。
3. 局部候选计算方法对枚举性能有很大影响，基于集合交集的方法在竞争方法中表现最好。因此，有必要构建辅助数据结构来保持候选者之间的边缘。
4. 失败的集修剪技术会减慢小型查询的性能，但可以显着提高大型查询的性能并减少未解决查询的数量。

#### Recommendation:

1. 默认使用GraphQL的候选顶点计算方法。如果预处理时间通常支配查询时间，则切换到 CFL 或 DP-iso 的方法。
2. 分别采用GraphQL和RI在密集和稀疏数据图上的排序方法
3. 使用 CECI/DP-iso 风格的辅助数据结构来保持候选者之间的边缘，并采用基于集合交集的局部候选计算。如果数据图非常密集，则使用 QFilter 作为集合交集方法。
4. 启用在大型查询上的失败集修剪，但在小型查询上禁用它。
