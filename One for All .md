#### 题目：

​	ONE FOR ALL: TOWARDS TRAINING ONE GRAPHMODEL FOR ALL CLASSIFICATION TASKS

#### 背景：

​	LLMs因其强大的学习能力，在执行跨领域的下游任务上取巨大的成功，但是在图领域上还没有一个统一的模型解决图结构数据上不同的任务。

#### 创新点：

#####  文本属性图（TAG）：

​	OFA使用文本属性图（TAGs）将来自不同领域的图数据整合到一个大的TAG数据集，并利用LLMs的能力从所有领域中共同学习。9个常见的来自不同领域的数据集，用文本描述图中所有的结点和边，然后使用单个LLM将来自不同领域的文本嵌入到同一个嵌入空间。

#####  NOI子图 和NOI提示结点：

​	NOI（nodes-of-interest）子图 和NOI提示结点，不仅统一了不同类型的图任务，而且提高了基础模型学习图中结构信息的能力。

#####  图提示范式 (GPP)：

​	图提示范式 (graph prompting paradigm，GPP)——插入一个提示图到原始的输入图中的特定任务的方式。提示图中的节点包含关于下游任务的所有相关信息（由文本描述，并由与输入图相同的LLM编码器编码）。然后，修改后的图形成为基础模型的实际输入。



#### 介绍：

![image-20241020105228726](./One for All .assets/image-20241020105228726.png)

LLM可以将图的任务描述和图中的跨域文本共同嵌入到同一空间中，OFA中的GPP转化为统一任务表示的提示图，从而允许自适应下游预测。

建立基础模型统一不同域的图形数据：
Text feature of nodes: Feature node. <feature description>: <feature content>; <feature
description>: <feature content>

Example: 

1. Feature node. Atom: Carbon, Atomic number 6, helix chirality, is not in a ring

2. Feature node. Paper title and abstract: Attention is all you need. The dominant sequence transduction models are .



Text feature of edges: Feature edge. <feature description>: <feature content>; <feature
description>: <feature content>;

Example:

1. Feature edge. Chemical Bond: ionic bonding, is conjugated,

2. Feature edge. Citation from one paper to another



Text feature of the NOI prompt node: Prompt node. <task description>.

Example:

1. Prompt node. Graph classification on molecule properties.
2. Prompt node. Node classification on the literature category of the paper.



Text feature of class node: Prompt node. <class description>.

Example: 

1. Prompt node. Molecule property. The molecule is effective in:
2. Prompt node. Literature Category. cs.AI (Artificial Intelligence). Covers all areas of
   AI except Vision



![image-20241020105233181](./One for All .assets/image-20241020105233181.png)

认识到上下文学习的核心原则涉及操作输入数据，使其与下游任务保持一致。本文提出图提示范式(GPP)来操纵输入图，使图型可以从输入本身获取与任务相关的信息。这种范式赋予图模型对可见类和未见类都具有上下文学习能力，从而实现零样本学习。具体来说，提示图$$P = (V_p,E_p, R_p)$$ 有两种类型的节点，第一种节点类型是NOI提示节点，假设我们正在查询目标NOI子图$$G^q_h(T^q)=(V^h_q,E^h_q,R^h_q)$$ 并且NOI提示节点是$$p_q$$ ,GPP在NOI提示节点和NOI中的每个节点之间添加边，如图中虚线所示，



#### 实验：

![image-20241020105236336](./One for All .assets/image-20241020105236336.png)





#### 局限性：

由于回归目标可以是无限的所以学习回归任务存在不足

