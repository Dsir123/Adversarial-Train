# 对抗训练实验报告

## 实验任务

理解论文[ICLR2020](https://arxiv.org/abs/2001.03994)《FAST IS BETTER THAN FREE: REVISITING ADVERSARIAL TRAINING》里的对抗训练方法PGD、“Free”、FGSM，并应用到TextCNN中，对比Baseline及引入这三种对抗训练方法后的性能，评价指标包括Precision、Recall、F1-score、Acc、Early stop时的迭代次数及用时。

## 实验结果

表1 各对抗训练方法的实验结果

模型 |	Precision	| Recall	| F1	| Acc	| Steps	| Cost 
:---:|:----------:|--------:|:----:|:---:|:-----:|:----:|
Baseline | 90.84%	| 90.82%	| 90.81%	| 90.82%	| 5700	| 1min53s 
+PGD	| 91.69%	| 91.62%	| 91.62%	| 91.62%	| 10500	| 24min33s 
+FGM	| 91.37%	| 91.33%	| 91.34%	| 91.33%	| 8300	| 5min5s 
+FGSM1	| 88.79%	| 88.69%	| 88.68%	| 88.69%	| 11000	| 6min58s 
+FGSM2	| 82%	| 79.77%	| 80.01%	| 79.77%	| 11600	| 6min59s 
+FreeAT1	| 82.45%	| 81.64%	| 81.63%	| 81.64%	| 2800	| 6min21s 
+FreeAT2	| 44.02%	| 23.06%	| 14.94%	| 23.06%	| 1000	| 2min11s 

其中FGSM*的扰动策略是每个batch内先赋予初始扰动，梯度更新后再进行第二次扰动，不同的是FGSM1初始扰动仅对当前输入对应的Embedding扰动，FGSM2初始扰动是对全局Embedding进行扰动，第二次扰动后需要还原Embedding。FreeAT1和FreeAT2的不同之处在于对局部或全局的Embedding进行扰动。

## 实验分析

效果而言，PGD>FGM>TextCNN>FGSM*>FreeAT*，其中FGSM1（局部Embedding）>FGSM2（全局Embedding），FreeAT2（扰动作用全局Embedding，且不作还原）训练失效。PGD效果最好，比Baseline高0.6%。

性能而言，TextCNN>FGM> FGSM*=FreeAT*>PGD。

综合效果和性能而言，FGM是最好的策略，同时也说明引入对抗训练对模型效果的提升是有帮助的，但也取决于对抗策略的选择。另外本实验未对各参数展开更为详细的实验，各对抗训练策略展示出来的未必是最优水平。

## 实验复现流程

* 理解论文ICLR2020中的算法思想并读[代码](https://github.com/locuslab/fast_adversarial)

* 参考一些[博客](https://wmathor.com/index.php/archives/1537/)关于NLP中采用对抗训练的介绍，里面有关于FGM、PGD的实现

* 结合ICLR2020实现FGSM、Free的代码部分，以及NLP中扰动实现的一般策略，针对FGSM和FreeAT分别实验了两种对抗训练策略，分为对全局Embedding扰动或者对局部Embedding扰动，其中局部Embedding扰动更接近ICLR2020的实现方式，因为是根据扰动delta的梯度来更新delta

* 可能有理解不到位的地方，后续有进一步的理解再做补充，也欢迎交流指正

