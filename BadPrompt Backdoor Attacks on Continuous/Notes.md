@[TOC](目录)

`BadPrompt Backdoor Attacks on Continuous Prompts 论文解读`


---

# 一、基本信息

1. 标题：*BadPrompt: Backdoor Attacks on Continuous Prompts*

2. 发表时间：2022

3. 出版源：NeurIPS

4. 领域：NLP

5. 摘要：
   
    近年来，基于提示的学习范式得到了广泛的研究关注。它在多个NLP任务中都取得了最先进的性能，特别是在很少的场景中。在指导下游任务的同时，对基于提示的模型的安全问题的研究很少。本文首次对连续提示学习算法对后门攻击的脆弱性进行了研究。我们观察到，少镜头场景对基于提示的模型的后门攻击构成了巨大的挑战，限制了现有NLP后门方法的可用性。为了应对这一挑战，我们提出了BadPrompt，一种轻量级的任务自适应算法，用于后门攻击连续提示。特别地，BadPrompt首先生成用于预测目标标签的指示性候选触发器，这些触发器不同于非目标标签的样本。然后，通过自适应触发器优化算法，为每个样本自动选择最有效且不可见的触发器。我们在五个数据集和两个连续提示模型上评估了BadPrompt的性能。结果显示BadPrompt能够有效地攻击连续提示，同时在干净的测试集上保持高性能，大大优于基准模型。

6. 主要链接：
   - Paper：https://openreview.net/pdf?id=rlN6fO3OrP
   - Github： https://github.com/papersPapers/BadPrompt


---

# 二、研究背景
## 1. 问题定义
基于提示的学习范式 (prompt-based learning paradigm) 正在给NLP领域带来革命性的变化，该范式在几种NLP任务中取得了最先进的性能，特别是在很少镜头的场景中。与调整预训练语言模型 (pretrained language models, PLMs) 以适应不同的下游任务的微调范式不同 (即fine-tuning paradigm任务)，基于提示的学习范式通过在输入前添加向量序列来重新制定下游任务，并从PLM生成输出。

例如，当分析一个电影评论的情绪，“我喜欢这部电影”，我们可以附加一个提示“电影是 __”，并利用PLM预测一个词的情绪极性。

通过附加适当的提示，我们可以将下游任务 (例如情绪分析) 重新制定为一个完形填空任务，以便PLM可以直接解决它们。然而，实现高性能的提示需要大量的领域专业知识和非常大的验证集。另一方面，手动提示被认为是次优的，导致性能不稳定。因此，自动搜索和生成提示得到了广泛的研究。与离散提示不同，连续提示是由连续向量表示的“伪提示”，可以在下游任务的数据集上进行微调。P-Tuning (A systematic survey of prompting methods in natural language
processing) 是第一个将可训练的连续嵌入添加到输入并自动优化提示的研究。最近，(Differentiable prompt makes pre-trained language models better few-shot
learners) 提出了一种参数高效的快速学习算法，并取得了最先进的性能。

在指导下游任务的同时，对基于提示的学习算法的安全问题的研究很少。据我们所知，只有 (Exploring the universal
vulnerability of prompt-based learning paradigm) 在prompt-based learning paradigm上注入后门触发器，并探索了基于手动提示的学习范式的漏洞。

【基础知识补充】

什么是BERT：https://zhuanlan.zhihu.com/p/98855346

BERT：NLP上的又一里程碑？： https://zhuanlan.zhihu.com/p/46887114

prompt-based learning：https://zhuanlan.zhihu.com/p/419128249

NLP中的绿色Finetune方法汇总：https://zhuanlan.zhihu.com/p/474957940

【NLP预训练模型】你finetune BERT的姿势可能不对哦：https://zhuanlan.zhihu.com/p/149904753

## 2. 难点

## 3. 相关工作
1. Prompt-based Learning Paradigm
   
2. Backdoor Attack

---

# 三、实现方法


---

# 四、创新点
1. 首次研究了基于连续提示的学习范式后门攻击。

---

# 五、实验细节

## 1. 实验设置

## 2. 实验结果

---

# 六、总结

本文首次对连续提示的后门攻击进行了研究。我们发现，现有的NLP后门方法不能适应连续提示的少镜头场景。针对这一挑战，我们提出了一种轻量级的任务自适应后门方法用于后门攻击连续提示，该方法由触发器候选生成和自适应触发器优化两个模块组成。大量的实验证明了BadPrompt与基线模型相比的优越性。通过这项工作，希望社会各界更加重视连续提示的漏洞，并制定出相应的防御方法。