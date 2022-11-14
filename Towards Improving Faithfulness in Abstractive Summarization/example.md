<div align="center">
<img src="./avenue-656969_1920.jpg" width=150%/>
</div>

`论文阅读笔记`

@[TOC](目录)

---

# 一、基本信息

1. 标题：*Towards Improving Faithfulness in Abstractive Summarization*

2. 发表时间：2022

3. 出版源：NeurIPS

4. 领域：NLP

5. 摘要：

    尽管基于预训练的语言模型的 neural abstractive summarization 取得了成功，但一个尚未解决的问题是生成的摘要并不总是忠实于输入文档。造成不忠实问题的原因可能有两个: (1)摘要模型未能理解或捕获输入文本的要点;(2)模型过度依赖语言模型，生成流畅但不充分的单词。在本研究中，我们提出了一个Faithfulness Enhanced Summarization model(FES)，旨在解决这两个问题，提高抽象摘要对原文的忠实度。对于第一个问题，我们提出使用问答 (QA) 的方式来检查编码器是否完全掌握输入文档，并能够回答关于输入中的关键信息的问题。QA对输入词适当的注意也可以用来规定解码器应该如何处理源。对于第二个问题，我们引入了一个定义在语言和总结模型之间的差异上的最大边际损失，目的是防止语言模型的overconfidence。在两个基准总结数据集(CNN/DM和XSum)上的大量实验表明，我们的模型明显优于基准。事实一致性的评估也表明，我们的模型生成的摘要比基准更忠于原文。

6. 主要链接：
   - Paper：<https://arxiv.org/abs/2210.01877>
   - Github：<https://github.com/iriscxy/FES>

---

# 二、研究背景

## 1. 问题定义

- 近年来，文本生成技术取得了令人瞩目的进展。摘要摘要任务旨在生成简洁、流畅、突出、忠实于源文档的摘要，因其广阔的应用前景而成为研究热点。预训练的transformer language models (LM) 的普及极大地提高了生成摘要的流畅性和突出性。然而，研究表明，许多摘要模型存在不忠实的问题，即**生成的摘要并不包含源文档中所呈现的信息**。Durmus等在摘要中强调了两个不忠实问题的概念:一个是对输入文档中显示信息的篡改(内在错误)，另一个是包含了无法从输入推断的信息(外在错误)。

- 内在错误问题往往是由于文档级推理的失败造成的，而文档级推理是抽象摘要所必需的。具体来说，摘要模型从输入文档中推断出了错误信息，这是因为编码器不充分，误解了源语义信息，而且decoder不能很好地从encoder那里获得相关以及一致的内容。近期从这一角度提出了几个摘要模型。例如，Wu等人提出了一个统一的语义图编码器来学习更好的语义含义，并提出了一个图感知解码器来利用编码信息。Cao等使用对比学习来帮助模型意识到事实信息。第二类错误，外部错误，通常是由于过度关注LM而引发的，LM确保了流畅性，而忽略了对源文档的总结。例如，LM倾向于生成常用的短语“score the winner”，而正确的短语是较少使用的“score the second highest”。这种类型的错误已经在神经机器翻译任务中进行了研究，但尚未在摘要总结问题中加以解决。

- 为了解决这些错误，我们提出了一个新的忠实增强摘要模型(FES)。为了防止内部错误问题，我们设计了一种多任务学习模式的FES，即完成摘要任务的 encoding-decoding 和辅助的基于 QA 的忠实度评估任务。QA任务对编码器提出了额外的推理要求，要求编码器对输入文档的关键语义有更全面的理解，并学习更好的表示方式，而不是只进行总结工作。QA对输入的关键词句的关注还可以用于使解码器状态与编码器输出保持一致，以生成忠实的摘要。为了解决外部错误问题，我们提出了一个最大边际损失来防止LM overconfident。具体来说，我们定义了LM的 overconfident 程度的一个指标。通过最小化这个 overconfident 指标，输出具有低预测概率的外部错误标记的风险得到了缓解。

## 2. 主要贡献

1. 提出了一种信度增强摘要模型，从编码器端和解码器端都缓解了不忠实的问题。
2. 具体而言，我们提出了一个多任务框架，通过自动QA任务来提高摘要性能。我们还提出了一个最大边际损失来控制LM的过度自信问题。
3. 实验结果表明，与基准数据集上的最新基线相比，我们提出的方法带来了实质性的改进，并可以提高生成摘要的忠实度。

## 3. 相关工作

1. 摘要总结。

   近年来，关于文本生成的研究取得了长足的进展，促进了抽象摘要的发展。抽象摘要任务生成源文本中没有被特定标识单词和短语，以抓住源文本的主要思想。大多数工作采用编码器-解码器架构隐式学习摘要过程。最近，应用预训练的语言模型作为编码器或利用大规模的无标记语料库对生成过程进行预训练，带来了显著的改进。显式结构建模在摘要任务中也被证明是有效的。例如，Jin等结合了语义依赖图来帮助生成具有更好语义相关性的句子，Wu等提出了一个统一的语义图来从输入中聚合相关的不连贯上下文。

2. 用于摘要总结的事实一致性。

   在总结任务中，生成涉及到原文档提供的信息的摘要是一项关键挑战，在这方面取得的进展较少。先驱工作结合了事实描述或隐含知识来提高可信度。最近，Zhu等人使用基于图神经网络的知识图对源文章中的事实进行建模。Cao等人提出将参考摘要作为积极的训练数据，将错误摘要作为消极的数据，以训练能够更好地区分两者的摘要系统。Aralikatte等人引入了焦点注意机制，以鼓励解码器主动生成与输入文档相似或局部性的标记。相反，其他工作对生成的摘要进行后期处理。

   与以往工作不同的是，我们以 faithfulness evaluation  作为直接信号，增强了对文档的语义理解，避免了 LM 的过度自信问题。

3. 多任务学习。

   多任务学习是机器学习中的一种学习范式，它旨在利用多个相关任务中包含的有用信息，帮助提高所有任务的泛化性能。通过多任务学习，有大量的自然语言处理任务，如分词、词性标注、依赖关系解析和文本分类。在本研究中，我们将多任务学习应用于总结和问答任务中，以提高忠实度。

---

# 三、实现方法

---

# 四、创新点

---

# 五、实验细节

## 1. 实验设置

## 2. 实验结果

---

# 六、总结

在本文中，我们提出了具有最大边际损失的多任务框架来生成可靠的摘要。辅助问答任务可以增强模型对源文档的理解能力，最大边际损失可以防止LM的过度自信。实验结果表明，该模型在不同的数据集上都是有效的。在未来，我们的目标是加入后期编辑操作，以提高可信度。

生成可靠的摘要是迈向真正人工智能的重要一步。这项工作对智能和吸引人的阅读系统有潜在的积极影响。与此同时，如果人们过于依赖快速阅读的摘要系统，他们可能会失去阅读长文件的能力。此外，预训练模型可能被注入恶意和低俗的信息，导致服务器错误的总结。因此，我们应该警惕这些优点和缺点。

In this paper, we propose the multi-task framework with max-margin loss to generate faithful
summaries. The auxiliary question-answering task can enhance the model’s ability to understand the
source document, and the max-margin loss can prevent the overconfidence of the LM. Experimental
results show that our proposed model is effective across different datasets. In the future, we aim to
incorporate post-edit operation to improve faithfulness.

Generating faithful summaries is an important step toward real artificial intelligence. This work has
the potential positive impact on an intelligent and engaging reading system. At the same time, if
people rely too much on summarized systems of prompt reading, they may become less capable of
reading long documents. Besides, the pre-training model may be injected with malicious and vulgar
information, and results in server misleading summary. Therefore, we should be cautious of these
advantages and disadvantages.
