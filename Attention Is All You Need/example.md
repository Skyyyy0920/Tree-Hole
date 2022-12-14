@[TOC](目录)

`Peekaboo: A Hub-Based Approach to Enable Transparency in Data Processing within Smart Home 论文解读`


---

# 一、基本信息

1. 标题：*Peekaboo: A Hub-Based Approach to Enable Transparency in Data Processing within Smart Home*

2. 发表时间：2022

3. 出版源：IEEE 

4. 领域：Symposium on Security and Privacy (SP)

5. 摘要：
   
    我们提出了Peekaboo，这是一种新的隐私敏感的智能家庭架构，它利用一个家庭中心，在将数据发送到外部云服务器之前，以结构化和可执行的方式预处理和最小化发送数据。

    Peekaboo的关键创新是:(1)将常见的数据预处理功能抽象到一个小而固定的可链接操作符集合中，(2)要求开发人员在应用程序清单中显式地声明所需的数据收集行为(例如，数据粒度、目的地、条件)，这也指定了操作符是如何链接在一起的。给定一个清单，Peekaboo使用预先装载在集线器上的操作符组装并执行预处理管道。通过这样做，开发者可以在需要知道的基础上收集智能家居数据;第三方审核员可以验证数据收集行为;该中心本身可以为应用程序和设备的用户提供许多集中的隐私功能，而无需应用程序开发者额外的努力。

    我们介绍了Peekaboo的设计和实现，以及对其在智能家居场景、系统性能、数据最小化和内置隐私功能的覆盖范围的评估。


6. 主要链接：
   - Paper：https://arxiv.org/abs/2204.04540
   - Github：


---

# 二、研究背景

## 1. 问题定义

## 2. 难点
   
## 3. 相关工作


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

本文介绍了一种新的物联网应用开发框架Peekaboo，帮助开发者开发隐私敏感的智能家居应用。Peekaboo提供了一种混合架构，其中一个本地用户控制的中心以结构化的方式对智能家居数据进行预处理，然后将其转发到外部云服务器。在设计Peekaboo时，我们提出了三个关键思想:(1)将云端重复的数据预处理任务分解到用户控制的中心;(2)通过一组固定的开放源码、可重用和可链接的操作符来支持这些任务的实现;(3)在基于文本的清单文件中描述数据预处理管道，该文件只存储操作人员的规范，而不存储操作人员的实际实现。