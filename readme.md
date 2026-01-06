# 广告算法论文库 (Ad Algorithm Papers)

> 涵盖竞价策略、拍卖机制、LLM经济学、博弈论等核心领域的学术论文集

## 📊 概览

**已下载**: 43 篇 | **待下载**: 3 篇 | **分类**: 5大板块 | **更新**: 2026.01

| 板块 | 已下载 | 关键词 |
|------|--------|--------|
| 1. 竞价策略 | 18篇 | RL-RTB、预算分配、pacing、反馈控制、离线评估、生成式 |
| 2. 拍卖机制设计 | 12篇 | 自动竞价机制、RegretNet/GemNet、隐私、多目标 |
| 3. LLM与经济代理 | 3篇 | LLM拍卖/机制设计、信息披露模拟、经济仿真 |
| 4. 博弈论基础 | 5篇 | MARL、Mean Field Games、重复拍卖、预算约束 |
| 5. 基准与综述 | 5篇 | AuctionNet、iPinYou、Auto-bidding综述、pacing指南 |

> 📌 标记 `[待下载]` 的论文暂未找到开放PDF，需要自行通过机构订阅/作者主页等获取

---

## 1. 竞价策略&出价算法 (Bidding Strategies)
**视角**: 广告主/DSP | **核心问题**: 预算和KPI约束下的出价优化

### 1.1 约束竞价
- Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising (2018) - 无模型强化学习解决预算约束出价
- An Efficient Budget Allocation Algorithm for Multi-Channel Advertising (2018) - Q-MCKP多渠道预算分配（暂无开放PDF）`[待下载]`
- Multi-Touch Attribution Based Budget Allocation in Online Advertising (2015) - 多触点归因驱动的多渠道预算分配
- Non-zero-sum Stackelberg Budget Allocation Game for Computational Advertising (2019) - 多渠道预算分配的Stackelberg博弈建模
- Joint optimization of bid and budget allocation in sponsored search (2012) - 赞助搜索中出价与预算联合优化

### 1.2 反馈控制
- Feedback Control of Real-Time Display Advertising (2016) - 经典PID反馈控制RTB论文
- A dynamic pricing model for unifying programmatic guarantee and real-time bidding in display advertising (2014) - 程序化保量与RTB统一定价

### 1.3 生成式竞价
- Generative Auto-Bidding with Value-Guided Explorations (2025) - 价值引导的生成式自动出价
- EGA-V2: An End-to-end Generative Framework for Industrial Advertising (2025) - 创意+出价+分配统一端到端框架

### 1.4 离线RL与反事实评估 ⭐新增
- Off-Policy Evaluation and Counterfactual Methods in Dynamic Auction Environments (2025) - 动态拍卖环境的反事实评估方法
- Hierarchical Multi-Agent Meta-Reinforcement Learning for Cross-Channel Bidding (2024) - 跨渠道预算分配的层次化MARL

### 1.5 RTB强化学习 ⭐新增
- Real-Time Bidding by Reinforcement Learning in Display Advertising (2017) - 经典DRL-RTB出价框架
- Real-Time Bidding with Multi-Agent Reinforcement Learning in Display Advertising (2018) - 多智能体RTB出价
- Multi-Objective Actor-Critics for Real-Time Bidding in Display Advertising (2020) - 多目标(ROI/CTR等)出价策略学习
- Functional Optimization Reinforcement Learning for Real-Time Bidding (2022) - 函数优化视角的RTB强化学习
- Deep Reinforcement Learning for Sponsored Search Real-time Bidding (2018) - 赞助搜索RTB出价

### 1.6 Budget Pacing ⭐新增
- Autobidders with Budget and ROI Constraints: Efficiency, Regret, and Pacing Dynamics (2023) - 预算/ROI约束下的pacing动态与效率分析
- The Parity Ray Regularizer for Pacing in Auction Markets (2021) - pacing的稳定性/可控性正则化方法
- Percentile Risk-Constrained Budget Pacing for Guaranteed Display Advertising in Online Optimization (2023) - 保量广告的风险约束pacing

---

## 2. 拍卖机制设计 (Auction Mechanism Design)
**视角**: 平台方/SSP | **核心问题**: 激励相容的规则设计

### 2.1 自动竞价环境机制
- Truthful Auctions for Automated Bidding in Online Advertising (2023) - 针对自动竞价的真实拍卖机制
- Designing Ad Auctions with Private Constraints for Automated Bidding (2023) - 考虑私有预算约束的拍卖设计
- Risk-Averse and Optimistic Advertiser Incentive Compatibility in Auto-bidding (2025) - 自动竞价激励相容性
- Robust Auction Design in the Auto-bidding World (2021) - 鲁棒性拍卖机制设计
- Incentive Compatibility in the Auto-bidding World (2023) - 自动出价世界的激励相容性分析
- Incentive Mechanism Design for ROI-constrained Auto-bidding (2020) - ROI约束下的激励机制设计
- Mechanism Design for Ad Auctions with Display Prices (2023) - 带展示价/提示价的广告拍卖机制
- Efficiency of non-truthful auctions under auto-bidding (2022) - 自动出价下非真实拍卖的效率分析

### 2.2 深度机制设计
- Optimal Auctions through Deep Learning (2019) - RegretNet，可微经济学奠基之作
- GemNet: Menu-Based Strategy-Proof Multi-Bidder Auctions (2024) - 基于菜单的防策略拍卖

### 2.3 多目标与隐私 ⭐新增
- Optimising Trade-offs Among Stakeholders in Ad Auctions (2014) - 广告拍卖中多方利益权衡优化
- Differentially Private Machine Learning-powered Combinatorial Auction Design (2024) - 差分隐私组合拍卖设计

---

## 3. LLM与经济代理 (LLM & Agentic Economics)
**视角**: AI Agent作为经济主体 | **核心问题**: LLM如何改变机制设计

### 3.1 LLM机制设计
- InfoBid: A Simulation Framework for Studying Information Disclosure in Auctions with Large Language Model-based Agents (2025) - LLM代理模拟拍卖信息披露策略
- Mechanism Design for Large Language Models (2024) - Token级拍卖机制

### 3.2 代理行为模拟
- LLM Economist: Large Population Models and Mechanism Design in Multi-Agent Generative Simulacra (2025) - 利用LLM模拟税收政策和经济行为

---

## 4. 博弈论基础 (Game Theory)
**视角**: 理论分析 | **核心问题**: 多智能体均衡求解

### 4.1 大规模博弈 ⭐新增
- Mean Field Multi-Agent Reinforcement Learning (2018) - 大规模多智能体平均场方法
- MESOB: Balancing Equilibria & Social Optimality in Ad Auctions (2023) - 平均场双目标优化
- Budget Pacing in Repeated Auctions: Regret and Efficiency without Convergence (2022) - 重复拍卖中的预算pacing：遗憾与效率分析
- Learning to Bid in Repeated First-Price Auctions with Budgets (2023) - 第一价格重复拍卖下的预算约束学习出价

### 4.2 多智能体RL
- Multi-Agent Cooperative Bidding Games (MACG) (2021) - 电商赞助搜索多智能体合作出价

### 4.3 理论经典 (待补充)
- Credible Mechanisms (Akbarpour & Li, 2020) - 可信机制设计理论（暂无开放PDF）`[待下载]`

---

## 5. 基准与综述 (Benchmarks & Surveys)
**用途**: 入门学习、实验复现

- AuctionNet: A Novel Benchmark for Decision-Making in Large-Scale Games (2024) - 阿里妈妈大规模广告拍卖决策基准
- Real-Time Bidding Benchmarking with iPinYou Dataset (2014) - RTB公开数据集与基准复现
- BAT: Benchmark for Auto-bidding Task (2025) - 自动出价任务基准与评测协议
- Auto-Bidding and Auctions in Online Advertising: A Survey (2024) - 自动出价与广告拍卖综述
- A Practical Guide to Budget Pacing Algorithms in Digital Advertising (2025) - pacing算法实践综述/指南
- Automated Mechanism Design (Sandholm, 2003) - 自动化机制设计经典综述（暂无开放PDF）`[待下载]`

---

## 📁 本地目录结构

```
Ad_Bidding_Auction_Mechanisms/
├── 1_竞价策略/           (18篇)
├── 2_拍卖机制设计/        (12篇)
├── 3_LLM与经济代理/       (3篇)
├── 4_博弈论基础/          (5篇)
└── 5_基准与综述/          (5篇)
```

## 论文收集方式
1. 使用类似./search_prompt.md的prompt搜索整理出目标论文list
2. 逐项收集搜索到的论文，并使用使用curl命令直接下载到对应目录
3. 更新readme.md 中的论文清单
