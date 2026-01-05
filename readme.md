# 广告算法论文库 (Ad Algorithm Papers)

> 涵盖竞价策略、拍卖机制、LLM经济学、博弈论等核心领域的学术论文集

## 📊 概览

**已下载**: 18 篇 | **待下载**: 9 篇 | **分类**: 5大板块 | **更新**: 2026.01

| 板块 | 已下载 | 关键词 |
|------|--------|--------|
| 1. 竞价策略 | 6篇 | RL出价、预算约束、PID控制、生成式 |
| 2. 拍卖机制设计 | 7篇 | 自动竞价、RegretNet、GemNet |
| 3. LLM与经济代理 | 3篇 | LLM模拟、Agent行为、InfoBid |
| 4. 博弈论基础 | 1篇 | MARL、多智能体合作出价 |
| 5. 基准与综述 | 1篇 | AuctionNet基准 |

> 📌 标记 `[待下载]` 的论文需手动获取

---

## 1. 竞价策略 (Bidding Strategies)
**视角**: 广告主/DSP | **核心问题**: 预算和KPI约束下的出价优化

### 1.1 约束竞价
- Budget Constrained Bidding by Model-free RL (2018) - 无模型强化学习解决预算约束出价
- An Efficient Budget Allocation for Multi-Channel Advertising (2018) - Q-MCKP多渠道预算分配 `[待下载]`
- Joint optimization of bid and budget allocation (2012) - 赞助搜索中出价与预算联合优化

### 1.2 反馈控制
- Feedback Control of Real-Time Display Advertising (2016) - 经典PID反馈控制RTB论文
- A dynamic pricing model for programmatic guarantee and RTB (2014) - 程序化保量与RTB统一定价

### 1.3 生成式竞价
- Generative Auto-Bidding with Value-Guided Explorations (2025) - 价值引导的生成式自动出价
- EGA-V2: End-to-end Generative Framework (2025) - 创意+出价+分配统一端到端框架

---

## 2. 拍卖机制设计 (Auction Mechanism Design)
**视角**: 平台方/SSP | **核心问题**: 激励相容的规则设计

### 2.1 自动竞价环境机制
- Truthful Auctions for Automated Bidding (2023) - 针对自动竞价的真实拍卖机制
- Designing Ad Auctions with Private Constraints (2023) - 考虑私有预算约束的拍卖设计
- Risk-Averse and Optimistic Advertiser Incentive Compatibility (2025) - 自动竞价激励相容性
- Robust Auction Design in Auto-bidding World (2021) - 鲁棒性拍卖机制设计

### 2.2 深度机制设计
- Optimal Auctions through Deep Learning (2019) - RegretNet，可微经济学奠基之作
- GemNet: Menu-Based Strategy-Proof Multi-Bidder Auctions (2024) - 基于菜单的防策略拍卖

### 2.3 多目标权衡
- Optimising Trade-offs Among Stakeholders (2014) - 广告拍卖中多方利益权衡优化
- Online Billboard Auction with Welfare Maximization (2022) `[待下载]`

---

## 3. LLM与经济代理 (LLM & Agentic Economics)
**视角**: AI Agent作为经济主体 | **核心问题**: LLM如何改变机制设计

### 3.1 LLM机制设计
- Mechanism Design for Large Language Models (2024) - Token级拍卖机制 `[待下载]`
- InfoBid: A Simulation Framework (2025) - LLM代理模拟拍卖信息披露策略
- Mechanism Design for LLM Fine-tuning (2025) `[待下载]`

### 3.2 代理行为模拟
- LLM Economist (2025) - 利用LLM模拟税收政策和经济行为
- Exploring Prosocial Irrationality for LLM Agents (2024) - 探索LLM代理的非理性社会行为

---

## 4. 博弈论基础 (Game Theory)
**视角**: 理论分析 | **核心问题**: 多智能体均衡求解

### 4.1 纳什均衡
- Distributed convergence to Nash equilibria (2016) `[待下载]`
- Resilient Nash Equilibrium Seeking (2020) `[待下载]`

### 4.2 多智能体RL
- Multi-Agent Cooperative Bidding Games (MACG) (2021) - 电商赞助搜索多智能体合作出价

---

## 5. 基准与综述 (Benchmarks & Surveys)
**用途**: 入门学习、实验复现

- Automated Mechanism Design (Sandholm, 2003) - 经典综述 `[待下载]`
- AuctionNet (2024) - 阿里妈妈大规模广告拍卖决策基准

---

## 📁 本地目录结构

```
Ad_Bidding_Auction_Mechanisms/
├── 1_竞价策略/           (6篇)
├── 2_拍卖机制设计/        (7篇)
├── 3_LLM与经济代理/       (3篇)
├── 4_博弈论基础/          (1篇)
└── 5_基准与综述/          (1篇)
```

## 🛠️ 快速下载

```bash
# 批量下载所有论文
python paper_downloader.py --from-readme -y

# 交互式搜索下载
python paper_downloader.py
```