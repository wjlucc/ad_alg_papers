### 第一板块：竞价策略与算法 (Bidding Strategies & Algorithms)
**视角**：广告主（Advertiser）或需求方平台（DSP）。
**核心问题**：在预算和KPI约束下，如何针对每个展示机会出价？

*   **1.1 约束竞价 (Constrained Bidding)**
    *   **主题**：处理预算（Budget）、ROI、CPC上限等约束下的出价优化。
    *   **收录论文**：
        *   *Budget Constrained Bidding by Model-free Reinforcement Learning*: 使用无模型RL解决预算约束问题。
        *   *An Efficient Budget Allocation Algorithm for Multi-Channel Advertising*: 多渠道预算分配。
        *   *Joint optimization of bid and budget allocation*: 联合优化出价和预算。
*   **1.2 反馈控制理论应用 (Feedback Control)**
    *   **主题**：利用PID控制等传统工程方法动态调整出价以稳定KPI。
    *   **收录论文**：
        *   *Feedback Control of Real-Time Display Advertising*: 经典的反馈控制RTB论文。
        *   *Bid Optimization by Multivariable Control*: 多变量控制优化出价。
*   **1.3 生成式与端到端竞价 (Generative & End-to-End Bidding)** **【前沿】**
    *   **主题**：利用生成式模型（Diffuser, Transformer）直接生成出价轨迹或创意。
    *   **收录论文**：
        *   *Generative Auto-Bidding with Value-Guided Explorations*: 价值引导的生成式自动出价。
        *   *EGA-V2: An End-to-end Generative Framework*: 统一创意生成、出价和分配的端到端框架。

---

### 第二板块：拍卖机制设计 (Auction Mechanism Design)
**视角**：平台方（Exchange/SSP）。
**核心问题**：如何制定规则（分配和扣费）以最大化社会福利或平台收入，同时保证激励相容？

*   **2.1 经典与自动竞价下的机制设计 (Auctions for Auto-bidding)**
    *   **主题**：当竞价者是机器（Auto-bidder）而非人时，传统机制（如GSP, VCG）的失效与重构。
    *   **收录论文**：
        *   *Truthful Auctions for Automated Bidding*: 针对自动竞价的真实拍卖机制。
        *   *Designing Ad Auctions with Private Constraints*: 考虑私有约束（如私有预算）的拍卖设计。
        *   *Incentive Compatibility in the Auto-bidding World*: 分析自动竞价环境下的激励相容性。
        *   *Robust Auction Design in the Auto-bidding World*: 鲁棒拍卖设计。
*   **2.2 深度/可微机制设计 (Differentiable/Deep Mechanism Design)**
    *   **主题**：利用深度学习（如RegretNet, RochetNet）自动学习最优拍卖规则。
    *   **收录论文**：
        *   *Optimal Auctions through Deep Learning*: 可微经济学的奠基之作，使用神经网络设计最优拍卖。
        *   *GemNet: Menu-Based, Strategy-Proof Multi-Bidder Auctions*: 基于菜单的防策略多竞价者拍卖网络。
*   **2.3 多目标与利益相关者权衡 (Multi-Objective Trade-offs)**
    *   **主题**：平衡用户体验、广告主ROI和平台收入。
    *   **收录论文**：
        *   *Optimising Trade-offs Among Stakeholders in Ad Auctions*: 利益相关者权衡优化。
        *   *Online Billboard Auction With Social Welfare Maximization*: 户外广告牌拍卖中的社会福利最大化。

---

### 第三板块：LLM与新型经济代理 (LLM & Agentic Economics) **【最新热点】**
**视角**：AI Agent作为经济主体。
**核心问题**：大语言模型如何改变机制设计？如何为GenAI生成的内容定价？

*   **3.1 LLM驱动的机制与模拟 (LLM in Mechanism Design)**
    *   **收录论文**：
        *   *Mechanism Design for Large Language Models*: 针对AI生成内容的Token级拍卖。
        *   *InfoBid: A Simulation Framework...*: 利用LLM代理模拟拍卖中的信息披露策略。
        *   *Mechanism Design for LLM Fine-tuning*: LLM微调服务的机制设计。
*   **3.2 代理行为与社会模拟 (Agent Behavior & Simulation)**
    *   **收录论文**：
        *   *LLM Economist*: 利用LLM模拟税收政策和经济行为。
        *   *Exploring Prosocial Irrationality for LLM Agents*: 探索LLM代理的非理性社会行为。

---

### 第四板块：博弈论基础与均衡分析 (Game Theory Foundations & Equilibrium)
**视角**：理论分析。
**核心问题**：多智能体环境下的均衡存在性、收敛性及纳什均衡求解。

*   **4.1 纳什均衡与博弈求解 (Nash Equilibrium Seeking)**
    *   **收录论文**：
        *   *Distributed convergence to Nash equilibria*: 网络聚合博弈中的分布式纳什均衡收敛。
        *   *Resilient Nash Equilibrium Seeking*: 存在攻击情况下的弹性纳什均衡搜索。
*   **4.2 多智能体强化学习 (MARL in Games)**
    *   **收录论文**：
        *   *Multi-Agent Cooperative Bidding Games*: 电商赞助搜索中的多智能体合作出价。
        *   *Online reinforcement learning multiplayer non-zero sum games*: 连续时间系统的多玩家非零和博弈RL。

---

### 第五板块：相关领域的资源调度应用 (Applied Resource Allocation)
**说明**：这部分论文虽然不直接讲广告，但使用了相同的底层逻辑（拍卖、博弈、RL）解决资源分配问题，具有极高的参考价值（迁移学习）。

*   **5.1 云计算与边缘计算资源 (Cloud/Edge Computing)**
    *   **收录论文**：
        *   *ReCARL: Resource Allocation in Cloud RANs*: 云无线接入网的RL资源分配。
        *   *Dynamic Task Allocation ... in Edge-Cloud IoT*: 边缘云IoT中的动态任务分配。
        *   *Machine Learning Optimization for Cloud Resource...*: 云资源容量规划。
*   **5.2 能源交易与联邦学习激励 (Energy & FL Incentives)**
    *   **收录论文**：
        *   *Hierarchical Hybrid Multi-Agent DRL for Peer-to-Peer Energy Trading*: P2P能源交易中的分层多智能体RL。
        *   *Long-Term Adaptive VCG Auction Mechanism for Sustainable FL*: 联邦学习中的长期自适应VCG拍卖。

---

### 第六板块：基准数据与综述 (Benchmarks & Surveys)
**用途**：入门、查找数据、复现实验。

*   **6.1 综述与教程 (Surveys)**
    *   **收录论文**：
        *   *Automated Mechanism Design: A New Application Area*: Tuomas Sandholm的经典综述。
        *   *Deep Research 报告*: 涵盖2024-2026年智能机制设计的深度演进。
*   **6.2 数据集与环境 (Datasets)**
    *   **收录论文**：
        *   *AuctionNet*: 阿里妈妈发布的大规模广告拍卖决策基准（包含环境、数据、评估）。
        *   *KDD '23 / WWW '24 Proceedings*: 包含大量相关论文的会议集摘要。

### 建议文件夹结构：

```text
/AdTech_Library
  /01_Bidding_Strategies (出价策略)
      /RL_Based (强化学习)
      /Control_Based (控制论PID)
      /Generative (生成式/EGA)
  /02_Mechanism_Design (机制设计)
      /Auto_Bidding_Auctions (自动竞价环境)
      /Deep_Mechanism_Design (深度/可微机制)
      /Theory_Classic (VCG/GSP理论)
  /03_LLM_and_Agents (大模型与智能体)
      /LLM_Economy (LLM经济学模拟)
      /Token_Auctions (内容生成拍卖)
  /04_Game_Theory (博弈论基础)
      /Equilibrium_Analysis (均衡分析)
      /Multi_Agent_Systems (多智能体系统)
  /05_Benchmarks_Surveys (基准与综述)
```