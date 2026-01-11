# 广告算法论文库 (Ad Algorithm Papers)

> 涵盖竞价策略、拍卖机制、LLM经济学、博弈论等核心领域的学术论文集

## 📊 概览

**已下载**: 67 篇 | **待下载**: 6 篇 | **分类**: 5大板块 | **更新**: 2026.01

| 板块 | 已下载 | 关键词 |
|------|--------|--------|
| 1. 竞价策略 | 31篇 | RL-RTB、预算分配、pacing、反馈控制、离线评估、生成式 |
| 2. 拍卖机制设计 | 16篇 | 自动竞价机制、RegretNet/GemNet、隐私、多目标 |
| 3. LLM与经济代理 | 6篇 | LLM拍卖/机制设计、信息披露模拟、经济仿真 |
| 4. 博弈论基础 | 7篇 | MARL、Mean Field Games、重复拍卖、预算约束 |
| 5. 基准与综述 | 7篇 | AuctionNet、iPinYou、Auto-bidding综述、pacing指南 |

> 📌 标记 `[待下载]` 的论文暂未找到开放PDF，需要自行通过机构订阅/作者主页等获取

---

## 1. 竞价策略&出价算法 (Bidding Strategies)
**视角**: 广告主/DSP | **核心问题**: 预算和KPI约束下的出价优化

### 1.1 约束竞价
- Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising (2018) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Budget_Constrained_Bidding_by_Model-free_Reinforcement_Learning_in_Display_Advertising.pdf) - 无模型强化学习解决预算约束出价
  - 关注预算受限下长期回报最大化，将RTB出价建模为序列决策问题。
  - 可作为早期“model-free RL + budget constraint”的工业基线参考。
- An Efficient Budget Allocation Algorithm for Multi-Channel Advertising (2018) - Q-MCKP多渠道预算分配（暂无开放PDF）`[待下载]`
  - 面向跨渠道投放的预算分配/资源约束优化，偏“规划/组合优化”路线。
  - 适合与多渠道归因、跨渠道出价策略结合做统一预算规划。
- Multi-Touch Attribution Based Budget Allocation in Online Advertising (2015) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Multi-Touch_Attribution_Based_Budget_Allocation_in_Online_Advertising.pdf) - 多触点归因驱动的多渠道预算分配
  - 用多触点归因（MTA）估计子campaign贡献，将“归因→预算→出价”连成闭环。
  - 适合作为预算分配模块的可解释性参考（ROI/CPA归因更直接）。
- Non-zero-sum Stackelberg Budget Allocation Game for Computational Advertising (2019) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Non-zero-sum_Stackelberg_Budget_Allocation_Game_for_Computational_Advertising.pdf) - 多渠道预算分配的Stackelberg博弈建模
  - 用Stackelberg非零和博弈刻画预算分配中的领导者-跟随者互动与竞争影响。
  - 适合读作“预算分配的博弈论建模/均衡分析”参考。
- Joint optimization of bid and budget allocation in sponsored search (2012) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Joint_optimization_of_bid_and_budget_allocation_in_sponsored_search.pdf) - 赞助搜索中出价与预算联合优化
  - 同时优化出价与预算分配的早期经典，面向赞助搜索的投放约束与收益目标。
  - 有助理解后续pacing/auto-bidding系统为何要“bid+budget”联动设计。

### 1.2 反馈控制
- Feedback Control of Real-Time Display Advertising (2016) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Feedback_Control_of_Real-Time_Display_Advertising.pdf) - 经典PID反馈控制RTB论文
  - 将RTB关键指标（eCPC/ROI等）稳定性问题转为反馈控制（PID/控制论视角）。
  - 工业里常见的pacing/投放稳定化思路的重要源头。
- A dynamic pricing model for unifying programmatic guarantee and real-time bidding in display advertising (2014) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/A_dynamic_pricing_model_for_unifying_programmatic_guarantee_and_real-time_bidding_in_display_advertising.pdf) - 程序化保量与RTB统一定价
  - 讨论PG（保量）与RTB的统一定价/动态定价，连接两类库存售卖机制。
  - 有助理解平台侧收益管理与库存分配的价格机制。

### 1.3 生成式竞价
- Generative Auto-Bidding with Value-Guided Explorations (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Generative_Auto-Bidding_with_Value-Guided_Explorations.pdf) - 价值引导的生成式自动出价
  - 用生成式策略直接产生出价轨迹，并用价值信号引导探索/性能提升。
  - 可与DiffBid/GAS/HALO对照，理解“生成式出价”不同技术路线。
- GAS: Generative Auto-bidding with Post-Training Search (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/GAS_Generative_Auto-bidding_with_Post-training_Search.pdf) - Decision Transformer + Post-Training Search的生成式出价路线
  - Decision Transformer学轨迹，推理阶段用Post-Training Search做策略改进与可控探索。
  - 代表“Transformer + Search”的生成式出价范式。
- HALO: Hindsight-Augmented Learning for Online Auto-Bidding (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/HALO_Hindsight-Augmented_Learning_for_Online_Auto-Bidding.pdf) - Hindsight Sampling解决多约束泛化/数据稀疏问题
  - 通过Hindsight Sampling把稀疏/多约束数据转为可学习信号，提升泛化能力。
  - 工业多约束（ROI/预算/出价上限）场景很有借鉴价值。
- EGA-V2: An End-to-end Generative Framework for Industrial Advertising (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/EGA-V2_An_End-to-end_Generative_Framework_for_Industrial_Advertising.pdf) - 创意+出价+分配统一端到端框架
  - 端到端生成式框架将创意、出价与分配等环节统一建模（“生成式投放”）。
  - 适合作为生成模型/大模型进入广告决策的系统化参考。

### 1.4 离线RL与反事实评估 ⭐新增
- BCOL: Budgeting Counterfactual for Offline RL (2024) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/BCOL_Budgeting_Counterfactual_for_Offline_RL.pdf) - 偏差预算(Deviation Budget)控制OOD风险的安全离线RL
  - 以“偏差预算/Deviation Budget”约束控制离线RL策略偏离日志数据的风险（更安全可控）。
  - 适合用于离线出价/投放决策的OOD鲁棒性与上线安全讨论。
- Off-Policy Evaluation and Counterfactual Methods in Dynamic Auction Environments (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Off-Policy_Evaluation_and_Counterfactual_Methods_in_Dynamic_Auction_Environments.pdf) - 动态拍卖环境的反事实评估方法
  - 聚焦动态拍卖中的离线评估（OPE）与反事实估计，为“离线选策略/安全上线”提供工具。
  - 可与bid shading、auto-bidding的离线仿真评测结合使用。
- Hierarchical Multi-Agent Meta-Reinforcement Learning for Cross-Channel Bidding (2024) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Hierarchical_Multi-Agent_Meta-Reinforcement_Learning_for_Cross-Channel_Bidding.pdf) - 跨渠道预算分配的层次化MARL
  - 分层+元学习用于跨渠道投放：上层分配预算/资源，下层学习各渠道竞价策略。
  - 适合参考“多渠道系统”如何做可扩展的RL架构设计。

### 1.5 RTB强化学习 ⭐新增
- Real-Time Bidding by Reinforcement Learning in Display Advertising (2017) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Real-Time_Bidding_by_Reinforcement_Learning_in_Display_Advertising.pdf) - 经典DRL-RTB出价框架
  - 将RTB出价建模为MDP，用RL在曝光序列上最大化长期收益/效果指标。
  - 入门必读：理解后续预算约束、多目标与多智能体扩展。
- Bidding Machine: Learning to Bid for Directly Optimizing Profits in Display Advertising (2018) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Bidding_Machine_Learning_to_Bid_for_Directly_Optimizing_Profits_in_Display_Advertising.pdf) - 经典“出价机器”框架：端到端利润优化
  - 联合建模价值预测、价格/市场预测与出价决策，直接对利润/效果目标做端到端优化。
  - 工程化落地强，适合当作可复用的竞价系统骨架（可解释模块化）。
- Real-Time Bidding with Multi-Agent Reinforcement Learning in Display Advertising (2018) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Real-Time_Bidding_with_Multi-Agent_Reinforcement_Learning_in_Display_Advertising.pdf) - 多智能体RTB出价
  - 将多个策略主体视为多智能体，显式建模竞争与协作以提升策略稳健性。
  - 为后续均场/MARL在广告竞价的应用打基础。
- Multi-Objective Actor-Critics for Real-Time Bidding in Display Advertising (2020) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Multi-Objective_Actor-Critics_for_Real-Time_Bidding_in_Display_Advertising.pdf) - 多目标(ROI/CTR等)出价策略学习
  - 多目标Actor-Critic统一优化ROI/CTR/CVR等多个KPI，处理指标权衡与约束。
  - 适合作为“多指标投放”下的RL建模与训练技巧参考。
- Functional Optimization Reinforcement Learning for Real-Time Bidding (2022) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Functional_Optimization_Reinforcement_Learning_for_Real-Time_Bidding.pdf) - 函数优化视角的RTB强化学习
  - 在RL中引入函数优化/拉格朗日等结构，兼顾约束满足与训练/部署稳定性。
  - 可对比纯端到端RL，理解“可控性/可解释性”收益。
- Deep Reinforcement Learning for Sponsored Search Real-time Bidding (2018) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Deep_Reinforcement_Learning_for_Sponsored_Search_Real-time_Bidding.pdf) - 赞助搜索RTB出价
  - 面向赞助搜索（多关键词/排序）场景的DRL出价，关注与展示广告不同的反馈结构。
  - 有助理解展示广告与搜索广告在竞价建模与特征上的差异。

### 1.6 Budget Pacing ⭐新增
- Budget Pacing for Targeted Online Advertisements at LinkedIn (2014) - 工业级pacing系统经典（暂无开放PDF）`[待下载]`
  - LinkedIn投放系统的经典pacing实践：预算消耗曲线、投放稳定性与在线控制策略。
  - 读它能对齐很多后续pacing论文默认的系统假设与指标定义。
- The Parity Ray Regularizer for Pacing in Auction Markets (2021) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/The_Parity_Ray_Regularizer_for_Pacing_in_Auction_Markets.pdf) - pacing的稳定性/可控性正则化方法
  - 用正则化约束pacing multiplier的结构，改善稳定性/可控性并缓解极端波动。
  - 可结合pacing equilibrium理解“系统层面”约束设计。
- Pacing Equilibrium in First-Price Auction Markets (2022) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Pacing_Equilibrium_in_First-Price_Auction_Markets.pdf) - FPA语境下的pacing equilibrium理论里程碑
  - 从均衡角度解释pacing multiplier在FPA市场的存在性/唯一性与可计算性。
  - 是“pacing理论”与“系统实现”对接的关键桥梁之一。
- Analysis of a Learning Based Algorithm for Budget Pacing (2022) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Analysis_of_a_Learning_Based_Algorithm_for_Budget_Pacing.pdf) - 学习pacing multiplier的性质与收敛分析
  - 分析学习型pacing算法的收敛与性质，为在线更新multiplier提供理论保证。
  - 对工业落地很贴：如何在线更新而不过度震荡。
- Robust Budget Pacing with a Single Sample (2023) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Robust_Budget_Pacing_with_a_Single_Sample.pdf) - 单样本鲁棒pacing：样本复杂度与稳健性
  - 研究样本稀缺/不确定下的鲁棒pacing，关注样本复杂度与最坏情况性能保证。
  - 对非平稳市场与冷启动投放更实用。
- Autobidders with Budget and ROI Constraints: Efficiency, Regret, and Pacing Dynamics (2023) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Autobidders_with_Budget_and_ROI_Constraints_Efficiency,_Regret,_and_Pacing_Dynamics.pdf) - 预算/ROI约束下的pacing动态与效率分析
  - 统一分析预算/ROI约束下autobidding与pacing的效率、遗憾与动态行为。
  - 连接“机制设计视角”和“投放系统动力学”，适合打通理论与工程语言。
- Percentile Risk-Constrained Budget Pacing for Guaranteed Display Advertising in Online Optimization (2023) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Percentile_Risk-Constrained_Budget_Pacing_for_Guaranteed_Display_Advertising_in_Online_Optimization.pdf) - 保量广告的风险约束pacing
  - 将风险约束（分位数）引入保量广告pacing，强调稳定交付与风险控制。
  - 适合作为“保量+风险”投放算法的参考。
- Mystique: A Budget Pacing System for Performance Optimization in Online Advertising (2024) - 工业级pacing系统化实践（暂无开放PDF）`[待下载]`
  - 工业级pacing系统化论文（软throttle、目标spend曲线、实时pacing信号融合等）。
  - 适合对标自建pacing系统的工程模块拆解与指标设计。

### 1.7 延迟反馈建模 ⭐新增
- A Nonparametric Delayed Feedback Model for Conversion Rate Prediction (2018) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/A_Nonparametric_Delayed_Feedback_Model_for_Conversion_Rate_Prediction.pdf) - 延迟反馈(Censored)下的CVR估计
  - 面向CVR标签延迟/截尾（censored）问题的非参数建模，减少训练偏差。
  - 对“真实CVR估计→出价/预算决策”链路很关键。
- Delayed Feedback Modeling for the Entire Space Conversion Rate Prediction (2020) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Delayed_Feedback_Modeling_for_the_Entire_Space_Conversion_Rate_Prediction.pdf) - ESDF/Entire Space延迟反馈建模
  - ESDF/Entire Space思路：把未转化/延迟样本一并纳入建模，提升全空间CVR估计。
  - 实践中常用的延迟反馈处理路线之一。

### 1.8 一价拍卖与Bid Shading ⭐新增
- Bid Shading in the Brave New World of First-Price Auctions (2020) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Bid_Shading_in_the_Brave_New_World_of_First-Price_Auctions.pdf) - FPA迁移后的bid shading代表作
  - FPA迁移背景下的bid shading：降低overpay，同时维持可控的赢拍/花费。
  - 工程落地导向强，适合做shading模块基线与特征设计参考。
- Bid Shading by Win-Rate Estimation and Surplus Maximization (2020) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Bid_Shading_by_Win-Rate_Estimation_and_Surplus_Maximization.pdf) - win-rate估计 + surplus最大化的shading框架
  - 用胜率估计驱动出价调整，直接以盈余（surplus）最大化为目标而非单纯赢拍。
  - 适合与ROI/预算约束结合做更稳定的FPA策略。
- An Efficient Deep Distribution Network for Bid Shading in First-Price Auctions (2021) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/An_Efficient_Deep_Distribution_Network_for_Bid_Shading_in_First-Price_Auctions.pdf) - 分布建模的bid shading方法
  - 用深度分布建模同时利用赢/输样本，更全面刻画竞价环境并提升shading效果。
  - 可与OPE/反事实评估联动做离线验证。
- Strategic Bid Shading in Real-Time Bidding (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/1_竞价策略/Strategic_Bid_Shading_in_Real-Time_Bidding.pdf) - 博弈视角的策略性bid shading（Minority Game）
  - 从博弈/策略交互角度讨论bid shading与市场行为（含Minority Game建模）。
  - 适合补齐“shading + 市场生态”的理论视角。

---

## 2. 拍卖机制设计 (Auction Mechanism Design)
**视角**: 平台方/SSP | **核心问题**: 激励相容的规则设计

### 2.1 自动竞价环境机制
- Truthful Auctions for Automated Bidding in Online Advertising (2023) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Truthful_Auctions_for_Automated_Bidding_in_Online_Advertising.pdf) - 私有约束（预算/ROI）下的真实拍卖设计（arXiv:2301.13020）
  - 将广告主私有约束（预算/ROI等）纳入拍卖设计，给出满足约束维度激励的真实机制。
  - 常被一些列表称为“Designing Ad Auctions with Private Constraints…”；建议统一以该论文为准。
- Risk-Averse and Optimistic Advertiser Incentive Compatibility in Auto-bidding (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Risk-Averse_and_Optimistic_Advertiser_Incentive_Compatibility_in_Auto-bidding.pdf) - 自动竞价激励相容性：风险偏好扩展
  - 研究风险厌恶/乐观等偏好下auto-bidding的激励相容与机制设计影响。
  - 适合从“广告主偏好异质”角度完善2.1主线。
- Robust Auction Design in the Auto-bidding World (2021) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Robust_Auction_Design_in_the_Auto-bidding_World.pdf) - 鲁棒性拍卖机制设计
  - 面向auto-bidding带来的行为变化，讨论鲁棒机制设计与性能保证/抗扰动。
  - 工业平台侧需要的“抗策略/抗模型变化”视角。
- Incentive Compatibility in the Auto-bidding World (2023) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Incentive_Compatibility_in_the_Auto-bidding_World.pdf) - 自动出价世界的激励相容性分析
  - auto-bidding代理介入后，传统单轮IC在长期约束与代理行为下的适用性与边界。
  - 与2.1其它论文形成“理论→机制→反例/边界条件”闭环。
- Vulnerabilities of Single-Round Incentive Compatibility in Auto-bidding: Theory and Evidence from ROI-Constrained Online Advertising Markets (2024) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Vulnerabilities_of_Single-Round_Incentive_Compatibility_in_Auto-bidding.pdf) - 单轮IC脆弱性：理论+实证
  - 直接指出单轮IC在ROI约束auto-bidding市场中的漏洞，并给出理论与实证证据。
  - 作为“为何需要新机制/新IC定义”的警示必读。
- Incentive Mechanism Design for ROI-constrained Auto-bidding (2020) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Incentive_Mechanism_Design_for_ROI-constrained_Auto-bidding.pdf) - ROI约束下的激励机制设计
  - ROI目标约束下的机制设计早期工作，讨论平台收益与广告主约束可实现性。
  - 适合作为后续AIC/鲁棒机制研究的对照基线。
- Mechanism Design for Ad Auctions with Display Prices (2023) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Mechanism_Design_for_Ad_Auctions_with_Display_Prices.pdf) - 带展示价/提示价的广告拍卖机制
  - 引入展示价/提示价等信息披露设计，分析对竞价行为、平台收益与激励的影响。
  - 与LLM/生成式广告中的“信息披露”主题有呼应。
- Efficiency of non-truthful auctions under auto-bidding (2022) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Efficiency_of_non-truthful_auctions_under_auto-bidding.pdf) - 自动出价下非真实拍卖的效率分析
  - 分析非真实拍卖在auto-bidding下的效率损失与均衡性质，贴近真实市场规则。
  - 为平台选择FPA/SPA/变体规则提供理论依据。

### 2.2 深度机制设计
- Optimal Auctions through Deep Learning (2019) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Optimal_Auctions_through_Deep_Learning.pdf) - RegretNet，可微经济学奠基之作
  - 用神经网络参数化机制（分配/支付），并通过regret惩罚实现近似IC的端到端学习。
  - 深度机制设计入门核心，后续GemNet/PreferenceNet/BundleFlow均可对照。
- Neural Auction: End-to-End Learning of Auction Mechanisms for E-Commerce Advertising (2021) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Neural_Auction_End-to-End_Learning_of_Auction_Mechanisms_for_E-Commerce_Advertising.pdf) - 电商广告语境下的端到端神经拍卖
  - 将神经拍卖落地到电商广告多槽位/排序等场景，强调可训练、可部署与可扩展。
  - 填补“通用机制学习→广告应用”的落差。
- Mode Connectivity in Auction Design (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Mode_Connectivity_in_Auction_Design.pdf) - 神经机制设计理论：解释神经拍卖解的可连通性
  - 从理论角度解释神经拍卖优化景观（局部最优间可连通），支撑可微经济学可行性。
  - 适合补齐深度机制设计的理论基础。
- GemNet: Menu-Based Strategy-Proof Multi-Bidder Auctions (2024) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/GemNet_Menu-Based_Strategy-Proof_Multi-Bidder_Auctions.pdf) - 基于菜单的防策略拍卖
  - 通过“菜单”结构提升机制表达力，同时保持策略防护/可证明性质。
  - 与RegretNet互补：从机制表示与约束方式上增强可用性。
- BundleFlow: Deep Menus for Combinatorial Auctions (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/BundleFlow_Deep_Menus_for_Combinatorial_Auctions.pdf) - 大规模组合拍卖的深度菜单生成
  - 用流匹配/扩散式优化生成高维组合拍卖菜单，绕开枚举组合的计算瓶颈。
  - 组合拍卖SOTA路线之一，可与DP组合拍卖对比。

### 2.3 多目标/偏好与隐私 ⭐新增
- Optimising Trade-offs Among Stakeholders in Ad Auctions (2014) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Optimising_Trade-offs_Among_Stakeholders_in_Ad_Auctions.pdf) - 广告拍卖中多方利益权衡优化
  - 多目标广告拍卖经典：在平台收益、用户体验（点击）与广告主福利之间做权衡。
  - 为后续公平/偏好/多目标机制学习提供早期基线。
- PreferenceNet: Encoding Human Preferences in Auction Design with Deep Learning (2021) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/PreferenceNet_Encoding_Human_Preferences_in_Auction_Design_with_Deep_Learning.pdf) - 偏好/公平约束下的机制学习
  - 将偏好/公平/多目标约束显式编码进机制学习，使“偏好”成为机制设计的一等公民。
  - 与多目标广告拍卖、LLM偏好对齐方向相呼应。
- Differentially Private Machine Learning-powered Combinatorial Auction Design (2024) [[PDF]](Ad_Bidding_Auction_Mechanisms/2_拍卖机制设计/Differentially_Private_Machine_Learning-powered_Combinatorial_Auction_Design.pdf) - 差分隐私组合拍卖设计
  - 将差分隐私引入组合拍卖机制学习，兼顾隐私保护与收益/效率目标。
  - 适合研究“隐私计算 + 机制设计”的可落地方案。

---

## 3. LLM与经济代理 (LLM & Agentic Economics)
**视角**: AI Agent作为经济主体 | **核心问题**: LLM如何改变机制设计

### 3.1 LLM机制设计
- InfoBid: A Simulation Framework for Studying Information Disclosure in Auctions with Large Language Model-based Agents (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/3_LLM与经济代理/InfoBid_A_Simulation_Framework_for_Studying_Information_Disclosure_in_Auctions_with_Large_Language_Model-based_Agents.pdf) - LLM代理信息披露仿真
  - 提供LLM代理拍卖仿真框架，用于研究不同信息披露策略下的竞价/均衡变化。
  - 适合作为“LLM代理 + 机制设计”的可控实验基准。
- Mechanism Design for Large Language Models (2024) [[PDF]](Ad_Bidding_Auction_Mechanisms/3_LLM与经济代理/Mechanism_Design_for_Large_Language_Models.pdf) - LLM原生场景的机制设计
  - 面向LLM生态中的分配/定价/激励问题（如token级机制）的机制设计框架与讨论。
  - 为“生成式广告/LLM拍卖”提供理论工具箱。
- Ad Auctions for LLMs via Retrieval Augmented Generation (2024) [[PDF]](Ad_Bidding_Auction_Mechanisms/3_LLM与经济代理/Ad_Auctions_for_LLMs_via_Retrieval_Augmented_Generation.pdf) - RAG Auction/段落级拍卖
  - 提出RAG Auction：把检索候选与竞价定价嵌入生成式内容流程，实现段落级广告分配。
  - LLM商业化变现（生成式搜索广告）核心参考之一。
- LLM-Auction: Generative Auction towards LLM-Native Advertising (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/3_LLM与经济代理/LLM-Auction_Generative_Auction_towards_LLM-Native_Advertising.pdf) - IRPO：将拍卖机制转化为LLM偏好对齐问题
  - 将广告分配/定价视为LLM偏好对齐问题，提出IRPO等训练方法实现端到端生成式拍卖。
  - 与传统机制设计形成“对齐/奖励建模”新连接。

### 3.2 代理行为模拟
- RTBAgent: A LLM-based Agent System for Real-Time Bidding (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/3_LLM与经济代理/RTBAgent_A_LLM-based_Agent_System_for_Real-Time_Bidding.pdf) - LLM直接参与实时出价决策的Agent系统
  - 让LLM直接参与RTB决策：工具调用、记忆检索、两阶段决策等系统化组件。
  - 偏工程实现，适合参考AI Agent出价系统的模块拆分与评测方式。
- LLM Economist: Large Population Models and Mechanism Design in Multi-Agent Generative Simulacra (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/3_LLM与经济代理/LLM_Economist_Large_Population_Models_and_Mechanism_Design_in_Multi-Agent_Generative_Simulacra.pdf) - LLM经济仿真与机制评估
  - 用LLM模拟大规模经济主体与政策/机制效果（generative simulacra），用于机制设计实验。
  - 可作为“社会级模拟→机制评估”的方法论参考。

---

## 4. 博弈论基础 (Game Theory)
**视角**: 理论分析 | **核心问题**: 多智能体均衡求解

### 4.1 大规模博弈 ⭐新增
- Mean Field Multi-Agent Reinforcement Learning (2018) [[PDF]](Ad_Bidding_Auction_Mechanisms/4_博弈论基础/Mean_Field_Multi-Agent_Reinforcement_Learning.pdf) - 大规模多智能体平均场方法
  - 平均场MARL框架：用群体分布近似大规模多智能体交互，显著降低训练/推理复杂度。
  - 在广告市场（大量广告主）建模与训练中常用作理论基础。
- MESOB: Balancing Equilibria & Social Optimality in Ad Auctions (2023) [[PDF]](Ad_Bidding_Auction_Mechanisms/4_博弈论基础/MESOB_Balancing_Equilibria_Social_Optimality_Ad_Auctions.pdf) - 均衡与社会最优的折中优化
  - 在纳什均衡与社会最优之间做折中（均场/双层等），面向竞价推荐与平台目标优化。
  - 适合理解“平台目标 vs 广告主策略均衡”的冲突与折中。
- Budget Pacing in Repeated Auctions: Regret and Efficiency without Convergence (2022) [[PDF]](Ad_Bidding_Auction_Mechanisms/4_博弈论基础/Budget_Pacing_in_Repeated_Auctions_Regret_and_Efficiency_without_Convergence.pdf) - 重复拍卖中的预算pacing：遗憾与效率
  - 研究重复拍卖下pacing的遗憾与效率：即便不收敛，也能给出性能保证。
  - 与pacing系统的稳定性/动力学问题紧密相关。
- Learning in Repeated Auctions with Budgets: Regret Minimization and Equilibrium (2017) - budget pacing/学习的理论基础（暂无开放PDF）`[待下载]`
  - 预算约束重复拍卖的在线学习理论：遗憾最小化与均衡关系，是pacing理论的重要基石。
  - 建议优先补齐PDF，后续很多pacing/无悔学习分析都会引用。
- Learning to Bid in Repeated First-Price Auctions with Budgets (2023) [[PDF]](Ad_Bidding_Auction_Mechanisms/4_博弈论基础/Learning_to_Bid_in_Repeated_First-Price_Auctions_with_Budgets.pdf) - 一价重复拍卖下的预算约束学习出价
  - 面向一价重复拍卖+预算约束，研究学习出价策略及其理论界/收敛行为。
  - 可与No-Regret Autobidding、pacing equilibrium主线串联阅读。
- Online Ad Procurement in Non-stationary Autobidding Worlds (2023) [[PDF]](Ad_Bidding_Auction_Mechanisms/4_博弈论基础/Online_Ad_Procurement_in_Non-stationary_Autobidding_Worlds.pdf) - 非平稳环境下的在线采购/自动出价
  - 针对非平稳市场（季节/竞争变化）下的在线采购/投放策略学习，强调适应性。
  - 更贴近生产假设，适合补齐“动态市场”视角。
- No-Regret Online Autobidding Algorithms in First-price Auctions (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/4_博弈论基础/No-Regret_Online_Autobidding_Algorithms_in_First-price_Auctions.pdf) - ROI约束一价拍卖下的无悔学习与遗憾界
  - ROI约束FPA下的无悔学习算法与遗憾界，为DSP常见设定提供理论指导。
  - 适合作为“约束 + FPA + 在线学习”的理论主线材料。

### 4.2 多智能体RL
- A Cooperative-Competitive Multi-Agent Framework for Auto-bidding in Online Advertising (2021) [[PDF]](Ad_Bidding_Auction_Mechanisms/4_博弈论基础/A_Cooperative-Competitive_Multi-Agent_Framework_for_Auto-bidding_in_Online_Advertising.pdf) - 竞争-协作混合范式的多智能体自动出价框架
  - 多智能体视角建模auto-bidding中的竞争/协作，提出信用分配并用均场方法适配大规模广告主。
  - 连接MARL与市场机制，适合作为大规模auto-bidding系统的算法框架参考。

### 4.3 理论经典 (待补充)
- Credible Mechanisms (Akbarpour & Li, 2020) - 可信机制设计理论（暂无开放PDF）`[待下载]`
  - “可信机制”强调机制设计者的可信承诺：即便想作弊也难以偏离承诺结果（credibility/commitment）。
  - 对鲁棒拍卖、可信拍卖与平台-广告主博弈理解非常重要。

---

## 5. 基准与综述 (Benchmarks & Surveys)
**用途**: 入门学习、实验复现

- AuctionNet: A Novel Benchmark for Decision-Making in Large-Scale Games (2024) [[PDF]](Ad_Bidding_Auction_Mechanisms/5_基准与综述/AuctionNet_A_Novel_Benchmark_for_Decision-Making_in_Large-Scale_Games.pdf) - 阿里妈妈大规模广告拍卖决策基准
  - 面向大规模博弈/拍卖决策的统一基准，可用于比较auto-bidding/生成式策略等算法。
  - 适合作为DiffBid/BCOL等方法的统一评测平台。
- Real-Time Bidding Benchmarking with iPinYou Dataset (2014) [[PDF]](Ad_Bidding_Auction_Mechanisms/5_基准与综述/Real-Time_Bidding_Benchmarking_with_iPinYou_Dataset.pdf) - RTB公开数据集与基准复现
  - iPinYou公开数据集与RTB基准复现，经典离线评测入口。
  - 适合做CTR/CVR+出价策略的复现实验与对比。
- BAT: Benchmark for Auto-bidding Task (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/5_基准与综述/BAT_Benchmark_for_Auto-bidding_Task.pdf) - 自动出价任务基准与评测协议
  - 自动出价任务的基准与评测协议，强调可复现与统一指标/对照设置。
  - 可用于对齐不同论文的实验设置并降低复现实验成本。
- Auto-Bidding and Auctions in Online Advertising: A Survey (2024) [[PDF]](Ad_Bidding_Auction_Mechanisms/5_基准与综述/Auto-Bidding_and_Auctions_in_Online_Advertising_A_Survey.pdf) - 自动出价与广告拍卖综述
  - 权威综述：覆盖auto-bidding算法、拍卖机制与市场动态等核心问题。
  - 快速建立“出价-机制-博弈”全景并定位研究切入点。
- A Survey of Online Auction Mechanism Design Using Deep Learning Approaches (2021) [[PDF]](Ad_Bidding_Auction_Mechanisms/5_基准与综述/A_Survey_of_Online_Auction_Mechanism_Design_Using_Deep_Learning_Approaches.pdf) - 深度学习拍卖/机制设计综述
  - 深度学习与机制设计综述，梳理RegretNet等路线与在线拍卖应用。
  - 适合作为2.2深度机制设计板块的综述入口。
- A Practical Guide to Budget Pacing Algorithms in Digital Advertising (2025) [[PDF]](Ad_Bidding_Auction_Mechanisms/5_基准与综述/A_Practical_Guide_to_Budget_Pacing_Algorithms_in_Digital_Advertising.pdf) - pacing算法实践综述/指南
  - 工程实践导向的pacing指南：常见pacing策略、实现细节与调参经验总结。
  - 对自建/改造pacing系统非常实用。
- A Field Guide for Pacing Budget and ROS Constraints (2024) [[PDF]](Ad_Bidding_Auction_Mechanisms/5_基准与综述/A_Field_Guide_for_Pacing_Budget_and_ROS_Constraints.pdf) - pacing算法与ROS/预算约束的对比指南
  - 系统比较多类pacing算法在预算/ROS约束下的行为与适用条件，偏“算法选型”。
  - 可作为pacing系统设计评审时的参考清单。
- Automated Mechanism Design (Sandholm, 2003) - 自动化机制设计经典综述（暂无开放PDF）`[待下载]`
  - 自动机制设计经典：用算法搜索/优化机制规则，是“可微/学习机制设计”的历史源头。
  - 有助将深度机制设计工作放入更长的研究脉络中理解。

---

## 📁 本地目录结构

```
Ad_Bidding_Auction_Mechanisms/
├── 1_竞价策略/           (31篇)
├── 2_拍卖机制设计/        (16篇)
├── 3_LLM与经济代理/       (6篇)
├── 4_博弈论基础/          (7篇)
└── 5_基准与综述/          (7篇)
```

## 论文收集方式
1. 使用类似./search_prompt.md的prompt搜索整理出目标论文list
2. 逐项收集搜索到的论文，并使用使用curl命令直接下载到对应目录
3. 更新readme.md 中的论文清单
