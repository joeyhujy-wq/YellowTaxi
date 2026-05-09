# NYC Yellow Taxi 数据分析与智能问答系统

> 基于 2023 年 1 月纽约市黄色出租车运营数据（约 304 万条订单），完成数据清洗、可视化分析、需求预测建模与智能问答系统的完整数据科学项目。

---

## 项目概览

本项目从原始 Parquet 数据出发，历经 **清洗 → 可视化 → 预测 → 问答** 四个阶段，形成一套可运行的数据分析 pipeline：

| 模块 | 核心任务 | 关键技术 | 产出 |
|:---|:---|:---|:---|
| **M1** 数据清洗 | 缺失值处理、异常值剔除、特征工程 | pandas, IQR, 众数填充 | 清洗后数据 + 10 步清洗策略报告 |
| **M2** 可视化 | 时间规律、区域热度、车费因素、拥堵分析 | matplotlib, seaborn | 10+ 张图表 + 洞察报告 |
| **M3** 需求预测 | 区域 161 小时级需求量预测 | PyTorch, Random Forest, StandardScaler | 神经网络 vs 随机森林对比报告 |
| **M4** 智能问答 | 自然语言问答系统 | 规则引擎 + Kimi API | 命令行交互程序 + 设计报告 |

---

## 项目结构

```
.
├── data/
│   ├── yellow_tripdata_2023-01.parquet          # 原始数据
│   └── yellow_tripdata_2023-01_cleaned.parquet  # M1 清洗后数据
│
├── outputs/                                     # 图表输出目录
│   ├── M2-1_hourly_demand.png                   # 分小时出行需求折线图
│   ├── M2-1_hourly_area.png                     # 出行需求面积图
│   ├── M2-2a_top10_pickup.png                   # TOP 10 上车区域
│   ├── M2-2b_top10_dropoff.png                  # TOP 10 下车区域
│   ├── M2-2c_zone_hour_heatmap.png              # 区域 × 小时热力图
│   ├── M2-3a_distance_fare_scatter.png          # 距离-车费散点图
│   ├── M2-3b_rushhour_fare_box.png              # 高峰/非高峰车费箱线图
│   ├── M2-3c_passenger_fare_violin.png          # 乘客人数-车费小提琴图
│   ├── M2-4a_speed_heatmap.png                  # 速度时空热力图
│   ├── M2-4b_airport_vs_city.png                # 机场 vs 市区效率对比
│   ├── M3_A_nn_loss_curve.png                   # 神经网络 Loss 曲线
│   ├── M3_A_timeseries_comparison.png           # 时序预测对比
│   ├── M3_A_prediction_scatter.png              # 预测散点图
│   ├── M3_A_metrics_comparison.png              # 指标对比柱状图
│   └── qa_zone161_hourly.png                    # 问答系统动态生成图
│
├── taxi_analysis.py                             # M1: 数据清洗与特征工程
├── taxi_visualization_m2.py                     # M2: 可视化分析
├── taxi_demand_prediction.py                    # M3: 日度预测（初版）
├── taxi_demand_prediction_hourly.py             # M3: 小时级预测（方案 A）
├── main.py                                      # M4: 智能问答系统
│
├── yellow_taxi_analysis_report.md               # M1 数据质量与清洗报告
├── M2_insights_summary.md                       # M2 可视化核心洞察
├── M3_A_model_comparison_analysis.md            # M3 模型对比分析（自动输出）
├── 任务3_区域161小时级需求预测报告.md           # M3 任务报告
├── 任务4_智能问答系统设计与实现.md              # M4 系统设计报告
├── 人机协作报告.md                              # AI 协作过程记录
│
└── README.md                                    # 本文件
```

---

## 环境依赖

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openai torch
```

核心依赖版本参考：
- Python >= 3.9
- pandas >= 2.0
- matplotlib >= 3.8
- seaborn >= 0.13
- scikit-learn >= 1.3
- openai >= 1.0
- torch >= 2.0

> **中文字体**: Windows 系统需确保存在 `C:\Windows\Fonts\msyh.ttc`（微软雅黑）。若使用其他系统，请修改各脚本中的 `font_path` 变量。

---

## 快速开始

### M1：数据清洗与特征工程

```bash
python taxi_analysis.py
```

- 输入：`data/yellow_tripdata_2023-01.parquet`
- 输出：`data/yellow_tripdata_2023-01_cleaned.parquet`（28 列，含新生成特征）
- 报告：`yellow_taxi_analysis_report.md`

### M2：可视化分析

```bash
python taxi_visualization_m2.py
```

- 依赖：需先运行 M1 生成清洗后数据
- 输出：`outputs/` 目录下 10 张分析图表
- 报告：`M2_insights_summary.md`

### M3：小时级需求预测

```bash
python taxi_demand_prediction_hourly.py
```

- 依赖：需先运行 M1
- 输出：Loss 曲线、时序对比图、散点图、指标对比图
- 报告：`任务3_区域161小时级需求预测报告.md`

> 注：`taxi_demand_prediction.py` 为日度预测初版（样本过少，已弃用），保留供参考。

### M4：智能问答系统

```bash
# 方式 1：环境变量传入 API Key
set MOONSHOT_API_KEY=your-key-here
python main.py

# 方式 2：运行时交互输入
python main.py
```

- 支持 6 种问题类型：时段查询、区域排名、需求预测、费用估算、速度/拥堵查询、数据概览
- 规则未匹配时自动调用 Kimi 大模型 API
- 输入 `quit` 或 `exit` 退出交互

---

## 核心发现摘要

### M1 数据清洗
- 原始数据 3,066,766 条，清洗后 3,040,881 条（移除 0.84%）
- 缺失值集中在 71,743 条记录，采用众数/零值填充策略
- 生成 4 个衍生特征：`avg_speed_mph`、`tip_percentage`、`cost_per_mile`、`is_airport_trip`

### M2 可视化洞察
- 工作日呈现通勤双峰（8–9 点、17–19 点），周末夜间需求更高
- 区域 237、161、236 为曼哈顿核心热区
- 高峰时段车费中位数略高，主要来自拥堵附加费
- 工作日早高峰平均速度跌至 8–10 mph，凌晨 4–6 点最通畅

### M3 需求预测
- **目标**：区域 161 小时级需求量预测
- **样本**：训练集 576 条 / 测试集 145 条
- **最佳模型**：随机森林（MAE = 19.88，RMSE = 27.50）
- **关键特征**：`demand_lag_1` 重要性高达 86.9%，揭示强自回归特性

### M4 智能问答
- 双层架构：规则引擎（6 类结构化查询）+ Kimi API（开放式问题兜底）
- System Prompt 历经 v1.0 → v1.1 迭代，通过显式枚举字段降低幻觉概率

---

## 报告清单

| 报告文件 | 内容 |
|:---|:---|
| `yellow_taxi_analysis_report.md` | M1 数据质量报告、清洗策略详解、特征工程说明 |
| `M2_insights_summary.md` | M2 四项可视化分析的核心发现速览 |
| `任务3_区域161小时级需求预测报告.md` | M3 实验设定、性能对比、优劣分析 |
| `任务4_智能问答系统设计与实现.md` | M4 架构设计、Prompt 迭代、交互示例 |
| `人机协作报告.md` | 全项目 AI 交互日志、三阶段对比、反思 |

---

## 技术栈

- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **机器学习**: scikit-learn (Random Forest)
- **深度学习**: PyTorch
- **大模型 API**: Moonshot (Kimi) API
- **数据格式**: Apache Parquet

---

## 免责声明

- 数据集来源为 NYC TLC 公开数据（2023-01），仅用于学术分析与算法演示。
- Kimi API 调用需要用户自行申请 `MOONSHOT_API_KEY`。
- 需求预测结果基于历史均值估计，不构成实际运营决策建议。

---

*项目完成日期: 2026-05-09*
