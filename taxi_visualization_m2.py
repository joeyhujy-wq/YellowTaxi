"""
M2: NYC Yellow Taxi 数据可视化分析
=====================================
1. 出行需求时间规律（分小时、工作日/周末）
2. 区域热度分析（TOP 10 区域、高峰时段分布）
3. 车费影响因素分析（距离、时段、乘客人数 vs 车费）
4. 推荐分析：行程效率与城市拥堵时空分析

所有图表自动保存至 outputs/ 目录
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# ============================================================
# 初始化设置
# ============================================================
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文字体（Windows 优先尝试微软雅黑，降级到 SimHei）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

# ============================================================
# 加载清洗后的数据
# ============================================================
print("加载数据...")
df = pd.read_parquet('yellow_tripdata_2023-01_cleaned.parquet')
print(f"数据维度: {df.shape}")

# 为了绘图速度，对超大散点图进行采样
SAMPLE_SIZE = 50000

# ============================================================
# M2-1: 出行需求时间规律
# ============================================================
print("\n[M2-1] 出行需求时间规律...")

# 计算工作日 / 周末 分小时订单量
df['week_type'] = df['is_weekend'].map({0: '工作日', 1: '周末'})
hourly_demand = df.groupby(['pickup_hour', 'week_type']).size().reset_index(name='订单量')
hourly_pivot = hourly_demand.pivot(index='pickup_hour', columns='week_type', values='订单量')

fig, ax = plt.subplots(figsize=(10, 5))
hourly_pivot.plot(kind='line', marker='o', ax=ax, linewidth=2)
ax.set_title('M2-1a: 分小时平均出行需求（工作日 vs 周末）', fontsize=14, fontweight='bold')
ax.set_xlabel('小时 (0-23)', fontsize=12)
ax.set_ylabel('订单量', fontsize=12)
ax.set_xticks(range(0, 24))
ax.legend(title='日期类型')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'M2-1_hourly_demand.png'), dpi=300, bbox_inches='tight')
plt.close()

# 工作日 vs 周末 全天分布对比（面积图）
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(hourly_pivot.index, hourly_pivot['工作日'], alpha=0.4, label='工作日')
ax.fill_between(hourly_pivot.index, hourly_pivot['周末'], alpha=0.4, label='周末')
ax.plot(hourly_pivot.index, hourly_pivot['工作日'], linewidth=2)
ax.plot(hourly_pivot.index, hourly_pivot['周末'], linewidth=2)
ax.set_title('M2-1b: 出行需求时间分布面积图（工作日 vs 周末）', fontsize=14, fontweight='bold')
ax.set_xlabel('小时 (0-23)', fontsize=12)
ax.set_ylabel('订单量', fontsize=12)
ax.set_xticks(range(0, 24))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'M2-1_hourly_area.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# M2-2: 区域热度分析
# ============================================================
print("[M2-2] 区域热度分析...")

# TOP 10 上车区域
top10_pu = df['PULocationID'].value_counts().head(10).reset_index()
top10_pu.columns = ['区域ID', '上车量']

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=top10_pu, x='区域ID', y='上车量', palette='Blues_r', ax=ax, order=top10_pu['区域ID'])
ax.set_title('M2-2a: 上车量 TOP 10 区域', fontsize=14, fontweight='bold')
ax.set_xlabel('区域 ID (PULocationID)', fontsize=12)
ax.set_ylabel('上车订单量', fontsize=12)
for i, v in enumerate(top10_pu['上车量']):
    ax.text(i, v + max(top10_pu['上车量']) * 0.01, f'{v:,}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'M2-2a_top10_pickup.png'), dpi=300, bbox_inches='tight')
plt.close()

# TOP 10 下车区域
top10_do = df['DOLocationID'].value_counts().head(10).reset_index()
top10_do.columns = ['区域ID', '下车量']

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=top10_do, x='区域ID', y='下车量', palette='Oranges_r', ax=ax, order=top10_do['区域ID'])
ax.set_title('M2-2b: 下车量 TOP 10 区域', fontsize=14, fontweight='bold')
ax.set_xlabel('区域 ID (DOLocationID)', fontsize=12)
ax.set_ylabel('下车订单量', fontsize=12)
for i, v in enumerate(top10_do['下车量']):
    ax.text(i, v + max(top10_do['下车量']) * 0.01, f'{v:,}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'M2-2b_top10_dropoff.png'), dpi=300, bbox_inches='tight')
plt.close()

# 高峰时段 TOP 10 上车区域热力图（小时 × TOP10 区域）
top10_pu_ids = top10_pu['区域ID'].tolist()
pickup_heatmap = df[df['PULocationID'].isin(top10_pu_ids)].groupby(['PULocationID', 'pickup_hour']).size().reset_index(name='订单量')
pickup_heatmap_pivot = pickup_heatmap.pivot(index='PULocationID', columns='pickup_hour', values='订单量').fillna(0)
# 按总订单量排序
pickup_heatmap_pivot = pickup_heatmap_pivot.loc[top10_pu_ids]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pickup_heatmap_pivot, cmap='YlOrRd', annot=False, fmt='.0f', linewidths=0.5, ax=ax)
ax.set_title('M2-2c: TOP 10 上车区域 × 小时 热力图', fontsize=14, fontweight='bold')
ax.set_xlabel('小时 (0-23)', fontsize=12)
ax.set_ylabel('区域 ID (PULocationID)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'M2-2c_zone_hour_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# M2-3: 车费影响因素分析
# ============================================================
print("[M2-3] 车费影响因素分析...")

# 距离-车费散点图（采样，避免过度绘制）
df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(df_sample['trip_distance'], df_sample['total_amount'],
                     c=df_sample['pickup_hour'], cmap='viridis', alpha=0.5, s=10)
ax.set_title('M2-3a: 行程距离 vs 总金额（颜色=上车小时）', fontsize=14, fontweight='bold')
ax.set_xlabel('行程距离 (英里)', fontsize=12)
ax.set_ylabel('总金额 ($)', fontsize=12)
ax.set_xlim(0, 30)  # 聚焦主流范围
ax.set_ylim(0, 150)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('上车小时')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'M2-3a_distance_fare_scatter.png'), dpi=300, bbox_inches='tight')
plt.close()

# 高峰/非高峰 vs 车费 箱线图
fig, ax = plt.subplots(figsize=(8, 5))
df_plot = df[df['total_amount'] <= 100].copy()  # 限制范围以便观察
df_plot['时段类型'] = df_plot['is_rush_hour'].map({0: '非高峰', 1: '高峰'})
sns.boxplot(data=df_plot, x='时段类型', y='total_amount', palette='Set2', ax=ax)
ax.set_title('M2-3b: 高峰/非高峰时段的总金额分布', fontsize=14, fontweight='bold')
ax.set_xlabel('时段类型', fontsize=12)
ax.set_ylabel('总金额 ($)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'M2-3b_rushhour_fare_box.png'), dpi=300, bbox_inches='tight')
plt.close()

# 乘客人数 vs 车费（小提琴图）
fig, ax = plt.subplots(figsize=(9, 5))
df_pass = df[(df['passenger_count'] <= 6) & (df['total_amount'] <= 100)].copy()
sns.violinplot(data=df_pass, x='passenger_count', y='total_amount', palette='muted', ax=ax, inner='quartile')
ax.set_title('M2-3c: 乘客人数 vs 总金额分布', fontsize=14, fontweight='bold')
ax.set_xlabel('乘客人数', fontsize=12)
ax.set_ylabel('总金额 ($)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'M2-3c_passenger_fare_violin.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# M2-4: 推荐分析 — 行程效率与城市拥堵时空分析
# ============================================================
print("[M2-4] 推荐分析：行程效率与城市拥堵时空分析...")

# 核心洞察：出租车平均速度是城市拥堵的"体温计"，直接反映道路通行效率。
# 通过分析平均速度在"星期×小时"维度的分布，可以精准定位 NYC 的拥堵时空规律。

# 4a: 平均速度热力图（星期 × 小时）
speed_heatmap = df.groupby(['pickup_weekday', 'pickup_hour'])['avg_speed_mph'].median().reset_index()
speed_pivot = speed_heatmap.pivot(index='pickup_weekday', columns='pickup_hour', values='avg_speed_mph')
weekday_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
speed_pivot.index = weekday_labels

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(speed_pivot, cmap='RdYlGn', annot=False, fmt='.1f', linewidths=0.5, ax=ax, vmin=0, vmax=20)
ax.set_title('M2-4a: 中位平均速度热力图（星期 × 小时）\n颜色越绿=越快，越红=越堵', fontsize=14, fontweight='bold')
ax.set_xlabel('小时 (0-23)', fontsize=12)
ax.set_ylabel('星期', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'M2-4a_speed_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4b: 机场 vs 非机场行程的速度与耗时对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df_air = df.copy()
df_air['行程类型'] = df_air['is_airport_trip'].map({0: '市区普通', 1: '机场行程'})

# 左：平均速度对比（过滤极端值）
speed_data = df_air[df_air['avg_speed_mph'].notna() & (df_air['avg_speed_mph'] <= 50)]
sns.boxplot(data=speed_data, x='行程类型', y='avg_speed_mph', palette='coolwarm', ax=axes[0])
axes[0].set_title('平均速度对比', fontsize=12, fontweight='bold')
axes[0].set_ylabel('平均速度 (mph)')

# 右：行程时长对比（分钟）
df_air['trip_duration_min'] = (df_air['tpep_dropoff_datetime'] - df_air['tpep_pickup_datetime']).dt.total_seconds() / 60
duration_data = df_air[df_air['trip_duration_min'] <= 120]  # 过滤极端长行程
sns.boxplot(data=duration_data, x='行程类型', y='trip_duration_min', palette='coolwarm', ax=axes[1])
axes[1].set_title('行程时长对比', fontsize=12, fontweight='bold')
axes[1].set_ylabel('行程时长 (分钟)')

fig.suptitle('M2-4b: 机场行程 vs 市区普通行程效率对比', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'M2-4b_airport_vs_city.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 50)
print("所有图表已生成并保存至 outputs/ 目录:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        print(f"  - {f}")
print("=" * 50)
