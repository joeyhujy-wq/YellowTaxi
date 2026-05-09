"""
NYC Yellow Taxi Data Analysis (2023-01)
========================================
1. 加载数据并生成数据质量报告（缺失率、异常值统计）
2. 清洗数据，每步附策略说明
3. 提取时间相关特征（小时、星期、是否高峰）
4. 构造至少 2 个衍生特征
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("1. 数据加载")
print("=" * 60)

df = pd.read_parquet('yellow_tripdata_2023-01.parquet')
print(f"原始数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n字段列表:\n{df.dtypes}")

# ============================================================
# 2. 数据质量报告
# ============================================================
print("\n" + "=" * 60)
print("2. 数据质量报告")
print("=" * 60)

# 2.1 缺失率统计
print("\n--- 2.1 缺失率统计 ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(4)
missing_report = pd.DataFrame({
    '缺失数量': missing,
    '缺失率(%)': missing_pct
})
print(missing_report[missing_report['缺失数量'] > 0])

# 2.2 异常值统计（基于 IQR 与业务常识）
print("\n--- 2.2 异常值统计 ---")

numeric_cols = [
    'passenger_count', 'trip_distance', 'fare_amount',
    'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
    'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'airport_fee'
]

outlier_report = []

for col in numeric_cols:
    if col not in df.columns:
        continue
    series = df[col].dropna()
    
    # IQR 法
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    iqr_outliers = ((series < lower) | (series > upper)).sum()
    
    # 零值/负值统计
    zero_count = (series == 0).sum()
    neg_count = (series < 0).sum()
    
    outlier_report.append({
        '字段': col,
        '最小值': series.min(),
        '最大值': series.max(),
        'IQR异常值数': iqr_outliers,
        'IQR异常值比例(%)': round(iqr_outliers / len(series) * 100, 4),
        '零值数': zero_count,
        '负值数': neg_count
    })

outlier_df = pd.DataFrame(outlier_report)
print(outlier_df.to_string(index=False))

# 2.3 时间字段异常
print("\n--- 2.3 时间字段异常 ---")
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0

neg_duration = (df['trip_duration'] < 0).sum()
zero_duration = (df['trip_duration'] == 0).sum()
extreme_duration = (df['trip_duration'] > 24 * 60).sum()  # 超过24小时

print(f"负行程时间（上车晚于下车）: {neg_duration} 条")
print(f"零行程时间: {zero_duration} 条")
print(f"超长行程时间（>24小时）: {extreme_duration} 条")

# 2.4 业务逻辑异常
print("\n--- 2.4 业务逻辑异常 ---")
print(f"乘车人数为 0: {(df['passenger_count'] == 0).sum()} 条")
print(f"乘车人数 > 6: {(df['passenger_count'] > 6).sum()} 条")
print(f"行驶距离为 0 但金额 > 0: {((df['trip_distance'] == 0) & (df['fare_amount'] > 0)).sum()} 条")
print(f"行驶距离 > 100 英里: {(df['trip_distance'] > 100).sum()} 条")

# ============================================================
# 3. 数据清洗
# ============================================================
print("\n" + "=" * 60)
print("3. 数据清洗")
print("=" * 60)

original_rows = len(df)

# ---- 策略 1: 删除行程时间为负的记录 ----
# 理由: 下车时间早于上车时间属于数据录入错误，无法正确计算行程时长，
#       且无法判断哪一字段出错，故整行删除。
mask_neg = df['trip_duration'] < 0
print(f"\n[策略1] 删除负行程时间记录: {mask_neg.sum()} 条")
df = df[~mask_neg].copy()

# ---- 策略 2: 删除行程时间为 0 且费用为 0 的记录 ----
# 理由: 既无行驶距离也无费用，说明是取消订单或无效记录，对分析无意义。
mask_zero = (df['trip_duration'] == 0) & (df['total_amount'] == 0)
print(f"[策略2] 删除零时长且零金额记录: {mask_zero.sum()} 条")
df = df[~mask_zero].copy()

# ---- 策略 3: 填充 passenger_count 缺失值 ----
# 理由: 缺失量通常较小（<5%），用众数 1 填充最符合单乘客出行的常见场景。
print(f"[策略3] passenger_count 缺失值用众数({df['passenger_count'].mode()[0]})填充: {df['passenger_count'].isnull().sum()} 条")
df['passenger_count'] = df['passenger_count'].fillna(df['passenger_count'].mode()[0])

# ---- 策略 4: 填充 RatecodeID 缺失值 ----
# 理由: 标准费率（1）占绝对多数，缺失时用众数填充不会引入系统性偏差。
print(f"[策略4] RatecodeID 缺失值用众数({df['RatecodeID'].mode()[0]})填充: {df['RatecodeID'].isnull().sum()} 条")
df['RatecodeID'] = df['RatecodeID'].fillna(df['RatecodeID'].mode()[0])

# ---- 策略 5: 将 store_and_fwd_flag 缺失视为 'N' ----
# 理由: 该字段表示是否先存盘再转发，绝大多数行程实时传输，缺失大概率代表未存储，按 'N' 处理。
print(f"[策略5] store_and_fwd_flag 缺失值填充为 'N': {df['store_and_fwd_flag'].isnull().sum()} 条")
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].fillna('N')

# ---- 策略 6: 填充各项附加费缺失为 0 ----
# 理由: 附加费（congestion_surcharge, airport_fee 等）缺失通常表示未产生该费用，等价于 0。
fee_cols = ['extra', 'mta_tax', 'improvement_surcharge', 'congestion_surcharge', 'airport_fee']
for col in fee_cols:
    missing_cnt = df[col].isnull().sum()
    if missing_cnt > 0:
        print(f"[策略6] {col} 缺失值填充为 0: {missing_cnt} 条")
        df[col] = df[col].fillna(0)

# ---- 策略 7: 剔除极端异常行程距离 ----
# 理由: 距离 > 100 英里或 < 0 英里超出纽约市出租车正常运营范围，可能是录入错误。
mask_dist = (df['trip_distance'] > 100) | (df['trip_distance'] < 0)
print(f"[策略7] 删除极端行程距离(>100或<0): {mask_dist.sum()} 条")
df = df[~mask_dist].copy()

# ---- 策略 8: 剔除极端异常金额 ----
# 理由: total_amount <= 0 意味着行程未产生费用或数据错误；> 1000 美元极可能是异常。
mask_amount = (df['total_amount'] <= 0) | (df['total_amount'] > 1000)
print(f"[策略8] 删除极端金额(<=0或>1000): {mask_amount.sum()} 条")
df = df[~mask_amount].copy()

# ---- 策略 9: 剔除超长行程时间 ----
# 理由: 行程超过 24 小时大概率是停车未结束计费或传感器故障，非真实出行。
mask_long = df['trip_duration'] > 24 * 60
print(f"[策略9] 删除超长行程时间(>24h): {mask_long.sum()} 条")
df = df[~mask_long].copy()

# ---- 策略 10: 删除非必要字段 trip_duration（将在特征工程中重新计算）----
# 理由: 当前仅用于清洗，后续会基于清洗后的时间重新精确计算。
print(f"[策略10] 移除临时字段 trip_duration")
df = df.drop(columns=['trip_duration'])

final_rows = len(df)
print(f"\n清洗完成: {original_rows} → {final_rows} 条，移除 {original_rows - final_rows} 条 ({(original_rows - final_rows)/original_rows*100:.2f}%)")

# ============================================================
# 4. 特征工程
# ============================================================
print("\n" + "=" * 60)
print("4. 特征工程")
print("=" * 60)

# 4.1 基础时间特征
# ----------------
# 提取小时（0-23）
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
# 提取星期几（周一=0, 周日=6）
df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
# 是否周末
df['is_weekend'] = df['pickup_weekday'].isin([5, 6]).astype(int)
# 是否高峰时段（工作日 7-9 点, 17-19 点）
# 理由: NYC 出租车高峰通常定义在工作日早晚通勤时段
morning_rush = (df['pickup_hour'].isin([7, 8, 9])) & (df['is_weekend'] == 0)
evening_rush = (df['pickup_hour'].isin([17, 18, 19])) & (df['is_weekend'] == 0)
df['is_rush_hour'] = (morning_rush | evening_rush).astype(int)

print("\n[基础时间特征] pickup_hour, pickup_weekday, is_weekend, is_rush_hour 已生成")

# 4.2 衍生特征 1: 平均速度 (mph)
# ----------------
# 理由: 平均速度能反映交通拥堵程度，也能识别异常低速（绕路/怠速）或异常高速（数据错误）。
# 先重新计算行程时长（小时）
df['trip_duration_hour'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 3600.0
# 避免除零
df['avg_speed_mph'] = np.where(
    df['trip_duration_hour'] > 0,
    df['trip_distance'] / df['trip_duration_hour'],
    np.nan
)
# 速度超过 80 mph 在纽约城区不现实，标记为异常（设为 NaN，可根据需要后续处理）
df.loc[df['avg_speed_mph'] > 80, 'avg_speed_mph'] = np.nan
print("[衍生特征1] avg_speed_mph (平均速度，mph) 已生成")

# 4.3 衍生特征 2: 小费占比 (%)
# ----------------
# 理由: 小费比例是衡量乘客满意度与司机服务质量的重要代理指标，比绝对小费金额更有意义。
df['tip_percentage'] = np.where(
    df['fare_amount'] > 0,
    df['tip_amount'] / df['fare_amount'] * 100,
    0
)
# 将超过 50% 的小费视为异常（信用卡录入错误或数据噪声）
df.loc[df['tip_percentage'] > 50, 'tip_percentage'] = np.nan
print("[衍生特征2] tip_percentage (小费占车费比例, %) 已生成")

# 4.4 衍生特征 3: 每英里成本 (cost_per_mile)
# ----------------
# 理由: 单位距离成本可剔除距离因素，用于比较不同区域的定价效率或识别绕路订单。
df['cost_per_mile'] = np.where(
    df['trip_distance'] > 0,
    df['total_amount'] / df['trip_distance'],
    np.nan
)
# 超过 50 美元/英里通常意味着极短距离高基础费或异常
df.loc[df['cost_per_mile'] > 50, 'cost_per_mile'] = np.nan
print("[衍生特征3] cost_per_mile (每英里总成本, $) 已生成")

# 4.5 衍生特征 4: 是否机场行程
# ----------------
# 理由: 机场行程（JFK/LGA/EWR）有固定费率与特殊附加费，出行模式与普通市区订单差异显著。
# 基于 airport_fee > 0 或 RatecodeID == 2 (JFK) 识别
# RatecodeID: 1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated, 6=Group ride
df['is_airport_trip'] = ((df['airport_fee'] > 0) | (df['RatecodeID'].isin([2, 3]))).astype(int)
print("[衍生特征4] is_airport_trip (是否机场行程) 已生成")

# ============================================================
# 5. 特征与数据预览
# ============================================================
print("\n" + "=" * 60)
print("5. 结果预览")
print("=" * 60)

feature_cols = [
    'tpep_pickup_datetime', 'tpep_dropoff_datetime',
    'pickup_hour', 'pickup_weekday', 'is_weekend', 'is_rush_hour',
    'trip_distance', 'trip_duration_hour',
    'avg_speed_mph', 'tip_percentage', 'cost_per_mile', 'is_airport_trip'
]

print("\n--- 新特征示例（前5行）---")
print(df[feature_cols].head())

print("\n--- 新特征描述性统计 ---")
print(df[['pickup_hour', 'pickup_weekday', 'is_rush_hour',
          'avg_speed_mph', 'tip_percentage', 'cost_per_mile', 'is_airport_trip']].describe())

# ============================================================
# 6. 保存清洗后的数据（可选）
# ============================================================
output_file = 'yellow_tripdata_2023-01_cleaned.parquet'
df.to_parquet(output_file, index=False)
print(f"\n清洗与特征工程后的数据已保存: {output_file}")
print(f"最终数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
