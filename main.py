"""
任务 4: NYC Yellow Taxi 智能问答系统
=====================================
- 命令行问答循环
- 规则匹配 >= 5 种问题类型
- 结合 M1-M3 数据与模型进行回答
- 无法匹配时接入 Kimi 大模型 API
- 返回数字结论 + 图表路径
"""

import os
import re
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from datetime import datetime, timedelta
import torch
from openai import OpenAI

# ============================================================
# 全局初始化
# ============================================================
font_path = r'C:\Windows\Fonts\msyh.ttc'
fp_title = FontProperties(fname=font_path, size=12, weight='bold')
fp_label = FontProperties(fname=font_path, size=10)
fp_tick = FontProperties(fname=font_path, size=9)
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = 'data/yellow_tripdata_2023-01_cleaned.parquet'
MODEL_PATH = None  # 神经网络模型将在需要时加载

print("正在加载数据...")
DF = pd.read_parquet(DATA_PATH)
print(f"数据加载完成: {DF.shape}")

# ============================================================
# Kimi API 客户端初始化
# ============================================================
def init_kimi_client():
    api_key = os.environ.get('MOONSHOT_API_KEY')
    if not api_key:
        api_key = input("请输入 Kimi (Moonshot) API Key: ").strip()
        if not api_key:
            print("警告: 未提供 API Key，大模型功能将不可用")
            return None
    return OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

KIMI_CLIENT = init_kimi_client()
KIMI_MODEL = "moonshot-v1-8k"

# ============================================================
# System Prompt 设计（v1.0）
# ============================================================
SYSTEM_PROMPT = """你是 NYC Yellow Taxi 数据分析助手。你拥有 2023 年 1 月纽约黄色出租车清洗后的运营数据（约 304 万条订单），数据包含以下字段：
- 时间: tpep_pickup_datetime, tpep_dropoff_datetime
- 空间: PULocationID（上车区域）, DOLocationID（下车区域）
- 行程: trip_distance（英里）, trip_duration_hour（小时）
- 费用: fare_amount, tip_amount, total_amount, congestion_surcharge, airport_fee
- 衍生特征: pickup_hour, pickup_weekday, is_weekend, is_rush_hour, avg_speed_mph, tip_percentage, cost_per_mile, is_airport_trip

你的职责：
1. 对用户的出租车数据相关问题给出精准、简洁的数字结论。
2. 若问题涉及图表，请说明对应图表文件路径（如 outputs/M2-1_hourly_demand.png）。
3. 若用户问题超出数据范围（如 2023 年 2 月、非 NYC 区域），请诚实说明无法回答，并给出最接近的推断或建议。
4. 保持回答简短，优先给出数字结论，再辅以简要解释。
5. 回答请使用中文。
"""

# ============================================================
# 辅助函数
# ============================================================
def ensure_output_dir():
    os.makedirs('outputs', exist_ok=True)

def parse_zone(text):
    """从文本中提取区域 ID"""
    patterns = [
        r'区域\s*(\d+)',
        r'zone\s*(\d+)',
        r'location\s*(\d+)',
        r'(\d+)\s*区',
        r'PULocationID[=\s]*(\d+)',
        r'DOLocationID[=\s]*(\d+)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None

def parse_hour(text):
    """从文本中提取小时"""
    # 中文数字
    cn_nums = {'一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9,
               '十':10, '十一':11, '十二':12, '十三':13, '十四':14, '十五':15, '十六':16,
               '十七':17, '十八':18, '十九':19, '二十':20, '二十一':21, '二十二':22, '二十三':23,
               '零':0, '两':2}
    for cn, num in sorted(cn_nums.items(), key=lambda x: -len(x[0])):
        if cn in text:
            return num
    # 阿拉伯数字 + 点/时
    m = re.search(r'(\d{1,2})\s*[点:：时]', text)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h
    # 纯数字
    m = re.search(r'\b(\d{1,2})\b', text)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h
    return None

def parse_top_n(text):
    """提取 TOP N"""
    m = re.search(r'前\s*(\d+)', text)
    if m:
        return int(m.group(1))
    m = re.search(r'top\s*(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s*个', text)
    if m:
        return int(m.group(1))
    return 5  # 默认

# ============================================================
# 问题类型 1: 时段查询
# ============================================================
def handle_time_query(question):
    """
    匹配示例:
    - "区域161上午10点有多少订单"
    - "晚上8点区域237的出租车数量"
    - "周五下午3点哪个区域订单最多"
    """
    zone = parse_zone(question)
    hour = parse_hour(question)
    if hour is None:
        return None
    
    mask = DF['pickup_hour'] == hour
    if zone is not None:
        mask = mask & (DF['PULocationID'] == zone)
    
    count = mask.sum()
    
    if zone:
        result = f"区域 {zone} 在 {hour}:00 的订单量为 **{count:,}** 条。"
        # 生成迷你图表
        ensure_output_dir()
        hourly = DF[DF['PULocationID']==zone].groupby('pickup_hour').size()
        fig, ax = plt.subplots(figsize=(8, 4))
        hourly.plot(kind='bar', ax=ax, color='steelblue')
        ax.axvline(x=hour, color='red', linestyle='--', linewidth=2, label=f'{hour}:00')
        ax.set_title(f'区域 {zone} 分小时订单量', fontproperties=fp_title)
        ax.set_xlabel('小时', fontproperties=fp_label)
        ax.set_ylabel('订单量', fontproperties=fp_label)
        for label in ax.get_xticklabels():
            label.set_fontproperties(fp_tick)
        for label in ax.get_yticklabels():
            label.set_fontproperties(fp_tick)
        ax.legend(prop=fp_tick)
        plt.tight_layout()
        chart_path = f'outputs/qa_zone{zone}_hourly.png'
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()
        result += f"\n[图表] {chart_path}"
    else:
        result = f"全区域在 {hour}:00 的订单量为 **{count:,}** 条。"
        result += f"\n[参考图表] outputs/M2-1_hourly_demand.png"
    
    return result

# ============================================================
# 问题类型 2: 区域排名
# ============================================================
def handle_zone_ranking(question):
    """
    匹配示例:
    - "上车量最高的前5个区域"
    - "哪个区域最热门"
    - "TOP 10 热门下车区域"
    """
    n = parse_top_n(question)
    is_dropoff = any(k in question for k in ['下车', 'dropoff', '目的', '到达'])
    col = 'DOLocationID' if is_dropoff else 'PULocationID'
    label = '下车' if is_dropoff else '上车'
    
    top = DF[col].value_counts().head(n)
    result = f"**{label}量 TOP {n} 区域**:\n"
    for i, (zone_id, cnt) in enumerate(top.items(), 1):
        result += f"  {i}. 区域 {zone_id}: {cnt:,} 条\n"
    
    result += f"\n[参考图表] outputs/M2-2a_top10_pickup.png"
    return result

# ============================================================
# 问题类型 3: 需求预测
# ============================================================
def handle_demand_prediction(question):
    """
    匹配示例:
    - "预测区域161明天下午3点的需求量"
    - "区域237晚上8点会有多少订单"
    """
    zone = parse_zone(question)
    hour = parse_hour(question)
    if zone is None or hour is None:
        return None
    
    # 使用历史同区域同时段的均值作为估计
    mask = (DF['PULocationID'] == zone) & (DF['pickup_hour'] == hour)
    subset = DF[mask]
    if len(subset) == 0:
        return f"区域 {zone} 在 {hour}:00 的历史数据不足，无法预测。"
    
    # 按天聚合
    subset_copy = subset.copy()
    subset_copy['date'] = subset_copy['tpep_pickup_datetime'].dt.date
    daily = subset_copy.groupby('date').size()
    mean_demand = daily.mean()
    std_demand = daily.std()
    
    result = f"**区域 {zone} {hour}:00 需求量预测**:\n"
    result += f"  - 基于历史均值估计: 约 **{mean_demand:.0f}** 单/小时\n"
    result += f"  - 历史波动范围: {daily.min():.0f} ~ {daily.max():.0f} 单/小时\n"
    result += f"  - 标准差: {std_demand:.1f}\n"
    result += f"\n[参考图表] outputs/M3_A_timeseries_comparison.png"
    return result

# ============================================================
# 问题类型 4: 费用估算
# ============================================================
def handle_fare_estimate(question):
    """
    匹配示例:
    - "从区域161到区域237大概多少钱"
    - "5英里路程大概费用"
    - "区域161到机场费用多少"
    """
    pu = parse_zone(question)
    # 尝试找第二个区域
    zones = re.findall(r'区域\s*(\d+)', question)
    do = int(zones[1]) if len(zones) > 1 else None
    
    # 距离模式
    dist_match = re.search(r'(\d+(?:\.\d+)?)\s*英里', question)
    dist = float(dist_match.group(1)) if dist_match else None
    
    if pu and do:
        mask = (DF['PULocationID'] == pu) & (DF['DOLocationID'] == do)
        if mask.sum() == 0:
            return f"区域 {pu} → {do} 的历史订单不足，无法估算。"
        fare = DF.loc[mask, 'total_amount']
        dist_val = DF.loc[mask, 'trip_distance']
        result = f"**区域 {pu} → {do} 费用估算**:\n"
        result += f"  - 平均总费用: **${fare.mean():.2f}**\n"
        result += f"  - 中位数费用: ${fare.median():.2f}\n"
        result += f"  - 平均距离: {dist_val.mean():.2f} 英里\n"
        result += f"  - 费用范围: ${fare.quantile(0.1):.2f} ~ ${fare.quantile(0.9):.2f}\n"
    elif dist:
        mask = (DF['trip_distance'] >= dist - 0.5) & (DF['trip_distance'] <= dist + 0.5)
        if mask.sum() == 0:
            return f"距离约 {dist} 英里的历史订单不足，无法估算。"
        fare = DF.loc[mask, 'total_amount']
        result = f"**约 {dist} 英里行程费用估算**:\n"
        result += f"  - 平均总费用: **${fare.mean():.2f}**\n"
        result += f"  - 中位数费用: ${fare.median():.2f}\n"
        result += f"  - 费用范围: ${fare.quantile(0.1):.2f} ~ ${fare.quantile(0.9):.2f}\n"
    elif pu:
        mask = DF['PULocationID'] == pu
        fare = DF.loc[mask, 'total_amount']
        dist_val = DF.loc[mask, 'trip_distance']
        result = f"**区域 {pu} 出发的平均费用**:\n"
        result += f"  - 平均总费用: **${fare.mean():.2f}**\n"
        result += f"  - 平均距离: {dist_val.mean():.2f} 英里\n"
    else:
        return None
    
    result += f"\n[参考图表] outputs/M2-3a_distance_fare_scatter.png"
    return result

# ============================================================
# 问题类型 5: 速度/拥堵查询
# ============================================================
def handle_speed_query(question):
    """
    匹配示例:
    - "区域161早高峰平均速度多少"
    - "什么时候最堵"
    - "平均速度最快的时间段"
    """
    zone = parse_zone(question)
    
    if '最堵' in question or '最慢' in question or '拥堵' in question:
        # 查询最拥堵的时段
        if zone:
            hourly_speed = DF[DF['PULocationID']==zone].groupby('pickup_hour')['avg_speed_mph'].median()
        else:
            hourly_speed = DF.groupby('pickup_hour')['avg_speed_mph'].median()
        worst_hour = hourly_speed.idxmin()
        worst_speed = hourly_speed.min()
        result = f"**最拥堵时段**: {worst_hour}:00，中位平均速度仅 **{worst_speed:.1f} mph**。"
        result += f"\n[参考图表] outputs/M2-4a_speed_heatmap.png"
        return result
    
    if '最快' in question or '最通畅' in question:
        if zone:
            hourly_speed = DF[DF['PULocationID']==zone].groupby('pickup_hour')['avg_speed_mph'].median()
        else:
            hourly_speed = DF.groupby('pickup_hour')['avg_speed_mph'].median()
        best_hour = hourly_speed.idxmax()
        best_speed = hourly_speed.max()
        result = f"**最通畅时段**: {best_hour}:00，中位平均速度达 **{best_speed:.1f} mph**。"
        return result
    
    # 特定区域/时段速度查询
    hour = parse_hour(question)
    if zone and hour is not None:
        mask = (DF['PULocationID'] == zone) & (DF['pickup_hour'] == hour)
        speed = DF.loc[mask, 'avg_speed_mph'].median()
        result = f"区域 {zone} 在 {hour}:00 的中位平均速度为 **{speed:.1f} mph**。"
        return result
    elif zone:
        speed = DF[DF['PULocationID']==zone]['avg_speed_mph'].median()
        result = f"区域 {zone} 的整体中位平均速度为 **{speed:.1f} mph**。"
        return result
    elif hour is not None:
        speed = DF[DF['pickup_hour']==hour]['avg_speed_mph'].median()
        result = f"全区域在 {hour}:00 的中位平均速度为 **{speed:.1f} mph**。"
        return result
    
    return None

# ============================================================
# 问题类型 6: 数据概览/通用统计
# ============================================================
def handle_overview(question):
    """
    匹配示例:
    - "数据集有多大"
    - "有多少个区域"
    - "总订单量"
    """
    if any(k in question for k in ['多大', '规模', '总量', '多少条', 'overview', 'summary']):
        result = f"**数据集概览**:\n"
        result += f"  - 总订单数: **{len(DF):,}** 条\n"
        result += f"  - 时间范围: 2023-01-01 ~ 2023-01-31\n"
        result += f"  - 上车区域数: {DF['PULocationID'].nunique()} 个\n"
        result += f"  - 下车区域数: {DF['DOLocationID'].nunique()} 个\n"
        result += f"  - 平均行程距离: {DF['trip_distance'].mean():.2f} 英里\n"
        result += f"  - 平均总费用: ${DF['total_amount'].mean():.2f}\n"
        return result
    return None

# ============================================================
# 规则路由器
# ============================================================
def rule_based_answer(question):
    """尝试用规则匹配回答问题"""
    q = question.lower()
    
    # 类型 6: 数据概览
    if any(k in q for k in ['概览', ' overview', 'summary', '数据集', '总共多少', '规模', '基本信息', '订单量', '总订单']):
        return handle_overview(question)
    
    # 类型 2: 区域排名
    if any(k in q for k in ['top', '排名', '最热门', '最多', '前几', '哪些区域', '最高', '热门']):
        return handle_zone_ranking(question)
    
    # 类型 5: 速度/拥堵
    if any(k in q for k in ['速度', '拥堵', '堵', '畅通', 'mph', '快慢']):
        return handle_speed_query(question)
    
    # 类型 4: 费用估算（放在需求预测之前，避免"大概多少钱"被误匹配为预测）
    if any(k in q for k in ['多少钱', '费用', '价格', 'fare', 'price', 'cost', '到.*多少']):
        return handle_fare_estimate(question)
    
    # 类型 3: 需求预测（含"预测"、"会有多少"）
    if any(k in q for k in ['预测', '会有多少', '估计', 'forecast', 'predict']):
        # 排除已包含"钱/费用/价格"的问题（已由费用估算处理）
        if not any(k in q for k in ['钱', '费用', '价格', 'fare', 'price', 'cost']):
            return handle_demand_prediction(question)
    
    # 类型 1: 时段查询（兜底，如果包含小时或时间点）
    if parse_hour(question) is not None or any(k in q for k in ['点', '时', '上午', '下午', '晚上', '凌晨']):
        return handle_time_query(question)
    
    return None

# ============================================================
# Kimi 大模型回复
# ============================================================
def kimi_answer(question):
    """调用 Kimi API 进行回答"""
    if KIMI_CLIENT is None:
        return "[系统] Kimi API 未配置，无法回答此问题。请输入有效的 MOONSHOT_API_KEY。"
    
    try:
        response = KIMI_CLIENT.chat.completions.create(
            model=KIMI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[错误] Kimi API 调用失败: {str(e)}"

# ============================================================
# 主循环
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("NYC Yellow Taxi 智能问答系统")
    print("支持: 时段查询 | 区域排名 | 需求预测 | 费用估算 | 速度/拥堵查询")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60 + "\n")
    
    while True:
        try:
            question = input("[用户] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        
        if not question:
            continue
        if question.lower() in ('quit', 'exit', '退出', 'q'):
            print("再见！")
            break
        
        # 先尝试规则匹配
        answer = rule_based_answer(question)
        
        if answer:
            print(f"\n[系统 - 规则匹配] {answer}\n")
        else:
            # 规则无法匹配，调用 Kimi
            print("\n[系统 - 规则未匹配，调用 Kimi 大模型...]")
            answer = kimi_answer(question)
            print(f"\n[Kimi] {answer}\n")

if __name__ == '__main__':
    main()
