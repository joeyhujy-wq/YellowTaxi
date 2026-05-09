"""
方案 A: 区域 161 小时级需求量预测
=====================================
- 目标: 预测区域 161 每个小时的出租车订单量
- 样本量: ~720 条（31 天 x 24 小时，扣除滞后缺失）
- 划分: 时间顺序 8:2
- 对比: PyTorch 神经网络 vs 随机森林
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 中文字体设置
# ============================================================
font_path = r'C:\Windows\Fonts\msyh.ttc'
fp_title = FontProperties(fname=font_path, size=14, weight='bold')
fp_label = FontProperties(fname=font_path, size=12)
fp_tick = FontProperties(fname=font_path, size=10)
plt.rcParams['axes.unicode_minus'] = False

def apply_fonts(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontproperties=fp_title)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=fp_label)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=fp_label)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fp_tick)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(fp_tick)

# ============================================================
# 1. 数据准备：区域 161 小时级需求量
# ============================================================
print("=" * 60)
print("方案 A: 区域 161 小时级需求量预测")
print("=" * 60)

df = pd.read_parquet('data/yellow_tripdata_2023-01_cleaned.parquet')

# 筛选区域 161
df_zone = df[df['PULocationID'] == 161].copy()
print(f"区域 161 原始记录数: {len(df_zone)}")

# 按天+小时聚合
df_zone['pickup_date'] = df_zone['tpep_pickup_datetime'].dt.date
hourly_demand = df_zone.groupby(['pickup_date', 'pickup_hour']).size().reset_index(name='demand')
hourly_demand['pickup_date'] = pd.to_datetime(hourly_demand['pickup_date'])

# 补全缺失的小时（某些小时可能无订单，需求量为 0）
full_range = pd.date_range(start=hourly_demand['pickup_date'].min(),
                           end=hourly_demand['pickup_date'].max(),
                           freq='H')
full_df = pd.DataFrame({'datetime': full_range})
full_df['pickup_date'] = full_df['datetime'].dt.date
full_df['pickup_hour'] = full_df['datetime'].dt.hour
full_df['pickup_date'] = pd.to_datetime(full_df['pickup_date'])

hourly_demand = full_df.merge(hourly_demand, on=['pickup_date', 'pickup_hour'], how='left')
hourly_demand['demand'] = hourly_demand['demand'].fillna(0).astype(int)

print(f"小时级样本数（含补零）: {len(hourly_demand)}")
print(f"平均每小时需求量: {hourly_demand['demand'].mean():.1f}")

# ============================================================
# 2. 特征工程
# ============================================================
print("\n" + "=" * 60)
print("2. 特征工程")
print("=" * 60)

hourly_demand['dayofweek'] = hourly_demand['datetime'].dt.dayofweek
hourly_demand['day'] = hourly_demand['datetime'].dt.day
hourly_demand['is_weekend'] = (hourly_demand['dayofweek'] >= 5).astype(int)
hourly_demand['is_rush_hour'] = (
    (hourly_demand['dayofweek'] < 5) &
    (hourly_demand['pickup_hour'].isin([7, 8, 9, 17, 18, 19]))
).astype(int)

# 滞后特征（跨天连续）
for lag in [1, 2, 3, 6, 12, 24]:
    hourly_demand[f'demand_lag_{lag}'] = hourly_demand['demand'].shift(lag)

# 滑动窗口统计
hourly_demand['demand_roll_mean_3'] = hourly_demand['demand'].shift(1).rolling(3).mean()
hourly_demand['demand_roll_mean_6'] = hourly_demand['demand'].shift(1).rolling(6).mean()
hourly_demand['demand_roll_std_3'] = hourly_demand['demand'].shift(1).rolling(3).std()

# 丢弃 NaN
hourly_demand = hourly_demand.dropna().reset_index(drop=True)
print(f"特征构造后样本数: {len(hourly_demand)}")

feature_cols = [
    'pickup_hour', 'dayofweek', 'day', 'is_weekend', 'is_rush_hour',
    'demand_lag_1', 'demand_lag_2', 'demand_lag_3',
    'demand_lag_6', 'demand_lag_12', 'demand_lag_24',
    'demand_roll_mean_3', 'demand_roll_mean_6', 'demand_roll_std_3'
]

X = hourly_demand[feature_cols].values
y = hourly_demand['demand'].values.reshape(-1, 1)

print(f"特征维度: {X.shape[1]}")
print(f"特征列表: {feature_cols}")

# ============================================================
# 3. 划分训练/测试集（时间顺序 8:2）
# ============================================================
print("\n" + "=" * 60)
print("3. 数据集划分（8:2，时间顺序）")
print("=" * 60)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
test_dates = hourly_demand['datetime'].iloc[split_idx:].values

print(f"训练集: {len(X_train)} 条")
print(f"测试集: {len(X_test)} 条")
print(f"测试时间范围: {hourly_demand['datetime'].iloc[split_idx]} ~ {hourly_demand['datetime'].iloc[-1]}")

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# ============================================================
# 4. PyTorch 神经网络
# ============================================================
print("\n" + "=" * 60)
print("4. PyTorch 神经网络")
print("=" * 60)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class HourlyDemandNet(nn.Module):
    def __init__(self, input_dim):
        super(HourlyDemandNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

model = HourlyDemandNet(X_train_scaled.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

epochs = 1000
train_losses = []

print("开始训练...")
model.train()
for epoch in range(epochs):
    epoch_losses = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    avg_loss = np.mean(epoch_losses)
    train_losses.append(avg_loss)
    scheduler.step()
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

# Loss 曲线
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, linewidth=2, color='steelblue')
apply_fonts(ax, title='PyTorch NN: 训练 Loss 曲线', xlabel='Epoch', ylabel='MSE Loss')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/M3_A_nn_loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 测试集预测
model.eval()
with torch.no_grad():
    y_pred_nn_scaled = model(X_test_tensor).numpy()
y_pred_nn = scaler_y.inverse_transform(y_pred_nn_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

mae_nn = mean_absolute_error(y_test_actual, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test_actual, y_pred_nn))

print(f"\n神经网络测试结果:")
print(f"  MAE = {mae_nn:.2f}")
print(f"  RMSE = {rmse_nn:.2f}")

# ============================================================
# 5. 随机森林
# ============================================================
print("\n" + "=" * 60)
print("5. 随机森林")
print("=" * 60)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train.ravel())
y_pred_rf = rf_model.predict(X_test).reshape(-1, 1)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"\n随机森林测试结果:")
print(f"  MAE = {mae_rf:.2f}")
print(f"  RMSE = {rmse_rf:.2f}")

# 特征重要性
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\n随机森林 TOP 10 特征重要性:")
print(importance_df.head(10).to_string(index=False))

# ============================================================
# 6. 结果对比与可视化
# ============================================================
print("\n" + "=" * 60)
print("6. 结果对比")
print("=" * 60)

results = pd.DataFrame({
    '模型': ['PyTorch NN', 'Random Forest'],
    'MAE': [mae_nn, mae_rf],
    'RMSE': [rmse_nn, rmse_rf]
})
print(results.to_string(index=False))

# 时间序列预测对比图
fig, ax = plt.subplots(figsize=(14, 5))
test_hours = range(len(y_test))
ax.plot(test_hours, y_test, 'o-', label='真实值', color='black', linewidth=1.5, markersize=4)
ax.plot(test_hours, y_pred_nn, 's--', label='PyTorch NN', color='steelblue', linewidth=1.5, markersize=4)
ax.plot(test_hours, y_pred_rf, '^--', label='Random Forest', color='forestgreen', linewidth=1.5, markersize=4)
apply_fonts(ax, title='区域 161 小时级需求量预测对比（测试集）',
            xlabel='测试集时间步', ylabel='需求量')
ax.legend(prop=fp_tick)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/M3_A_timeseries_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 散点对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test, y_pred_nn, alpha=0.6, edgecolors='k', s=60, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='完美预测线')
apply_fonts(axes[0], title=f'PyTorch NN\nMAE={mae_nn:.2f}, RMSE={rmse_nn:.2f}',
            xlabel='真实需求量', ylabel='预测需求量')
axes[0].legend(prop=fp_tick)

axes[1].scatter(y_test, y_pred_rf, alpha=0.6, edgecolors='k', s=60, color='forestgreen')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='完美预测线')
apply_fonts(axes[1], title=f'Random Forest\nMAE={mae_rf:.2f}, RMSE={rmse_rf:.2f}',
            xlabel='真实需求量', ylabel='预测需求量')
axes[1].legend(prop=fp_tick)

plt.tight_layout()
plt.savefig('outputs/M3_A_prediction_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 指标对比柱状图
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(2)
width = 0.35
bars1 = ax.bar(x - width/2, [mae_nn, mae_rf], width, label='MAE', color='steelblue')
bars2 = ax.bar(x + width/2, [rmse_nn, rmse_rf], width, label='RMSE', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(['PyTorch NN', 'Random Forest'], fontproperties=fp_tick)
apply_fonts(ax, title='模型性能对比（越低越好）', ylabel='误差值')
ax.legend(prop=fp_tick)
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/M3_A_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 7. 优劣分析
# ============================================================
print("\n" + "=" * 60)
print("7. 两种方法优劣分析")
print("=" * 60)

analysis_text = """
【PyTorch 神经网络】
优势：
  1. 非线性拟合能力强：通过多层 ReLU + Dropout，可捕捉小时、星期、滞后需求间的复杂交互。
  2. 扩展性好：未来可轻松引入 Embedding（区域 ID）、LSTM/Transformer 等时序模块。
  3. 端到端优化：基于梯度下降统一优化，适合大规模数据与在线学习。
劣势：
  1. 对数据量仍敏感：本任务约 500+ 训练样本虽可用，但需仔细调参避免过拟合。
  2. 超参数敏感：学习率、网络层数、批次大小需调优，训练过程有波动。
  3. 可解释性弱：黑盒模型，难以直观解释特征贡献。

【随机森林】
优势：
  1. 对 tabular 数据极其友好：在结构化特征上通常表现稳定，调参成本低。
  2. 免标准化：无需特征缩放，工程成本低。
  3. 可解释性强：天然输出特征重要性，可直观看到 lag_1、lag_24 等关键滞后特征。
  4. 训练速度快：并行建树，调参与迭代效率高。
劣势：
  1. 难以捕捉复杂非线性交互：虽然能处理非线性，但对特征间深层组合的建模能力弱于 NN。
  2. 外推能力有限：对超出训练分布的极端值预测保守。
  3. 时序依赖需人工构造：原生无记忆能力，lag 特征必须手工设计。

【方案 A 适配性结论】
  - 小时级预测将样本量提升至 ~720 条，两种模型都能有效学习。
  - 在此规模 tabular 数据上，随机森林与神经网络性能接近，但随机森林通常更稳定、调参更少。
  - 若未来扩展至多月、多区域数据，神经网络配合时序结构（LSTM/Transformer）将展现更强的
    规模化与非线性建模潜力。
"""
print(analysis_text)

# 保存报告
with open('M3_A_model_comparison_analysis.md', 'w', encoding='utf-8') as f:
    f.write("# 方案 A: 区域 161 小时级需求量预测 —— NN vs RF 对比\n\n")
    f.write("## 任务设定\n")
    f.write(f"- 目标变量: 区域 161 每小时出行需求量\n")
    f.write(f"- 样本量: 训练集 {len(X_train)} 条，测试集 {len(X_test)} 条\n")
    f.write(f"- 特征维度: {len(feature_cols)} 维\n\n")
    f.write("## 性能对比\n\n")
    f.write(results.to_markdown(index=False))
    f.write("\n\n")
    f.write("## 特征重要性 (Random Forest TOP 10)\n\n")
    f.write(importance_df.head(10).to_markdown(index=False))
    f.write("\n\n")
    f.write("## 优劣分析\n")
    f.write(analysis_text)

print("\n分析结果已保存至 M3_A_model_comparison_analysis.md")
print("图表已保存至 outputs/M3_A_*.png")
