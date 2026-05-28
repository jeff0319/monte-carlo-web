# Monte Carlo Simulator Web

基于 Web 的蒙特卡罗模拟工具，用于风险分析和不确定性量化。

## 特性

- 支持 9 种概率分布（正态、t、对数正态、均匀、三角、Beta、Gamma、指数、Weibull）
- 原始数据输入支持两种抽样方式：拟合理论分布，或使用 non-parametric bootstrap 从原始数据有放回抽样
- 自定义数学公式和 Python 函数
- 敏感性分析（Pareto 图、Tornado 图）
- 可视化报告生成
- 多用户 Session 隔离

## 技术栈

- 后端：Flask + NumPy + SciPy + Matplotlib
- 前端：HTML/CSS/JavaScript

## 快速开始

### Docker Compose（推荐）

```bash
docker-compose up -d
# 访问 http://localhost:5050
```

### 本地运行

```bash
cd backend
pip install -r requirements.txt
python app.py
```

## 原始数据输入

`data` 模式默认使用 `bootstrap`，不强制拟合理论分布。若需要旧的拟合方式，可以设置 `sampling_method` 为 `fit` 并提供 `distribution`：

```json
{
  "需求量": {
    "type": "data",
    "values": [82, 91, 88, 105, 97, 76, 110, 94, 89, 101],
    "sampling_method": "bootstrap",
    "bootstrap_statistic": "mean",
    "min_limit": 70,
    "max_limit": 120
  }
}
```

bootstrap 会先按 `min_limit`/`max_limit` 过滤原始数据，再抽样。`bootstrap_statistic: "mean"` 表示每轮有放回抽取原始样本量的数据并取均值，适合计算均值估计量的不确定性；`bootstrap_statistic: "value"` 表示每轮直接抽取一个原始观测值，适合传播单次观测值的经验分布。当前 bootstrap 按变量独立抽样，不保留多个变量原始数据之间的行配对关系。

## 许可证

MIT License
