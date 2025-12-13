# Monte Carlo Simulator Web

基于 Web 的蒙特卡罗模拟工具，用于风险分析和不确定性量化。

## 特性

- 支持 9 种概率分布（正态、t、对数正态、均匀、三角、Beta、Gamma、指数、Weibull）
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

## 许可证

MIT License
