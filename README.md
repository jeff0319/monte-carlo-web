# Monte Carlo Simulator Web 应用

一个基于 Web 的蒙特卡罗模拟工具，用于风险分析和不确定性量化。支持多变量输入、自定义公式计算、敏感性分析和可视化报告生成。

## 项目特性

- 📊 **灵活的变量输入**
  - 支持原始数据输入（自动拟合分布）
  - 支持统计参数输入（均值、标准差）
  - 支持正态分布和 t 分布
  - 可设置采样范围限制

- 🧮 **强大的计算功能**
  - 自定义数学公式（支持基本运算和数学函数）
  - 自定义 Python 函数（高级用户）
  - 大规模蒙特卡罗抽样（默认 100 万次）
  - CDF 拟合和插值

- 📈 **丰富的可视化分析**
  - 结果分布直方图（带 PDF 和 CDF）
  - 变量分布对比图
  - Pareto 图（变量重要性分析）
  - Tornado 图（敏感性分析）

- 📄 **报告生成**
  - 统计摘要（均值、中位数、标准差等）
  - 置信区间分析
  - 图表导出

- 👥 **多用户支持**
  - Session 隔离（每个用户独立的模拟环境）
  - 自动清理不活跃用户数据（1 小时）

## 技术栈

### 后端
- **Flask** 3.0.0 - Web 框架
- **NumPy** 1.24.3 - 数值计算
- **SciPy** 1.11.2 - 科学计算和统计
- **Matplotlib** 3.7.2 - 数据可视化
- **Seaborn** 0.12.2 - 统计图表

### 前端
- 纯 HTML/CSS/JavaScript（无框架依赖）
- 现代化 UI 设计
- 响应式布局

## 项目结构

```
monte-carlo-web/
├── backend/
│   ├── app.py              # Flask API 服务
│   ├── monte_carlo.py      # 蒙特卡罗核心算法
│   └── requirements.txt    # Python 依赖
├── frontend/
│   └── index.html          # Web 界面
├── Dockerfile              # Docker 镜像配置
├── docker-compose.yml      # Docker Compose 配置
├── .gitignore             # Git 忽略文件
└── README.md              # 项目文档
```

## 快速开始

### 方式 1: Docker Compose（推荐）

```bash
# 克隆项目
git clone <your-repo-url>
cd monte-carlo-web

# 启动服务
docker-compose up -d

# 访问应用
open http://localhost:5050
```

### 方式 2: 本地运行

#### 前置要求
- Python 3.10+
- pip

#### 安装步骤

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd monte-carlo-web

# 2. 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows

# 3. 安装依赖
cd backend
pip install -r requirements.txt

# 4. 启动服务
python app.py

# 5. 访问应用
# 浏览器打开 http://localhost:5050
```

## 使用指南

### 1. 添加变量

#### 方式 A: 单个添加（图形界面）
1. 输入变量名（如 `revenue`）
2. 选择输入方式：
   - **原始数据**：粘贴测试数据（逗号或空格分隔）
   - **统计参数**：输入均值和标准差
3. 选择分布类型（正态分布或 t 分布）
4. （可选）设置最小值和最大值限制
5. 点击"添加变量"

#### 方式 B: JSON 批量导入
```json
[
  {
    "name": "revenue",
    "input_mode": "params",
    "mean": 1000000,
    "std": 50000,
    "dist_type": "normal"
  },
  {
    "name": "cost",
    "input_mode": "data",
    "data": [800000, 850000, 820000, 900000],
    "dist_type": "normal"
  }
]
```

### 2. 运行模拟

#### 简单公式模式
在公式输入框输入表达式，例如：
```
revenue - cost
revenue * 0.15
sqrt(var_A^2 + var_B^2)
```

支持的数学函数：
- 基本运算：`+`、`-`、`*`、`/`、`^`（幂）
- 三角函数：`sin()`、`cos()`、`tan()`
- 指数对数：`exp()`、`log()`、`log10()`、`ln()`、`sqrt()`

#### 自定义函数模式
```python
def custom_formula(vars):
    revenue = vars['revenue']
    cost = vars['cost']
    
    # 复杂计算逻辑
    if revenue > 1000000:
        profit = (revenue - cost) * 0.85
    else:
        profit = (revenue - cost) * 0.90
    
    return profit
```

### 3. 分析结果

运行模拟后，系统会生成：
- **统计摘要**：均值、中位数、标准差、偏度、峰度
- **置信区间**：95%、99% 等置信水平的区间估计
- **可视化图表**：
  - 结果分布图（直方图 + PDF + CDF）
  - 变量分布对比
  - Pareto 图（识别关键变量）
  - Tornado 图（敏感性排名）

### 4. 生成报告

点击"生成报告"按钮，获取包含所有分析结果的文本报告。

## API 文档

### 变量管理

#### 添加单个变量
```http
POST /api/add_variable
Content-Type: application/json

{
  "name": "var_name",
  "input_mode": "params",
  "mean": 100,
  "std": 10,
  "dist_type": "normal",
  "min_value": null,
  "max_value": null
}
```

#### 批量添加变量
```http
POST /api/add_variables_json
Content-Type: application/json

{
  "variables": [...]
}
```

#### 列出所有变量
```http
GET /api/list_variables
```

### 模拟运行

#### 运行模拟
```http
POST /api/run_simulation
Content-Type: application/json

{
  "formula": "var_A + var_B",
  "result_name": "Result",
  "n_samples": 1000000,
  "cdf_fit_degree": 5,
  "use_custom_function": false,
  "custom_function_code": ""
}
```

#### 分析结果
```http
POST /api/analyze_result
Content-Type: application/json

{
  "confidence_levels": [0.95, 0.99]
}
```

#### 生成图表
```http
POST /api/generate_charts
Content-Type: application/json

{
  "chart_types": ["result_plot", "var_distributions", "pareto", "tornado"]
}
```

#### 生成报告
```http
POST /api/generate_report
Content-Type: application/json

{
  "confidence_levels": [0.95, 0.99]
}
```

### 系统管理

#### 重置模拟器
```http
POST /api/reset
```

### 测试

```bash
# 运行后端（开发模式）
cd backend
python app.py

# 测试 API
curl http://localhost:5050/api/list_variables
```

## 部署

### Docker 部署

```bash
# 构建镜像
docker build -t monte-carlo-web .

# 运行容器
docker run -d -p 5050:5050 --name monte-carlo monte-carlo-web
```

## 常见问题

### Q: 图表显示中文乱码怎么办？
A: 确保系统安装了中文字体（SimHei 或 Arial Unicode MS）。在 Docker 环境中，可能需要在 Dockerfile 中添加字体安装步骤。

### Q: 模拟速度慢怎么办？
A: 
1. 减少采样次数（`n_samples`）
2. 使用 Docker 部署（更稳定的性能）
3. 考虑使用更强大的服务器

### Q: 如何备份模拟数据？
A: 当前版本数据存储在内存中。如需持久化，可以：
1. 导出 JSON 配置
2. 保存生成的报告
3. 考虑添加数据库支持（未来版本）

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请联系 [jeff0319@gmail.com]

## 更新日志

### v1.0.0 (2025-10-26)
- 初始版本发布
- 支持基本的蒙特卡罗模拟功能
- Web 界面和 API
- Docker 支持
