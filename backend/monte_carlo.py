import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
from typing import Callable, List, Dict, Tuple, Optional
import warnings
import io
import base64
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Variable:
    """
    Monte Carlo模拟变量类
    管理单个变量的数据预处理、分布拟合和Monte Carlo抽样
    支持两种输入方式:
    1. 原始数据点
    2. 统计参数(均值和标准差)
    """
    
    def __init__(self, name: str, data: Optional[np.ndarray] = None, 
                 mean: Optional[float] = None, std: Optional[float] = None,
                 dist_type: str = 'norm', df: Optional[float] = None,
                 min_value: Optional[float] = None, max_value: Optional[float] = None):
        """
        初始化变量
        
        Parameters:
        -----------
        name : str
            变量名称
        data : np.ndarray, optional
            原始测试数据 (方式1)
        mean : float, optional
            均值 (方式2)
        std : float, optional
            标准差 (方式2)
        dist_type : str
            分布类型 ('norm' 或 't'),仅当使用方式2时需要指定
        df : float, optional
            自由度,仅当 dist_type='t' 且使用方式2时需要
        min_value : float, optional
            采样的最小值限制
        max_value : float, optional
            采样的最大值限制
        """
        self.name = name
        self.input_mode = None  # 'data' 或 'params'
        self.raw_data = None
        self.dist_type = None
        self.dist_params = None
        self.samples = None
        self.min_value = min_value
        self.max_value = max_value
        
        # 判断输入方式
        if data is not None:
            # 方式1: 原始数据
            self.input_mode = 'data'
            self.raw_data = np.array(data)
        elif mean is not None and std is not None:
            # 方式2: 统计参数
            self.input_mode = 'params'
            self.dist_type = dist_type
            
            if dist_type == 'norm':
                self.dist_params = (mean, std)
            elif dist_type == 't':
                if df is None:
                    raise ValueError("当使用t分布的统计参数时,必须提供自由度df")
                self.dist_params = (df, mean, std)
            else:
                raise ValueError("dist_type必须是'norm'或't'")
        else:
            raise ValueError("必须提供data或者(mean和std)")
        
    def fit_distribution(self, dist_type: str = 'norm'):
        """
        拟合分布参数 (仅适用于原始数据输入方式)
        
        Parameters:
        -----------
        dist_type: 'norm' 或 't'
        """
        if self.input_mode != 'data':
            return
        
        n = len(self.raw_data)
        
        if dist_type == 'norm':
            self.dist_type = 'norm'
            mu = np.mean(self.raw_data)
            sigma = np.std(self.raw_data, ddof=1)
            self.dist_params = (mu, sigma)
        
        elif dist_type == 't':
            self.dist_type = 't'
            df = n - 1
            loc = np.mean(self.raw_data)
            scale = np.std(self.raw_data, ddof=1)
            self.dist_params = (df, loc, scale)
        
        else:
            raise ValueError("dist_type必须是'norm'或't'")
    
    def monte_carlo_sample(self, n_samples: int = 1000000):
        """
        进行Monte Carlo抽样，支持限值约束（修正版本）
        """
        if self.dist_params is None:
            raise ValueError(f"请先对变量 {self.name} 进行分布拟合")
        
        # 无限值约束时直接采样
        if self.min_value is None and self.max_value is None:
            if self.dist_type == 'norm':
                self.samples = np.random.normal(
                    loc=self.dist_params[0],
                    scale=self.dist_params[1],
                    size=n_samples
                )
            elif self.dist_type == 't':
                self.samples = stats.t.rvs(
                    df=self.dist_params[0],
                    loc=self.dist_params[1],
                    scale=self.dist_params[2],
                    size=n_samples
                )
            return
        
        # 有限值约束时使用拒绝采样
        # 1. 计算接受率
        lower = self.min_value if self.min_value is not None else -np.inf
        upper = self.max_value if self.max_value is not None else np.inf
        
        if self.dist_type == 'norm':
            mean, std = self.dist_params[0], self.dist_params[1]
            acceptance_rate = stats.norm.cdf(upper, mean, std) - stats.norm.cdf(lower, mean, std)
        elif self.dist_type == 't':
            df, loc, scale = self.dist_params
            acceptance_rate = stats.t.cdf(upper, df, loc, scale) - stats.t.cdf(lower, df, loc, scale)
        
        acceptance_rate = max(acceptance_rate, 1e-10)
        
        # 2. 拒绝采样循环
        valid_samples = []
        attempts = 0
        max_attempts = 50
        
        while len(valid_samples) < n_samples and attempts < max_attempts:
            attempts += 1
            remaining = n_samples - len(valid_samples)
            
            # 过采样
            n_to_sample = int(remaining / acceptance_rate * 1.5)
            n_to_sample = min(n_to_sample, n_samples * 5)
            
            # 生成样本
            if self.dist_type == 'norm':
                new_samples = np.random.normal(
                    loc=self.dist_params[0],
                    scale=self.dist_params[1],
                    size=n_to_sample
                )
            elif self.dist_type == 't':
                new_samples = stats.t.rvs(
                    df=self.dist_params[0],
                    loc=self.dist_params[1],
                    scale=self.dist_params[2],
                    size=n_to_sample
                )
            
            # 过滤：只保留在限值范围内的样本
            valid_mask = np.ones(len(new_samples), dtype=bool)
            if self.min_value is not None:
                valid_mask &= (new_samples >= self.min_value)
            if self.max_value is not None:
                valid_mask &= (new_samples <= self.max_value)
            
            valid_samples.append(new_samples[valid_mask])
        
        # 3. 合并并精确截取
        if len(valid_samples) == 0:
            raise ValueError(f"变量 '{self.name}' 无法生成有效样本")
        
        all_samples = np.concatenate(valid_samples)
        
        if len(all_samples) >= n_samples:
            self.samples = all_samples[:n_samples]
        else:
            raise ValueError(
                f"变量 '{self.name}' 只生成了 {len(all_samples)}/{n_samples} 个样本"
            )
    
    def plot_distribution(self, figsize=(12, 5), show_samples=True):
        """
        绘制分布图,返回base64编码的图片
        """
        if self.dist_params is None:
            raise ValueError(f"请先对变量 {self.name} 进行分布拟合")
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 确定x轴范围
        if self.input_mode == 'data':
            x_min = self.raw_data.min() - 6 * self.raw_data.std()
            x_max = self.raw_data.max() + 6 * self.raw_data.std()
        else:
            mean = self.dist_params[1] if self.dist_type == 't' else self.dist_params[0]
            std = self.dist_params[2] if self.dist_type == 't' else self.dist_params[1]
            x_min = mean - 6 * std
            x_max = mean + 6 * std
        
        x_range = np.linspace(x_min, x_max, 200)
        
        # 绘制拟合的分布曲线
        if self.dist_type == 'norm':
            pdf = stats.norm.pdf(x_range, self.dist_params[0], self.dist_params[1])
            ax.plot(x_range, pdf, 'r-', lw=2.5, 
                   label=f'Normal Distribution\nμ={self.dist_params[0]:.3f}, σ={self.dist_params[1]:.3f}', 
                   zorder=3)
        else:
            pdf = stats.t.pdf(x_range, self.dist_params[0], self.dist_params[1], self.dist_params[2])
            ax.plot(x_range, pdf, 'r-', lw=2.5, 
                   label=f't Distribution\ndf={self.dist_params[0]:.2f}, loc={self.dist_params[1]:.3f}, scale={self.dist_params[2]:.3f}', 
                   zorder=3)
        
        # 如果有Monte Carlo样本且需要显示
        has_samples = self.samples is not None and show_samples
        if has_samples:
            ax.hist(self.samples, bins=100, density=True, alpha=0.3, 
                   label='Monte Carlo Samples', color='green', edgecolor='darkgreen', 
                   linewidth=0.5, zorder=1)
        
        # 显示原始数据
        if self.input_mode == 'data':
            if len(self.raw_data) <= 30:
                max_pdf = np.max(pdf)
                y_base = 0
                y_jitter = np.random.normal(0, 0.005 * max_pdf, len(self.raw_data))
                y_positions = y_base + y_jitter
                ax.scatter(self.raw_data, y_positions, alpha=0.5, s=60, 
                           label='Raw Data', color='steelblue', zorder=5)
                ax.set_ylim(y_base - 0.05 * max_pdf, max_pdf * 1.1)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
            else:
                ax.hist(self.raw_data, bins=min(30, len(self.raw_data)//3), 
                       density=True, alpha=0.5, label='Raw Data', 
                       color='steelblue', edgecolor='darkblue', zorder=2)
        
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'{self.name} - Distribution Fit{" and MC Samples" if has_samples else ""}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64


class MonteCarloSimulator:
    """
    Monte Carlo模拟器主类
    """
    
    def __init__(self):
        self.variables = {}
        self.result = None
        self.result_name = None
        self.confidence_levels = [0.95, 0.99]
        self.sensitivity_results = None
        self.formula_str = None
        # 图表缓存
        self.chart_cache = {}
        # CDF拟合相关
        self.cdf_fit_degree = 5
        self.cdf_fit_result = None
    
    def add_variable(self, name: str, data: Optional[List[float]] = None, 
                     mean: Optional[float] = None, std: Optional[float] = None,
                     dist_type: str = 'normal', df: Optional[float] = None,
                     min_value: Optional[float] = None, max_value: Optional[float] = None):
        """
        添加变量
        
        Parameters:
        -----------
        dist_type: 'normal' 或 't' (会自动转换为内部使用的 'norm' 或 't')
        """
        # 将 'normal' 转换为内部使用的 'norm'
        if dist_type == 'normal':
            internal_dist = 'norm'
        elif dist_type == 'norm':
            internal_dist = 'norm'  # 兼容旧代码
        elif dist_type == 't':
            internal_dist = 't'
        else:
            raise ValueError("dist_type必须是'normal'或't'")
        
        if data is not None:
            var = Variable(name, data=np.array(data), min_value=min_value, max_value=max_value)
            var.fit_distribution(internal_dist)
        elif mean is not None and std is not None:
            var = Variable(name, mean=mean, std=std, dist_type=internal_dist, df=df, 
                         min_value=min_value, max_value=max_value)
        else:
            raise ValueError("必须提供data或者(mean和std)")
        
        self.variables[name] = var
    
    def add_variables_from_json(self, json_data: dict):
        """
        从JSON批量添加变量
        
        Parameters:
        -----------
        json_data : dict
            JSON格式的变量定义
            
        示例格式:
        {
            "var_A": {
                "type": "data",
                "values": [10.5, 11.2, 9.8],
                "distribution": "normal",
                "min_limit": 8.0,
                "max_limit": 15.0
            },
            "var_B": {
                "type": "params",
                "mean": 100,
                "std": 5,
                "distribution": "t",
                "df": 20,
                "min_limit": 85,
                "max_limit": 115
            }
        }
        """
        for var_name, var_config in json_data.items():
            # 验证必填字段
            if 'type' not in var_config:
                raise ValueError(f"变量 '{var_name}' 缺少必填字段 'type'")
            if 'distribution' not in var_config:
                raise ValueError(f"变量 '{var_name}' 缺少必填字段 'distribution'")
            
            var_type = var_config['type']
            distribution = var_config['distribution']
            
            # 验证分布类型
            if distribution not in ['normal', 't']:
                raise ValueError(f"变量 '{var_name}' 的 distribution 必须是 'normal' 或 't'，得到: '{distribution}'")
            
            # 将 'normal' 转换为内部使用的 'norm'
            internal_dist = 'norm' if distribution == 'normal' else 't'
            
            # 获取可选的限值
            min_limit = var_config.get('min_limit', None)
            max_limit = var_config.get('max_limit', None)
            
            if var_type == 'data':
                # Data 类型变量
                if 'values' not in var_config:
                    raise ValueError(f"变量 '{var_name}' type为'data'时必须提供 'values' 字段")
                
                # 检测错误：data类型不应该有mean/std/df
                if any(k in var_config for k in ['mean', 'std', 'df']):
                    raise ValueError(f"变量 '{var_name}' type为'data'时不应该提供 'mean', 'std' 或 'df' 字段")
                
                values = var_config['values']
                if not isinstance(values, list) or len(values) == 0:
                    raise ValueError(f"变量 '{var_name}' 的 'values' 必须是非空数组")
                
                self.add_variable(
                    name=var_name,
                    data=values,
                    dist_type=internal_dist,
                    min_value=min_limit,
                    max_value=max_limit
                )
                
            elif var_type == 'params':
                # Params 类型变量
                if 'mean' not in var_config or 'std' not in var_config:
                    raise ValueError(f"变量 '{var_name}' type为'params'时必须提供 'mean' 和 'std' 字段")
                
                # 检测错误：params类型不应该有values
                if 'values' in var_config:
                    raise ValueError(f"变量 '{var_name}' type为'params'时不应该提供 'values' 字段")
                
                mean = var_config['mean']
                std = var_config['std']
                df = None
                
                if distribution == 't':
                    if 'df' not in var_config:
                        raise ValueError(f"变量 '{var_name}' distribution为't'时必须提供 'df' (自由度) 字段")
                    df = var_config['df']
                
                self.add_variable(
                    name=var_name,
                    mean=mean,
                    std=std,
                    dist_type=internal_dist,
                    df=df,
                    min_value=min_limit,
                    max_value=max_limit
                )
            else:
                raise ValueError(f"变量 '{var_name}' 的 type 必须是 'data' 或 'params'，得到: '{var_type}'")
    
    def run_simulation(self, formula: Callable, result_name: str = 'Result', 
                      n_samples: int = 1000000, formula_str: str = '', cdf_fit_degree: int = 5):
        """
        运行Monte Carlo模拟
        
        Parameters:
        -----------
        cdf_fit_degree : int
            CDF多项式拟合的次数，默认为5
        """
        # 对所有变量进行抽样
        for var in self.variables.values():
            var.monte_carlo_sample(n_samples)
        
        # 构建变量字典
        var_dict = {name: var.samples for name, var in self.variables.items()}
        
        # 计算结果
        self.result = formula(var_dict)
        self.result_name = result_name
        self.formula_str = formula_str
        self.cdf_fit_degree = cdf_fit_degree
        
        # 执行敏感性分析
        self._perform_sensitivity_analysis(var_dict)
        
        # 执行CDF拟合
        self._fit_cdf_polynomial()
        
        # 清空图表缓存
        self.chart_cache = {}
    
    def analyze_result(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        分析结果分布,返回统计信息字典
        """
        if self.result is None:
            raise ValueError("请先运行模拟")
        
        self.confidence_levels = confidence_levels
        
        analysis = {
            'mean': float(np.mean(self.result)),
            'median': float(np.median(self.result)),
            'std': float(np.std(self.result)),
            'min': float(np.min(self.result)),
            'max': float(np.max(self.result)),
            'skew': float(stats.skew(self.result)),
            'kurtosis': float(stats.kurtosis(self.result)),
            'confidence_intervals': {},
            'quantiles': {}
        }
        
        for cl in confidence_levels:
            alpha = 1 - cl
            lower = float(np.percentile(self.result, alpha / 2 * 100))
            upper = float(np.percentile(self.result, (1 - alpha / 2) * 100))
            analysis['confidence_intervals'][f'{int(cl*100)}%'] = [lower, upper]
        
        for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
            analysis['quantiles'][f'{int(q*100)}%'] = float(np.percentile(self.result, q*100))
        
        return analysis
    
    def plot_result(self, figsize=(15, 10)):
        """
        绘制结果的多种统计图,返回base64编码的图片
        """
        if self.result is None:
            raise ValueError("请先运行模拟")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        mean_val = np.mean(self.result)
        median_val = np.median(self.result)
        std_val = np.std(self.result)
        
        # 1. 直方图和KDE
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(self.result, bins=100, density=True, alpha=0.6, 
                color='skyblue', edgecolor='black', label='Histogram')
        
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(self.result)
        x_range = np.linspace(self.result.min(), self.result.max(), 200)
        ax1.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
        
        ax1.axvline(x=median_val, color='darkred', linestyle='-', 
                linewidth=2.5, alpha=0.8, label=f'Median: {median_val:.4f}')
        ax1.axvline(x=mean_val, color='blue', linestyle=':', 
                linewidth=2, alpha=0.6, label=f'Mean: {mean_val:.4f}')
        
        q25 = np.percentile(self.result, 25)
        q75 = np.percentile(self.result, 75)
        ax1.axvline(x=q25, color='green', linestyle='--', linewidth=1.5, alpha=0.7, 
                label=f'Q1 (25%): {q25:.4f}')
        ax1.axvline(x=q75, color='green', linestyle='--', linewidth=1.5, alpha=0.7, 
                label=f'Q3 (75%): {q75:.4f}')
        
        if hasattr(self, 'confidence_levels') and self.confidence_levels:
            ci_colors = {0.90: 'orange', 0.95: 'purple', 0.99: 'red'}
            for conf_level in sorted(self.confidence_levels):
                if conf_level in ci_colors:
                    color = ci_colors[conf_level]
                    tail = (1 - conf_level) / 2 * 100
                    q_lower = np.percentile(self.result, tail)
                    q_upper = np.percentile(self.result, 100 - tail)
                    ax1.axvline(x=q_lower, color=color, linestyle='--', linewidth=1.5, alpha=0.6,
                            label=f'{int(conf_level*100)}% CI lower: {q_lower:.4f}')
                    ax1.axvline(x=q_upper, color=color, linestyle='--', linewidth=1.5, alpha=0.6,
                            label=f'{int(conf_level*100)}% CI upper: {q_upper:.4f}')
        
        ax1.set_xlabel(f'{self.result_name} Value', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title(f'{self.result_name} - Distribution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # 2. 箱形图
        ax2 = fig.add_subplot(gs[0, 2])
        bp = ax2.boxplot([self.result], vert=True, patch_artist=True,
                        labels=[self.result_name],
                        boxprops=dict(facecolor='lightgreen', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        from matplotlib.offsetbox import AnchoredText
        stats_text = f"Mean: {mean_val:.4f}\nMedian: {median_val:.4f}\nStd: {std_val:.4f}"
        anchored_text = AnchoredText(stats_text, loc='upper right', 
                                    frameon=True, pad=0.5, prop=dict(size=10))
        anchored_text.patch.set_boxstyle("round,pad=0.5")
        anchored_text.patch.set_alpha(0.9)
        ax2.add_artist(anchored_text)
        
        ax2.set_ylabel(f'{self.result_name} Value', fontsize=12)
        ax2.set_title(f'{self.result_name} - Box Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 累积分布函数
        ax3 = fig.add_subplot(gs[1, :])
        sorted_result = np.sort(self.result)
        cdf = np.arange(1, len(sorted_result) + 1) / len(sorted_result)
        ax3.plot(sorted_result, cdf, lw=2.5, color='purple', label='Empirical CDF')
        
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        colors = ['blue', 'green', 'red', 'green', 'blue']
        for q, color in zip(quantiles, colors):
            val = np.percentile(self.result, q * 100)
            ax3.axhline(y=q, color=color, linestyle='--', alpha=0.3, linewidth=1)
            ax3.axvline(x=val, color=color, linestyle='--', alpha=0.3, linewidth=1)
            ax3.plot(val, q, 'o', color=color, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
            offset_y = 0.03 if q < 0.9 else -0.05
            ax3.text(val, q + offset_y, f'{q*100:.0f}%: {val:.2f}', 
                    fontsize=10, ha='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
        
        ax3.set_xlabel(f'{self.result_name} Value', fontsize=13)
        ax3.set_ylabel('Cumulative Probability', fontsize=13)
        ax3.set_title(f'{self.result_name} - CDF', fontsize=15, fontweight='bold')
        ax3.legend(fontsize=11, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.05, 1.05)
        
        # 转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    
    def get_variable_info(self):
        """
        获取所有变量信息
        """
        info = {}
        for name, var in self.variables.items():
            info[name] = {
                'input_mode': var.input_mode,
                'dist_type': var.dist_type,
                'dist_params': var.dist_params,
                'has_samples': var.samples is not None,
                'min_value': var.min_value,
                'max_value': var.max_value
            }
        return info
    
    def _perform_sensitivity_analysis(self, var_samples: Dict[str, np.ndarray]):
        """
        执行敏感性分析（内部方法）
        使用 Spearman 秩相关系数来衡量变量对结果的影响
        """
        if self.result is None:
            raise ValueError("请先运行模拟")
        
        sensitivities = {}
        
        for var_name, var_data in var_samples.items():
            # 计算 Spearman 秩相关系数
            corr, p_value = stats.spearmanr(var_data, self.result)
            sensitivities[var_name] = {
                'correlation': float(corr),
                'abs_correlation': float(abs(corr)),
                'p_value': float(p_value)
            }
        
        # 按绝对值排序
        self.sensitivity_results = dict(
            sorted(sensitivities.items(), 
                   key=lambda x: x[1]['abs_correlation'], 
                   reverse=True)
        )
    
    def _fit_cdf_polynomial(self, n_points: int = 500):
        """
        对CDF曲线进行多项式拟合（内部方法）
        
        Parameters:
        -----------
        n_points : int
            拟合使用的等分点数量，默认500
        """
        if self.result is None:
            raise ValueError("请先运行模拟")
        
        # 排序结果
        sorted_result = np.sort(self.result)
        n_total = len(sorted_result)
        
        # 生成500个等分点的索引
        indices = np.linspace(0, n_total - 1, n_points, dtype=int)
        x_points = sorted_result[indices]
        
        # 对应的CDF值
        cdf_values = (indices + 1) / n_total
        
        # 多项式拟合
        coefficients = np.polyfit(x_points, cdf_values, self.cdf_fit_degree)
        
        # 计算拟合优度
        fitted_cdf = np.polyval(coefficients, x_points)
        
        # R² score
        ss_res = np.sum((cdf_values - fitted_cdf) ** 2)
        ss_tot = np.sum((cdf_values - np.mean(cdf_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # RMSE
        rmse = np.sqrt(np.mean((cdf_values - fitted_cdf) ** 2))
        
        # 保存结果
        self.cdf_fit_result = {
            'degree': self.cdf_fit_degree,
            'n_points': n_points,
            'coefficients': coefficients.tolist(),
            'r_squared': float(r_squared),
            'rmse': float(rmse)
        }
    
    def generate_report(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        生成文本格式的分析报告
        
        Parameters:
        -----------
        confidence_levels : List[float]
            置信水平列表
        
        Returns:
        --------
        str : 格式化的文本报告
        """
        if self.result is None:
            raise ValueError("请先运行模拟")
        
        # 分析结果
        analysis = self.analyze_result(confidence_levels)
        
        # 构建报告
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append(f"Monte Carlo Simulation Report - {self.result_name}")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # 基本统计信息
        report_lines.append("=== Basic Statistics ===")
        report_lines.append(f"Mean:             {analysis['mean']:.6f}")
        report_lines.append(f"Median:           {analysis['median']:.6f}")
        report_lines.append(f"Std Deviation:    {analysis['std']:.6f}")
        report_lines.append(f"Minimum:          {analysis['min']:.6f}")
        report_lines.append(f"Maximum:          {analysis['max']:.6f}")
        report_lines.append(f"Skewness:         {analysis['skew']:.6f}")
        report_lines.append(f"Kurtosis:         {analysis['kurtosis']:.6f}")
        report_lines.append("")
        
        # 置信区间
        report_lines.append("=== Confidence Intervals ===")
        for level, (lower, upper) in analysis['confidence_intervals'].items():
            report_lines.append(f"{level:>4}: [{lower:.6f}, {upper:.6f}]")
        report_lines.append("")
        
        # 分位数
        report_lines.append("=== Quantiles ===")
        for q, value in analysis['quantiles'].items():
            report_lines.append(f"{q:>4}: {value:.6f}")
        report_lines.append("")
        
        # CDF多项式拟合结果
        if self.cdf_fit_result:
            report_lines.append("=== CDF Polynomial Fit Analysis ===")
            report_lines.append(f"Degree: {self.cdf_fit_result['degree']}")
            report_lines.append(f"Number of fitting points: {self.cdf_fit_result['n_points']}")
            report_lines.append("")
            
            report_lines.append("Polynomial Coefficients (highest to lowest degree):")
            coeffs = self.cdf_fit_result['coefficients']
            for i, coeff in enumerate(coeffs):
                power = self.cdf_fit_result['degree'] - i
                report_lines.append(f"  x^{power}: {coeff:.6e}")
            report_lines.append("")
            
            report_lines.append("Goodness of Fit:")
            report_lines.append(f"  R² Score: {self.cdf_fit_result['r_squared']:.6f}")
            report_lines.append(f"  RMSE: {self.cdf_fit_result['rmse']:.6f}")
            report_lines.append("")
            
            # 构建拟合函数字符串
            func_parts = []
            for i, coeff in enumerate(coeffs):
                power = self.cdf_fit_result['degree'] - i
                if power == 0:
                    func_parts.append(f"{coeff:.2e}")
                elif power == 1:
                    func_parts.append(f"{coeff:.2e}*x")
                else:
                    func_parts.append(f"{coeff:.2e}*x^{power}")
            
            func_str = " + ".join(func_parts).replace("+ -", "- ")
            report_lines.append("Fitted CDF Function:")
            report_lines.append(f"  CDF(x) = {func_str}")
            report_lines.append("")
        
        # 敏感性分析
        if self.sensitivity_results:
            report_lines.append("=== Sensitivity Analysis ===")
            report_lines.append("Variables ranked by impact (Spearman correlation):")
            for var_name, sens_data in self.sensitivity_results.items():
                corr = sens_data['correlation']
                report_lines.append(f"  {var_name:20s}: {corr:+.6f} (p-value: {sens_data['p_value']:.4e})")
            report_lines.append("")
        
        # 变量信息
        report_lines.append("=== Input Variables ===")
        for var_name, var in self.variables.items():
            report_lines.append(f"{var_name}:")
            report_lines.append(f"  Input Mode: {var.input_mode}")
            report_lines.append(f"  Distribution: {var.dist_type}")
            if var.dist_type == 'norm':
                report_lines.append(f"  Parameters: μ={var.dist_params[0]:.6f}, σ={var.dist_params[1]:.6f}")
            elif var.dist_type == 't':
                report_lines.append(f"  Parameters: df={var.dist_params[0]:.2f}, loc={var.dist_params[1]:.6f}, scale={var.dist_params[2]:.6f}")
            if var.min_value is not None or var.max_value is not None:
                report_lines.append(f"  Limits: [{var.min_value if var.min_value is not None else '-∞'}, {var.max_value if var.max_value is not None else '+∞'}]")
            report_lines.append("")
        
        # 公式
        if self.formula_str:
            report_lines.append("=== Formula ===")
            report_lines.append(self.formula_str)
            report_lines.append("")
        
        report_lines.append("=" * 70)
        report_lines.append("End of Report")
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def plot_pareto_chart(self, figsize=(12, 6), threshold=0.8):
        """
        绘制 Pareto 图，显示各变量对结果的贡献度
        """
        if self.sensitivity_results is None:
            raise ValueError("请先运行模拟以获得敏感性分析结果")
        
        # 提取数据
        var_names = list(self.sensitivity_results.keys())
        abs_corrs = [self.sensitivity_results[name]['abs_correlation'] 
                     for name in var_names]
        
        # 计算相对贡献度
        total = sum(abs_corrs)
        contributions = [(corr / total * 100) for corr in abs_corrs]
        cumulative = np.cumsum(contributions)
        
        # 创建图形
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # 柱状图
        x_pos = np.arange(len(var_names))
        bars = ax1.bar(x_pos, contributions, alpha=0.7, color='steelblue', 
                       edgecolor='navy', linewidth=1.5, label='Contribution (%)')
        
        # 标注数值
        for i, (bar, contrib) in enumerate(zip(bars, contributions)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{contrib:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Variable Name', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Contribution (%)', fontsize=13, fontweight='bold', color='steelblue')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(var_names, rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_ylim(0, max(contributions) * 1.15 if contributions else 1)
        
        # 折线图 - 累积贡献度
        ax2 = ax1.twinx()
        line = ax2.plot(x_pos, cumulative, color='red', marker='o', 
                       linewidth=2.5, markersize=8, label='Cumulative (%)')
        
        for i, (x, y) in enumerate(zip(x_pos, cumulative)):
            ax2.text(x, y + 2, f'{y:.1f}%',
                    ha='center', va='bottom', fontsize=9, 
                    color='darkred', fontweight='bold')
        
        ax2.set_ylabel('Cumulative Contribution (%)', fontsize=13, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 105)
        
        # 阈值线
        if threshold:
            ax2.axhline(y=threshold*100, color='green', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'{int(threshold*100)}% Threshold')
            
            n_vars_for_threshold = np.argmax(cumulative >= threshold*100) + 1
            ax2.text(len(var_names)-0.5, threshold*100 + 2, 
                    f'Top {n_vars_for_threshold} vars ≥{int(threshold*100)}%',
                    fontsize=11, color='green', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                             alpha=0.8, edgecolor='green'))
        
        plt.title(f'{self.result_name} - Sensitivity Analysis Pareto Chart', 
                 fontsize=15, fontweight='bold', pad=20)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                  loc='upper left', fontsize=11, framealpha=0.9)
        
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    
    def plot_tornado_chart(self, figsize=(10, 8)):
        """
        绘制龙卷风图（Tornado Chart），显示各变量对结果的影响方向
        """
        if self.sensitivity_results is None:
            raise ValueError("请先运行模拟以获得敏感性分析结果")
        
        # 提取数据并按绝对相关系数排序
        var_names = list(self.sensitivity_results.keys())
        correlations = [self.sensitivity_results[name]['correlation'] 
                       for name in var_names]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(var_names))
        
        # 绘制水平条形图
        colors = ['red' if corr < 0 else 'steelblue' for corr in correlations]
        bars = ax.barh(y_pos, correlations, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        
        # 标注数值
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            width = bar.get_width()
            ax.text(width + (0.02 if width > 0 else -0.02), bar.get_y() + bar.get_height()/2.,
                   f'{corr:.3f}',
                   ha='left' if width > 0 else 'right', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(var_names)
        ax.set_xlabel('Spearman Correlation', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.result_name} - Tornado Chart (Variable Impact)', 
                    fontsize=15, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=2)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', alpha=0.7, label='Positive'),
            Patch(facecolor='red', alpha=0.7, label='Negative')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=11)
        
        # 转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64

