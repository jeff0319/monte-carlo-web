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
import json
from datetime import datetime
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
    2. 统计参数
    
    支持的分布类型:
    - norm/normal: 正态分布
    - t: t分布
    - lognormal: 对数正态分布
    - uniform: 均匀分布
    - triangular: 三角分布
    - beta: Beta分布
    - gamma: Gamma分布
    - exponential: 指数分布
    - weibull: Weibull分布
    """
    
    def __init__(self, name: str, data: Optional[np.ndarray] = None, 
                 dist_type: str = 'norm', 
                 min_value: Optional[float] = None, 
                 max_value: Optional[float] = None,
                 **kwargs):
        """
        初始化变量
        
        Parameters:
        -----------
        name : str
            变量名称
        data : np.ndarray, optional
            原始测试数据 (方式1)
        dist_type : str
            分布类型
        min_value : float, optional
            采样的最小值限制
        max_value : float, optional
            采样的最大值限制
        **kwargs : 
            分布参数，根据不同分布类型而定:
            - norm/normal: mean, std
            - t: mean, std, df
            - lognormal: mean, std (对数空间)
            - uniform: min, max
            - triangular: min, mode, max
            - beta: alpha, beta, min, max
            - gamma: shape, scale
            - exponential: scale (或 rate)
            - weibull: shape, scale
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
        else:
            # 方式2: 统计参数
            self.input_mode = 'params'
            self.dist_type = dist_type
            self._parse_params(dist_type, kwargs)
    
    def _parse_params(self, dist_type: str, params: dict):
        """解析分布参数"""
        if dist_type in ['norm', 'normal']:
            mean = params.get('mean')
            std = params.get('std')
            if mean is None or std is None:
                raise ValueError("正态分布需要 mean 和 std 参数")
            self.dist_params = (mean, std)
            
        elif dist_type == 't':
            mean = params.get('mean')
            std = params.get('std')
            df = params.get('df')
            if mean is None or std is None or df is None:
                raise ValueError("t分布需要 mean, std 和 df 参数")
            self.dist_params = (df, mean, std)
            
        elif dist_type == 'lognormal':
            mean = params.get('mean')
            std = params.get('std')
            if mean is None or std is None:
                raise ValueError("对数正态分布需要 mean 和 std 参数")
            self.dist_params = (mean, std)
            
        elif dist_type == 'uniform':
            min_val = params.get('min')
            max_val = params.get('max')
            if min_val is None or max_val is None:
                raise ValueError("均匀分布需要 min 和 max 参数")
            self.dist_params = (min_val, max_val)
            
        elif dist_type == 'triangular':
            min_val = params.get('min')
            mode = params.get('mode')
            max_val = params.get('max')
            if min_val is None or mode is None or max_val is None:
                raise ValueError("三角分布需要 min, mode 和 max 参数")
            self.dist_params = (min_val, mode, max_val)
            
        elif dist_type == 'beta':
            alpha = params.get('alpha')
            beta = params.get('beta')
            min_val = params.get('min', 0)
            max_val = params.get('max', 1)
            if alpha is None or beta is None:
                raise ValueError("Beta分布需要 alpha 和 beta 参数")
            self.dist_params = (alpha, beta, min_val, max_val)
            
        elif dist_type == 'gamma':
            shape = params.get('shape')
            scale = params.get('scale')
            if shape is None or scale is None:
                raise ValueError("Gamma分布需要 shape 和 scale 参数")
            self.dist_params = (shape, scale)
            
        elif dist_type == 'exponential':
            scale = params.get('scale')
            rate = params.get('rate')
            if scale is None and rate is None:
                raise ValueError("指数分布需要 scale 或 rate 参数")
            if rate is not None:
                scale = 1.0 / rate
            self.dist_params = (scale,)
            
        elif dist_type == 'weibull':
            shape = params.get('shape')
            scale = params.get('scale')
            if shape is None or scale is None:
                raise ValueError("Weibull分布需要 shape 和 scale 参数")
            self.dist_params = (shape, scale)
            
        else:
            raise ValueError(f"不支持的分布类型: {dist_type}")
        
    def fit_distribution(self, dist_type: str = 'norm'):
        """
        拟合分布参数 (仅适用于原始数据输入方式)
        
        Parameters:
        -----------
        dist_type: 支持的分布类型
        """
        if self.input_mode != 'data':
            return
        
        n = len(self.raw_data)
        self.dist_type = dist_type
        
        if dist_type in ['norm', 'normal']:
            mu = np.mean(self.raw_data)
            sigma = np.std(self.raw_data, ddof=1)
            self.dist_params = (mu, sigma)
        
        elif dist_type == 't':
            df = n - 1
            loc = np.mean(self.raw_data)
            scale = np.std(self.raw_data, ddof=1)
            self.dist_params = (df, loc, scale)
        
        elif dist_type == 'lognormal':
            # 对数正态分布：拟合对数空间的参数
            log_data = np.log(self.raw_data[self.raw_data > 0])  # 只取正值
            if len(log_data) == 0:
                raise ValueError(f"变量 {self.name} 的数据必须为正值才能拟合对数正态分布")
            mu = np.mean(log_data)
            sigma = np.std(log_data, ddof=1)
            self.dist_params = (mu, sigma)
        
        elif dist_type == 'uniform':
            min_val = np.min(self.raw_data)
            max_val = np.max(self.raw_data)
            self.dist_params = (min_val, max_val)
        
        elif dist_type == 'triangular':
            min_val = np.min(self.raw_data)
            max_val = np.max(self.raw_data)
            mode = np.median(self.raw_data)  # 使用中位数作为众数估计
            self.dist_params = (min_val, mode, max_val)
        
        elif dist_type == 'beta':
            # Beta分布拟合
            min_val = np.min(self.raw_data)
            max_val = np.max(self.raw_data)
            # 标准化到[0,1]
            normalized = (self.raw_data - min_val) / (max_val - min_val)
            # 使用scipy拟合
            alpha, beta, loc, scale = stats.beta.fit(normalized, floc=0, fscale=1)
            self.dist_params = (alpha, beta, min_val, max_val)
        
        elif dist_type == 'gamma':
            # Gamma分布拟合（只能用于正值数据）
            if np.any(self.raw_data <= 0):
                raise ValueError(f"变量 {self.name} 的数据必须为正值才能拟合Gamma分布")
            shape, loc, scale = stats.gamma.fit(self.raw_data, floc=0)
            self.dist_params = (shape, scale)
        
        elif dist_type == 'exponential':
            # 指数分布拟合（只能用于正值数据）
            if np.any(self.raw_data <= 0):
                raise ValueError(f"变量 {self.name} 的数据必须为正值才能拟合指数分布")
            loc, scale = stats.expon.fit(self.raw_data, floc=0)
            self.dist_params = (scale,)
        
        elif dist_type == 'weibull':
            # Weibull分布拟合（只能用于正值数据）
            if np.any(self.raw_data <= 0):
                raise ValueError(f"变量 {self.name} 的数据必须为正值才能拟合Weibull分布")
            shape, loc, scale = stats.weibull_min.fit(self.raw_data, floc=0)
            self.dist_params = (shape, scale)
        
        else:
            raise ValueError(f"不支持的分布类型: {dist_type}")
    
    def monte_carlo_sample(self, n_samples: int = 1000000, random_seed: int = 4545):
        """
        进行Monte Carlo抽样，支持限值约束
        使用逆向CDF方法（Inverse Transform Sampling）统一处理所有分布
        
        Parameters:
        -----------
        n_samples : int
            采样数量
        random_seed : int
            随机数种子，默认4545，用于确保结果可重现
        """
        if self.dist_params is None:
            raise ValueError(f"请先对变量 {self.name} 进行分布拟合")

        # 设置随机数种子
        np.random.seed(random_seed)

        # 获取分布对象
        dist = self._get_distribution()

        # 计算CDF范围
        if self.min_value is not None or self.max_value is not None:
            lower = self.min_value if self.min_value is not None else -np.inf
            upper = self.max_value if self.max_value is not None else np.inf

            # 计算对应的CDF值
            cdf_lower = dist.cdf(lower)
            cdf_upper = dist.cdf(upper)

            # 在[cdf_lower, cdf_upper]范围内生成均匀分布
            u = np.random.uniform(cdf_lower, cdf_upper, size=n_samples)
        else:
            # 无限制时，在[0, 1]范围内生成均匀分布
            u = np.random.uniform(0, 1, size=n_samples)

        # 使用PPF（分位数函数）转换为目标分布
        self.samples = dist.ppf(u)

    def _get_distribution(self):
        """获取scipy分布对象"""
        if self.dist_type in ['norm', 'normal']:
            return stats.norm(loc=self.dist_params[0], scale=self.dist_params[1])
        elif self.dist_type == 't':
            return stats.t(df=self.dist_params[0], loc=self.dist_params[1], scale=self.dist_params[2])
        elif self.dist_type == 'lognormal':
            return stats.lognorm(s=self.dist_params[1], scale=np.exp(self.dist_params[0]))
        elif self.dist_type == 'uniform':
            return stats.uniform(loc=self.dist_params[0], scale=self.dist_params[1]-self.dist_params[0])
        elif self.dist_type == 'triangular':
            c = (self.dist_params[1] - self.dist_params[0]) / (self.dist_params[2] - self.dist_params[0])
            return stats.triang(c, loc=self.dist_params[0], scale=self.dist_params[2]-self.dist_params[0])
        elif self.dist_type == 'beta':
            alpha, beta, min_val, max_val = self.dist_params
            return stats.beta(alpha, beta, loc=min_val, scale=max_val-min_val)
        elif self.dist_type == 'gamma':
            return stats.gamma(a=self.dist_params[0], scale=self.dist_params[1])
        elif self.dist_type == 'exponential':
            return stats.expon(scale=self.dist_params[0])
        elif self.dist_type == 'weibull':
            return stats.weibull_min(c=self.dist_params[0], scale=self.dist_params[1])
        else:
            raise ValueError(f"不支持的分布类型: {self.dist_type}")
    
    def plot_distribution(self, figsize=(12, 5), show_samples=True):
        """
        绘制分布图,返回base64编码的图片
        """
        if self.dist_params is None:
            raise ValueError(f"请先对变量 {self.name} 进行分布拟合")
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 确定x轴范围 - 根据分布类型和是否有样本数据智能确定
        if self.samples is not None:
            # 如果已经有样本，使用样本的范围
            sample_min = np.min(self.samples)
            sample_max = np.max(self.samples)
            sample_std = np.std(self.samples)
            x_min = sample_min - 0.5 * sample_std
            x_max = sample_max + 0.5 * sample_std
        elif self.input_mode == 'data':
            # 使用原始数据的范围
            x_min = self.raw_data.min() - 0.5 * self.raw_data.std()
            x_max = self.raw_data.max() + 0.5 * self.raw_data.std()
        else:
            # 根据分布类型和参数确定范围
            if self.dist_type in ['norm', 'normal']:
                mean, std = self.dist_params[0], self.dist_params[1]
                x_min = mean - 4 * std
                x_max = mean + 4 * std
            elif self.dist_type == 't':
                df, loc, scale = self.dist_params
                x_min = loc - 4 * scale
                x_max = loc + 4 * scale
            elif self.dist_type == 'lognormal':
                # 对数正态分布：从对数空间参数计算实际空间的范围
                mu, sigma = self.dist_params[0], self.dist_params[1]
                # 实际空间的中位数 = exp(mu)
                median = np.exp(mu)
                # 使用分位数来确定范围
                x_min = max(0, stats.lognorm.ppf(0.001, s=sigma, scale=np.exp(mu)))
                x_max = stats.lognorm.ppf(0.999, s=sigma, scale=np.exp(mu))
            elif self.dist_type == 'uniform':
                min_val, max_val = self.dist_params[0], self.dist_params[1]
                range_width = max_val - min_val
                x_min = min_val - 0.1 * range_width
                x_max = max_val + 0.1 * range_width
            elif self.dist_type == 'triangular':
                min_val, mode, max_val = self.dist_params
                range_width = max_val - min_val
                x_min = min_val - 0.1 * range_width
                x_max = max_val + 0.1 * range_width
            elif self.dist_type == 'beta':
                alpha, beta, min_val, max_val = self.dist_params
                range_width = max_val - min_val
                x_min = min_val - 0.1 * range_width
                x_max = max_val + 0.1 * range_width
            elif self.dist_type == 'gamma':
                shape, scale = self.dist_params
                mean = shape * scale
                std = np.sqrt(shape) * scale
                x_min = max(0, mean - 3 * std)
                x_max = mean + 4 * std
            elif self.dist_type == 'exponential':
                scale = self.dist_params[0]
                x_min = 0
                x_max = scale * 5  # 约99.3%的数据
            elif self.dist_type == 'weibull':
                shape, scale = self.dist_params
                x_min = 0
                x_max = scale * 3  # 根据shape调整
            else:
                # 默认范围
                x_min, x_max = 0, 10
        
        x_range = np.linspace(x_min, x_max, 500)
        
        # 绘制拟合的分布曲线
        if self.dist_type in ['norm', 'normal']:
            pdf = stats.norm.pdf(x_range, self.dist_params[0], self.dist_params[1])
            ax.plot(x_range, pdf, 'r-', lw=2.5, 
                   label=f'Normal Distribution\nμ={self.dist_params[0]:.3f}, σ={self.dist_params[1]:.3f}', 
                   zorder=3)
        elif self.dist_type == 't':
            pdf = stats.t.pdf(x_range, self.dist_params[0], self.dist_params[1], self.dist_params[2])
            ax.plot(x_range, pdf, 'r-', lw=2.5, 
                   label=f't Distribution\ndf={self.dist_params[0]:.2f}, loc={self.dist_params[1]:.3f}, scale={self.dist_params[2]:.3f}', 
                   zorder=3)
        elif self.dist_type == 'lognormal':
            pdf = stats.lognorm.pdf(x_range, s=self.dist_params[1], scale=np.exp(self.dist_params[0]))
            ax.plot(x_range, pdf, 'r-', lw=2.5,
                   label=f'Lognormal Distribution\nμ={self.dist_params[0]:.3f}, σ={self.dist_params[1]:.3f}',
                   zorder=3)
        elif self.dist_type == 'uniform':
            pdf = stats.uniform.pdf(x_range, loc=self.dist_params[0], scale=self.dist_params[1]-self.dist_params[0])
            ax.plot(x_range, pdf, 'r-', lw=2.5,
                   label=f'Uniform Distribution\nmin={self.dist_params[0]:.3f}, max={self.dist_params[1]:.3f}',
                   zorder=3)
        elif self.dist_type == 'triangular':
            # 三角分布的PDF
            c = (self.dist_params[1] - self.dist_params[0]) / (self.dist_params[2] - self.dist_params[0])
            pdf = stats.triang.pdf(x_range, c, loc=self.dist_params[0], scale=self.dist_params[2]-self.dist_params[0])
            ax.plot(x_range, pdf, 'r-', lw=2.5,
                   label=f'Triangular Distribution\nmin={self.dist_params[0]:.3f}, mode={self.dist_params[1]:.3f}, max={self.dist_params[2]:.3f}',
                   zorder=3)
        elif self.dist_type == 'beta':
            alpha, beta, min_val, max_val = self.dist_params
            # 标准化x_range到[0,1]
            x_normalized = (x_range - min_val) / (max_val - min_val)
            x_normalized = np.clip(x_normalized, 0, 1)
            pdf = stats.beta.pdf(x_normalized, alpha, beta) / (max_val - min_val)
            ax.plot(x_range, pdf, 'r-', lw=2.5,
                   label=f'Beta Distribution\nα={alpha:.3f}, β={beta:.3f}\nrange=[{min_val:.3f}, {max_val:.3f}]',
                   zorder=3)
        elif self.dist_type == 'gamma':
            pdf = stats.gamma.pdf(x_range, a=self.dist_params[0], scale=self.dist_params[1])
            ax.plot(x_range, pdf, 'r-', lw=2.5,
                   label=f'Gamma Distribution\nshape={self.dist_params[0]:.3f}, scale={self.dist_params[1]:.3f}',
                   zorder=3)
        elif self.dist_type == 'exponential':
            pdf = stats.expon.pdf(x_range, scale=self.dist_params[0])
            ax.plot(x_range, pdf, 'r-', lw=2.5,
                   label=f'Exponential Distribution\nscale={self.dist_params[0]:.3f}',
                   zorder=3)
        elif self.dist_type == 'weibull':
            pdf = stats.weibull_min.pdf(x_range, c=self.dist_params[0], scale=self.dist_params[1])
            ax.plot(x_range, pdf, 'r-', lw=2.5,
                   label=f'Weibull Distribution\nshape={self.dist_params[0]:.3f}, scale={self.dist_params[1]:.3f}',
                   zorder=3)
        else:
            # 默认情况
            pdf = np.ones_like(x_range) * 0.01
            ax.plot(x_range, pdf, 'r-', lw=2.5, label=f'{self.dist_type} Distribution', zorder=3)
        
        # 计算 PDF 的最大值，用于统一 y 轴范围
        max_pdf = np.max(pdf)

        # 如果有Monte Carlo样本且需要显示
        has_samples = self.samples is not None and show_samples
        hist_max = max_pdf  # 初始化为 PDF 最大值

        if has_samples:
            counts, bins, patches = ax.hist(self.samples, bins=100, density=True, alpha=0.3,
                   label='Monte Carlo Samples', color='green', edgecolor='darkgreen',
                   linewidth=0.5, zorder=1)
            hist_max = max(hist_max, np.max(counts))

        # 显示原始数据
        if self.input_mode == 'data':
            if len(self.raw_data) <= 30:
                # 小样本：使用散点图
                y_base = 0
                y_jitter = np.random.normal(0, 0.005 * max_pdf, len(self.raw_data))
                y_positions = y_base + y_jitter
                ax.scatter(self.raw_data, y_positions, alpha=0.5, s=60,
                           label='Raw Data', color='steelblue', zorder=5)
                ax.set_ylim(y_base - 0.05 * max_pdf, max_pdf * 1.1)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
            else:
                # 大样本：使用直方图
                counts, bins, patches = ax.hist(self.raw_data, bins=min(30, len(self.raw_data)//3),
                       density=True, alpha=0.5, label='Raw Data',
                       color='steelblue', edgecolor='darkblue', zorder=2)
                hist_max = max(hist_max, np.max(counts))

        # 统一设置 y 轴范围（仅当有直方图时）
        if (has_samples or (self.input_mode == 'data' and len(self.raw_data) > 30)):
            ax.set_ylim(0, hist_max * 1.1)
        
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
    Monte Carlo 模拟器主类
    """
    
    def __init__(self):
        self.variables = {}
        self.result = None
        self.result_name = 'Result'
        self.formula_str = None
        self.confidence_levels = [0.90, 0.95, 0.99]  # 默认置信水平
        self.sensitivity_results = None
        self.cdf_fit_result = None
        self.cdf_inverse_fit_result = None  # 新增：反向CDF拟合
        self.cdf_fit_degree = 5
        self.result_samples = None  # 用于存储采样的结果数据
        self.var_samples = None  # 用于存储采样的变量数据
        self.chart_cache = {}  # 图表缓存
    
    def add_variable(self, name: str, data: Optional[np.ndarray] = None,
                    dist_type: str = 'norm', 
                    min_value: Optional[float] = None, 
                    max_value: Optional[float] = None,
                    **kwargs):
        """
        添加一个变量
        
        Parameters:
        -----------
        name : str
            变量名称
        data : np.ndarray, optional
            原始数据（如果提供，则为 data 模式）
        dist_type : str
            分布类型
        min_value : float, optional
            采样最小值限制
        max_value : float, optional
            采样最大值限制
        **kwargs : 
            其他分布参数（根据分布类型而定）
        """
        # 标准化 dist_type
        if dist_type == 'normal':
            dist_type = 'norm'
        
        # 如果提供了 data，使用 data 模式
        if data is not None:
            var = Variable(name, data=data, dist_type=dist_type, 
                          min_value=min_value, max_value=max_value)
            # 进行分布拟合
            var.fit_distribution(dist_type)
        else:
            # 使用 params 模式，传递所有参数
            var = Variable(name, dist_type=dist_type, 
                          min_value=min_value, max_value=max_value, **kwargs)
        
        self.variables[name] = var
    
    def add_variables_from_json(self, json_data):
        """
        从JSON数据批量添加变量
        
        支持两种格式:
        1. 数组格式: [{name: "var_A", input_mode: "params", ...}, ...]
        2. 对象格式: {"var_A": {type: "params", ...}, "var_B": {...}}
        
        Parameters:
        -----------
        json_data : List[Dict] or Dict
            包含变量定义的JSON数组或对象
        """
        # 如果是对象格式，转换为数组格式
        if isinstance(json_data, dict):
            # 对象格式: {"var_A": {...}, "var_B": {...}}
            var_list = []
            for var_name, var_config in json_data.items():
                if isinstance(var_config, dict):
                    var_config['name'] = var_name
                    # 兼容旧格式: type -> input_mode
                    if 'type' in var_config and 'input_mode' not in var_config:
                        var_config['input_mode'] = var_config['type']
                    # 兼容旧格式: values -> data
                    if 'values' in var_config and 'data' not in var_config:
                        var_config['data'] = var_config['values']
                    # 兼容旧格式: distribution -> dist_type
                    if 'distribution' in var_config and 'dist_type' not in var_config:
                        var_config['dist_type'] = var_config['distribution']
                    # 兼容旧格式: min_limit -> min_value, max_limit -> max_value
                    if 'min_limit' in var_config and 'min_value' not in var_config:
                        var_config['min_value'] = var_config['min_limit']
                    if 'max_limit' in var_config and 'max_value' not in var_config:
                        var_config['max_value'] = var_config['max_limit']
                    var_list.append(var_config)
            json_data = var_list
        
        # 处理数组格式
        for var_def in json_data:
            name = var_def.get('name')
            if not name:
                raise ValueError("每个变量必须有name字段")
            
            input_mode = var_def.get('input_mode')
            dist_type = var_def.get('dist_type', 'norm')
            min_value = var_def.get('min_value')
            max_value = var_def.get('max_value')
            
            # 标准化 dist_type
            if dist_type == 'normal':
                dist_type = 'norm'
            
            if input_mode == 'data':
                data = var_def.get('data')
                if not data:
                    raise ValueError(f"变量 {name} 使用 data 模式但未提供 data 字段")
                self.add_variable(name, data=np.array(data), dist_type=dist_type,
                                min_value=min_value, max_value=max_value)
            
            elif input_mode == 'params':
                # 提取所有可能的参数
                params = {}
                
                # 通用参数
                for key in ['mean', 'std', 'df', 'min', 'max', 'mode', 'alpha', 'beta', 
                           'shape', 'scale', 'rate']:
                    if key in var_def:
                        params[key] = var_def[key]
                
                # 使用 add_variable 方法，它会处理参数验证
                self.add_variable(name, dist_type=dist_type, 
                                min_value=min_value, max_value=max_value, **params)
            else:
                raise ValueError(f"变量 {name} 的 input_mode 必须是 'data' 或 'params'")
    
    def run_simulation(self, formula: Callable, result_name: str = 'Result', 
                      n_samples: int = 1000000, formula_str: str = None,
                      cdf_fit_degree: int = 5):
        """
        运行Monte Carlo模拟
        """
        if len(self.variables) == 0:
            raise ValueError("请先添加变量")
        
        self.result_name = result_name
        self.formula_str = formula_str
        self.cdf_fit_degree = cdf_fit_degree
        
        # 对所有变量进行Monte Carlo抽样
        var_samples = {}
        for name, var in self.variables.items():
            var.monte_carlo_sample(n_samples)
            var_samples[name] = var.samples
        
        # 保存变量样本
        self.var_samples = var_samples
        
        # 计算结果
        self.result = formula(var_samples)
        self.result_samples = self.result
        
        # 敏感性分析
        self._perform_sensitivity_analysis(var_samples)
        
        # CDF拟合 (Forward: Result → CDF)
        self._fit_cdf_polynomial()
        
        # CDF拟合 (Inverse: CDF → Result) 【新增】
        self._fit_cdf_inverse_polynomial()
        
        # 清空图表缓存
        self.chart_cache = {}
    
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

        # 使用子集计算以提升性能（对于大样本，10000个样本足够准确）
        sample_size = min(50000, len(self.result))
        if sample_size < len(self.result):
            indices = np.random.choice(len(self.result), size=sample_size, replace=False)
            result_subset = self.result[indices]
            var_samples_subset = {k: v[indices] for k, v in var_samples.items()}
        else:
            result_subset = self.result
            var_samples_subset = var_samples

        sensitivities = {}
        for var_name, var_data in var_samples_subset.items():
            corr, p_value = stats.spearmanr(var_data, result_subset)
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
        对CDF曲线进行多项式拟合（Forward: Result → CDF）
        
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
        
        # 生成等分点的索引
        indices = np.linspace(0, n_total - 1, n_points, dtype=int)
        x_points = sorted_result[indices]
        
        # 对应的CDF值
        cdf_values = (indices + 1) / n_total
        
        # 多项式拟合: CDF = f(result)
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
            'rmse': float(rmse),
            'x_points': x_points.tolist(),
            'cdf_values': cdf_values.tolist()
        }
    
    def _fit_cdf_inverse_polynomial(self, n_points: int = 500):
        """
        对CDF曲线进行反向多项式拟合（Inverse: CDF → Result）
        用于根据累积概率反算结果值（分位数函数）
        
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
        
        # 生成等分点的索引
        indices = np.linspace(0, n_total - 1, n_points, dtype=int)
        x_points = sorted_result[indices]
        
        # 对应的CDF值
        cdf_values = (indices + 1) / n_total
        
        # 反向多项式拟合: result = f(CDF)
        coefficients = np.polyfit(cdf_values, x_points, self.cdf_fit_degree)
        
        # 计算拟合优度
        fitted_result = np.polyval(coefficients, cdf_values)
        
        # R² score
        ss_res = np.sum((x_points - fitted_result) ** 2)
        ss_tot = np.sum((x_points - np.mean(x_points)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # RMSE
        rmse = np.sqrt(np.mean((x_points - fitted_result) ** 2))
        
        # 保存结果
        self.cdf_inverse_fit_result = {
            'degree': self.cdf_fit_degree,
            'n_points': n_points,
            'coefficients': coefficients.tolist(),
            'r_squared': float(r_squared),
            'rmse': float(rmse),
            'cdf_values': cdf_values.tolist(),
            'x_points': x_points.tolist()
        }
    
    def analyze_result(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        分析结果统计数据
        """
        if self.result is None:
            raise ValueError("请先运行模拟")
        
        analysis = {
            'mean': float(np.mean(self.result)),
            'std': float(np.std(self.result, ddof=1)),
            'median': float(np.median(self.result)),
            'min': float(np.min(self.result)),
            'max': float(np.max(self.result)),
            'percentiles': {}
        }
        
        # 常用百分位数
        common_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in common_percentiles:
            analysis['percentiles'][f'p{p}'] = float(np.percentile(self.result, p))
        
        # 置信区间
        analysis['confidence_intervals'] = {}
        for cl in confidence_levels:
            lower = (1 - cl) / 2
            upper = 1 - lower
            analysis['confidence_intervals'][f'{int(cl*100)}%'] = {
                'lower': float(np.percentile(self.result, lower * 100)),
                'upper': float(np.percentile(self.result, upper * 100))
            }
        
        # 敏感性分析
        if self.sensitivity_results:
            analysis['sensitivity'] = self.sensitivity_results
        
        # CDF拟合信息
        if self.cdf_fit_result:
            analysis['cdf_forward_fit'] = {
                'degree': self.cdf_fit_result['degree'],
                'r_squared': self.cdf_fit_result['r_squared'],
                'rmse': self.cdf_fit_result['rmse']
            }
        
        # 反向CDF拟合信息
        if self.cdf_inverse_fit_result:
            analysis['cdf_inverse_fit'] = {
                'degree': self.cdf_inverse_fit_result['degree'],
                'r_squared': self.cdf_inverse_fit_result['r_squared'],
                'rmse': self.cdf_inverse_fit_result['rmse']
            }
        
        return analysis
    
    def plot_result(self, figsize=(15, 10), trim_percentile=0.05):
        """
        绘制结果的多种统计图,返回base64编码的图片
        
        Parameters:
        -----------
        figsize : tuple
            图形大小
        trim_percentile : float
            裁剪极端尾部的百分位数阈值（默认0.1，即裁剪两端各0.1%的极端值）
            设置为0则不裁剪，显示全部数据范围
        """
        if self.result is None:
            raise ValueError("请先运行模拟")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        mean_val = np.mean(self.result)
        median_val = np.median(self.result)
        std_val = np.std(self.result)
        
        # 智能确定显示范围：裁剪极端尾部以聚焦主要分布
        if trim_percentile > 0:
            lower_bound = np.percentile(self.result, trim_percentile)
            upper_bound = np.percentile(self.result, 100 - trim_percentile)
            
            data_range = upper_bound - lower_bound
            margin = data_range * 0.05
            x_min_display = lower_bound - margin
            x_max_display = upper_bound + margin

            data_in_range = self.result[(self.result >= x_min_display) & (self.result <= x_max_display)]
        else:
            x_min_display = self.result.min()
            x_max_display = self.result.max()
            data_in_range = self.result

        # 简化的智能bins策略（3档分级）
        n_data_in_range = len(data_in_range)
        if n_data_in_range < 1000:
            n_bins = 50
        elif n_data_in_range < 50000:
            n_bins = 100
        else:
            n_bins = 150
        
        # 1. 直方图和KDE
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(self.result, bins=n_bins, density=True, alpha=0.6,
                color='skyblue', edgecolor='black', label='Histogram')

        from scipy.stats import gaussian_kde
        kde = gaussian_kde(self.result)
        x_range = np.linspace(x_min_display, x_max_display, 200)
        ax1.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
        
        ax1.set_xlim(x_min_display, x_max_display)
        
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
            ax3.text(val, q + offset_y, f'{q*100:.0f}%: {val:.4f}', 
                    fontsize=10, ha='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
        
        ax3.set_xlabel(f'{self.result_name} Value', fontsize=13)
        ax3.set_ylabel('Cumulative Probability', fontsize=13)
        ax3.set_title(f'{self.result_name} - CDF', fontsize=15, fontweight='bold')
        ax3.legend(fontsize=11, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_xlim(x_min_display, x_max_display)
        
        # 转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    
    def generate_report(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        生成文本格式的分析报告
        """
        if self.result is None:
            raise ValueError("请先运行模拟")
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("         MONTE CARLO SIMULATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        report_lines.append(f"Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Result Name: {self.result_name}")
        report_lines.append(f"Number of Samples: {len(self.result):,}")
        report_lines.append("")
        
        # 统计摘要
        report_lines.append("=== Statistical Summary ===")
        report_lines.append(f"Mean:              {np.mean(self.result):.6f}")
        report_lines.append(f"Std Deviation:     {np.std(self.result, ddof=1):.6f}")
        report_lines.append(f"Median (P50):      {np.median(self.result):.6f}")
        report_lines.append(f"Minimum:           {np.min(self.result):.6f}")
        report_lines.append(f"Maximum:           {np.max(self.result):.6f}")
        report_lines.append("")
        
        # 百分位数
        report_lines.append("=== Percentiles ===")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(self.result, p)
            report_lines.append(f"P{p:2d}:  {val:.6f}")
        report_lines.append("")
        
        # 置信区间
        report_lines.append("=== Confidence Intervals ===")
        for cl in confidence_levels:
            lower = (1 - cl) / 2
            upper = 1 - lower
            lower_val = np.percentile(self.result, lower * 100)
            upper_val = np.percentile(self.result, upper * 100)
            report_lines.append(f"{int(cl*100)}% CI: [{lower_val:.6f}, {upper_val:.6f}]")
        report_lines.append("")
        
        # CDF Forward Fit
        if self.cdf_fit_result:
            report_lines.append("=== CDF Forward Fit (Result → CDF) ===")
            report_lines.append(f"Degree: {self.cdf_fit_result['degree']}")
            report_lines.append(f"Number of fitting points: {self.cdf_fit_result['n_points']}")
            report_lines.append(f"R² Score: {self.cdf_fit_result['r_squared']:.8f}")
            report_lines.append(f"RMSE: {self.cdf_fit_result['rmse']:.8f}")
            report_lines.append("")
            
            report_lines.append("Polynomial Coefficients (highest to lowest degree):")
            coeffs = self.cdf_fit_result['coefficients']
            for i, coeff in enumerate(coeffs):
                power = self.cdf_fit_result['degree'] - i
                report_lines.append(f"  a{power}: {coeff:.15e}")
            report_lines.append("")
            
            # 构建函数字符串
            func_parts = []
            for i, coeff in enumerate(coeffs):
                power = self.cdf_fit_result['degree'] - i
                if power == 0:
                    func_parts.append(f"{coeff:.6e}")
                elif power == 1:
                    func_parts.append(f"{coeff:.6e}*x")
                else:
                    func_parts.append(f"{coeff:.6e}*x^{power}")
            
            func_str = " + ".join(func_parts).replace("+ -", "- ")
            report_lines.append("Fitted Function:")
            report_lines.append(f"  CDF(x) = {func_str}")
            report_lines.append("")
        
        # CDF Inverse Fit
        if self.cdf_inverse_fit_result:
            report_lines.append("=== CDF Inverse Fit (CDF → Result) ===")
            report_lines.append(f"Degree: {self.cdf_inverse_fit_result['degree']}")
            report_lines.append(f"Number of fitting points: {self.cdf_inverse_fit_result['n_points']}")
            report_lines.append(f"R² Score: {self.cdf_inverse_fit_result['r_squared']:.8f}")
            report_lines.append(f"RMSE: {self.cdf_inverse_fit_result['rmse']:.8f}")
            report_lines.append("")
            
            report_lines.append("Polynomial Coefficients (highest to lowest degree):")
            coeffs = self.cdf_inverse_fit_result['coefficients']
            for i, coeff in enumerate(coeffs):
                power = self.cdf_inverse_fit_result['degree'] - i
                report_lines.append(f"  b{power}: {coeff:.15e}")
            report_lines.append("")
            
            # 构建函数字符串
            func_parts = []
            for i, coeff in enumerate(coeffs):
                power = self.cdf_inverse_fit_result['degree'] - i
                if power == 0:
                    func_parts.append(f"{coeff:.6e}")
                elif power == 1:
                    func_parts.append(f"{coeff:.6e}*p")
                else:
                    func_parts.append(f"{coeff:.6e}*p^{power}")
            
            func_str = " + ".join(func_parts).replace("+ -", "- ")
            report_lines.append("Fitted Quantile Function:")
            report_lines.append(f"  x(p) = {func_str}")
            report_lines.append("")
        
        # 敏感性分析
        if self.sensitivity_results:
            report_lines.append("=== Sensitivity Analysis ===")
            report_lines.append(f"{'Rank':<6} {'Variable':<20} {'Correlation':<14} {'P-Value':<14} {'Significance'}")
            report_lines.append("-" * 70)
            
            for rank, (var_name, sens_data) in enumerate(self.sensitivity_results.items(), 1):
                corr = sens_data['correlation']
                p_val = sens_data['p_value']
                
                # 添加显著性标记
                if p_val < 0.001:
                    significance = "***"
                elif p_val < 0.01:
                    significance = "**"
                elif p_val < 0.05:
                    significance = "*"
                else:
                    significance = ""
                
                report_lines.append(f"{rank:<6} {var_name:<20} {corr:>+.6f}{'':<6} {p_val:.4e}{'':<6} {significance}")
            
            report_lines.append("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
            report_lines.append("")
            
            # Pareto 分析
            report_lines.append("=== Pareto Analysis (Contribution & Cumulative) ===")
            var_names = list(self.sensitivity_results.keys())
            abs_corrs = [self.sensitivity_results[name]['abs_correlation'] for name in var_names]
            total = sum(abs_corrs)
            
            if total > 0:
                contributions = [(corr / total * 100) for corr in abs_corrs]
                cumulative = 0
                key_var_count = 0
                threshold_reached = False
                
                report_lines.append(f"{'Rank':<6} {'Variable':<20} {'Contribution':<14} {'Cumulative':<14} {'Status'}")
                report_lines.append("-" * 70)
                
                for rank, (var_name, contrib) in enumerate(zip(var_names, contributions), 1):
                    cumulative += contrib
                    # 标记所有对达到80%有贡献的变量
                    if cumulative >= 80 and not threshold_reached:
                        threshold_reached = True
                        key_var_count = rank
                        status = "*** KEY ***"
                    elif not threshold_reached:
                        # 80%之前的所有变量都是关键的
                        status = "*** KEY ***"
                    else:
                        status = ""
                    
                    report_lines.append(f"{rank:<6} {var_name:<20} {contrib:>6.2f}%{'':<7} {cumulative:>6.2f}%{'':<7} {status}")
                
                if key_var_count > 0:
                    report_lines.append(f"Key Variables (80% rule): Top {key_var_count} variable(s) contribute ≥80% of impact")
                else:
                    report_lines.append(f"Note: All {len(var_names)} variables combined contribute < 80% (possible edge case)")
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
    
    def export_to_csv(self):
        """
        导出数据为CSV格式（包含metadata和采样数据）
        """
        if self.result is None:
            raise ValueError("请先运行模拟")
        
        lines = []
        
        # Metadata部分 - 更清晰的标题
        lines.append("# MONTE CARLO SIMULATION RESULTS")
        lines.append(f"# Simulation_Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# Formula: {self.formula_str if self.formula_str else 'N/A'}")
        lines.append(f"# Result_Name: {self.result_name}")
        lines.append(f"# Number_of_Samples: {len(self.result)}")
        lines.append("#")
        
        # 变量配置 - 表格化格式
        lines.append("# === VARIABLE CONFIGURATIONS ===")
        lines.append("# Name, Input_Mode, Distribution, Parameters, Limits")
        for var_name, var in self.variables.items():
            # 分布参数
            if var.dist_type == 'norm':
                params = f"mean={var.dist_params[0]:.6f}, std={var.dist_params[1]:.6f}"
            elif var.dist_type == 't':
                params = f"df={var.dist_params[0]:.2f}, loc={var.dist_params[1]:.6f}, scale={var.dist_params[2]:.6f}"
            else:
                params = "N/A"
            
            # 限制条件
            limits = ""
            if var.min_value is not None or var.max_value is not None:
                min_str = f"{var.min_value:.6f}" if var.min_value is not None else "-inf"
                max_str = f"{var.max_value:.6f}" if var.max_value is not None else "+inf"
                limits = f"[{min_str}, {max_str}]"
            else:
                limits = "[-inf, +inf]"
            
            lines.append(f"# {var_name}, {var.input_mode}, {var.dist_type}, {params}, {limits}")
        lines.append("#")
        
        # 统计摘要 - 表格格式
        lines.append("# === STATISTICAL SUMMARY ===")
        lines.append("# Statistic, Value")
        lines.append(f"# Mean, {np.mean(self.result):.8f}")
        lines.append(f"# Std, {np.std(self.result, ddof=1):.8f}")
        lines.append(f"# Median, {np.median(self.result):.8f}")
        lines.append(f"# Min, {np.min(self.result):.8f}")
        lines.append(f"# Max, {np.max(self.result):.8f}")
        
        # 分位数 - 统一格式
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(self.result, p)
            lines.append(f"# P{p:02d}, {val:.8f}")
        lines.append("#")
        
        # CDF Forward Fit
        if self.cdf_fit_result:
            lines.append("# === CDF FORWARD FIT (Result → CDF) ===")
            lines.append("# Parameter, Value")
            lines.append(f"# Degree, {self.cdf_fit_result['degree']}")
            lines.append(f"# R_squared, {self.cdf_fit_result['r_squared']:.10f}")
            lines.append(f"# RMSE, {self.cdf_fit_result['rmse']:.10f}")
            lines.append("# Coefficients:")
            for i, coeff in enumerate(self.cdf_fit_result['coefficients']):
                lines.append(f"#   coeff_{i}, {coeff:.15e}")
            lines.append("#")
        
        # CDF Inverse Fit
        if self.cdf_inverse_fit_result:
            lines.append("# === CDF INVERSE FIT (CDF → Result) ===")
            lines.append("# Parameter, Value")
            lines.append(f"# Degree, {self.cdf_inverse_fit_result['degree']}")
            lines.append(f"# R_squared, {self.cdf_inverse_fit_result['r_squared']:.10f}")
            lines.append(f"# RMSE, {self.cdf_inverse_fit_result['rmse']:.10f}")
            lines.append("# Coefficients:")
            for i, coeff in enumerate(self.cdf_inverse_fit_result['coefficients']):
                lines.append(f"#   coeff_{i}, {coeff:.15e}")
            lines.append("#")
        
        # 敏感性分析 - 表格格式
        if self.sensitivity_results:
            lines.append("# === SENSITIVITY ANALYSIS ===")
            lines.append("# Rank, Variable, Correlation, P_Value, Significance")
            
            for rank, (var_name, data) in enumerate(self.sensitivity_results.items(), 1):
                # 添加显著性标记
                p_val = data['p_value']
                if p_val < 0.001:
                    significance = "***"
                elif p_val < 0.01:
                    significance = "**"
                elif p_val < 0.05:
                    significance = "*"
                else:
                    significance = ""
                
                lines.append(f"# {rank}, {var_name}, {data['correlation']:+.6f}, {p_val:.4e}, {significance}")
            
            lines.append("# Significance: *** p<0.001, ** p<0.01, * p<0.05")
            lines.append("#")

            # ** Pareto 分析部分 **
            lines.append("# === PARETO ANALYSIS ===")
            lines.append("# Rank, Variable, Contribution_%, Cumulative_%, Status")
            
            var_names = list(self.sensitivity_results.keys())
            abs_corrs = [self.sensitivity_results[name]['abs_correlation'] for name in var_names]
            total = sum(abs_corrs)
            
            if total > 0:
                contributions = [(corr / total * 100) for corr in abs_corrs]
                cumulative = 0
                threshold_reached = False
                
                for rank, (var_name, contrib) in enumerate(zip(var_names, contributions), 1):
                    cumulative += contrib
                    if cumulative >= 80 and not threshold_reached:
                        threshold_reached = True
                        status = "KEY"
                    elif not threshold_reached:
                        status = "KEY"
                    else:
                        status = ""
                    
                    lines.append(f"# {rank}, {var_name}, {contrib:.2f}, {cumulative:.2f}, {status}")
            
            lines.append("#")
        
        # 原始数据部分
        lines.append("# === RAW SAMPLE DATA ===")
        
        # 数据头
        header = [f"{self.result_name}"] + list(self.variables.keys())
        lines.append(','.join(header))
        
        # # 数据行 - 可选：限制输出行数以避免文件过大
        # max_samples_to_export = min(10000, len(self.result))  # 最多导出10000行
        # if len(self.result) > max_samples_to_export:
        #     lines.append(f"# Note: Showing first {max_samples_to_export} samples out of {len(self.result)} total")
        #     indices = range(max_samples_to_export)
        # else:
        #     indices = range(len(self.result))
        
        indices = range(len(self.result))
        for i in indices:
            row = [f"{self.result[i]:.15e}"]
            for var_name in self.variables.keys():
                row.append(f"{self.var_samples[var_name][i]:.15e}")
            lines.append(','.join(row))
        
        # # 如果截断了数据，添加说明
        # if len(self.result) > max_samples_to_export:
        #     lines.append(f"# ... {len(self.result) - max_samples_to_export} more samples not shown")
        
        return '\n'.join(lines)
    
    def export_to_json(self, include_samples=True):
        """
        导出数据为JSON格式
        """
        if self.result is None:
            raise ValueError("请先运行模拟")
        
        data = {
            'metadata': {
                'simulation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'formula': self.formula_str if self.formula_str else None,
                'result_name': self.result_name,
                'n_samples': len(self.result)
            },
            'variables': {},
            f'{self.result_name}_statistics': {
                'mean': float(np.mean(self.result)),
                'std': float(np.std(self.result, ddof=1)),
                'median': float(np.median(self.result)),
                'min': float(np.min(self.result)),
                'max': float(np.max(self.result)),
                'percentiles': {}
            },
            'cdf_fit': {},
            'sensitivity': {}
        }
        
        # 百分位数
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            data[f'{self.result_name}_statistics']['percentiles'][f'p{p}'] = float(np.percentile(self.result, p))
        
        # 变量信息
        for var_name, var in self.variables.items():
            data['variables'][var_name] = {
                'input_mode': var.input_mode,
                'dist_type': var.dist_type,
                'dist_params': list(var.dist_params) if var.dist_params else None,
                'limits': {
                    'min': var.min_value,
                    'max': var.max_value
                }
            }
        
        # CDF拟合
        if self.cdf_fit_result:
            data['cdf_fit']['forward'] = {
                'type': 'polynomial',
                'degree': self.cdf_fit_result['degree'],
                'coefficients': self.cdf_fit_result['coefficients'],
                'r_squared': self.cdf_fit_result['r_squared'],
                'rmse': self.cdf_fit_result['rmse']
            }
        
        if self.cdf_inverse_fit_result:
            data['cdf_fit']['inverse'] = {
                'type': 'polynomial',
                'degree': self.cdf_inverse_fit_result['degree'],
                'coefficients': self.cdf_inverse_fit_result['coefficients'],
                'r_squared': self.cdf_inverse_fit_result['r_squared'],
                'rmse': self.cdf_inverse_fit_result['rmse']
            }
        
        # 敏感性分析
        if self.sensitivity_results:
            for rank, (var_name, sens_data) in enumerate(self.sensitivity_results.items(), 1):
                data['sensitivity'][var_name] = {
                    'rank': rank,
                    'correlation': sens_data['correlation'],
                    'abs_correlation': sens_data['abs_correlation'],
                    'p_value': sens_data['p_value']
                }
            # ** Pareto **
            var_names = list(self.sensitivity_results.keys())
            abs_corrs = [self.sensitivity_results[name]['abs_correlation'] for name in var_names]
            total = sum(abs_corrs)
            
            data['pareto_analysis'] = {}
            
            if total > 0:
                contributions = [(corr / total * 100) for corr in abs_corrs]
                cumulative = 0
                threshold_reached = False
                key_variables = []
                
                for rank, (var_name, contrib) in enumerate(zip(var_names, contributions), 1):
                    cumulative += contrib
                    is_key = False
                    
                    if cumulative >= 80 and not threshold_reached:
                        threshold_reached = True
                        is_key = True
                    elif not threshold_reached:
                        is_key = True
                    
                    if is_key:
                        key_variables.append(var_name)
                    
                    data['pareto_analysis'][var_name] = {
                        'rank': rank,
                        'contribution_percent': round(contrib, 2),
                        'cumulative_percent': round(cumulative, 2),
                        'is_key_variable': is_key
                    }
                
                data['pareto_analysis']['summary'] = {
                    'key_variable_count': len(key_variables),
                    'key_variables': key_variables,
                    'threshold': 80.0
                }
        
        # 采样数据（可选）
        if include_samples:
            data['samples'] = {
                f'{self.result_name}': self.result.tolist()
            }
            for var_name in self.variables.keys():
                data['samples'][var_name] = self.var_samples[var_name].tolist()
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
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
        ax1.set_xticklabels(var_names, rotation=15, ha='right')
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
        
        plt.title(f'{self.result_name} - Sensitivity Analysis Pareto Chart', 
                 fontsize=15, fontweight='bold', pad=20)
        
        # 合并图例 - 智能放置，避免与左侧长条形图重叠
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        # 判断第一个变量的贡献度，如果很高（>40%），图例放右上角；否则放左上角
        if contributions and contributions[0] > 40:
            legend_loc = 'upper right'
        else:
            legend_loc = 'upper left'
        
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                  loc=legend_loc, fontsize=11, framealpha=0.9)
        
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    
    def plot_tornado_chart(self, figsize=(12, 6)):
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
