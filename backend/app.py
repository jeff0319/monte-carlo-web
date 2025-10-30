from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from monte_carlo import MonteCarloSimulator
import json
import traceback
import numpy as np
import secrets
import uuid
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# 设置 session 密钥（用于用户隔离）
app.secret_key = secrets.token_hex(32)

# 存储每个用户的模拟器实例
user_simulators = {}
user_last_access = {}

def get_user_simulator():
    """获取当前用户的模拟器实例"""
    # 确保用户有 session ID
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    user_id = session['user_id']
    
    # 记录最后访问时间
    user_last_access[user_id] = datetime.now()
    
    # 如果用户的模拟器不存在，创建一个新的
    if user_id not in user_simulators:
        user_simulators[user_id] = MonteCarloSimulator()
    
    # 清理超过1小时未使用的用户数据
    cleanup_inactive_users()
    
    return user_simulators[user_id]

def cleanup_inactive_users():
    """清理超过1小时未活动的用户"""
    now = datetime.now()
    inactive_users = [
        user_id for user_id, last_access in user_last_access.items()
        if now - last_access > timedelta(hours=1)
    ]
    
    for user_id in inactive_users:
        if user_id in user_simulators:
            del user_simulators[user_id]
        del user_last_access[user_id]

@app.route('/')
def index():
    """返回前端页面"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/add_variable', methods=['POST'])
def add_variable():
    """添加变量（已废弃，保留用于向后兼容）"""
    try:
        simulator = get_user_simulator()
        
        data = request.json
        name = data.get('name')
        input_mode = data.get('input_mode')  # 'data' 或 'params'
        min_value = data.get('min_value')
        max_value = data.get('max_value')
        
        if input_mode == 'data':
            # 原始数据输入
            raw_data = data.get('data', [])
            dist_type = data.get('dist_type', 'normal')
            # 确保使用 'normal' 或 't'
            if dist_type not in ['normal', 't', 'norm']:
                dist_type = 'normal'
            simulator.add_variable(name, data=raw_data, dist_type=dist_type, 
                                 min_value=min_value, max_value=max_value)
        elif input_mode == 'params':
            # 统计参数输入
            mean = data.get('mean')
            std = data.get('std')
            dist_type = data.get('dist_type', 'normal')
            # 确保使用 'normal' 或 't'
            if dist_type not in ['normal', 't', 'norm']:
                dist_type = 'normal'
            df = data.get('df', None)
            simulator.add_variable(name, mean=mean, std=std, dist_type=dist_type, df=df,
                                 min_value=min_value, max_value=max_value)
        else:
            return jsonify({'error': 'Invalid input_mode'}), 400
        
        return jsonify({
            'success': True,
            'message': f'Variable {name} added successfully',
            'variables': simulator.get_variable_info()
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/add_variables_json', methods=['POST'])
def add_variables_json():
    """从JSON批量添加变量"""
    try:
        simulator = get_user_simulator()
        
        data = request.json
        json_data = data.get('variables')
        
        if not json_data:
            return jsonify({'error': '未提供 variables 字段'}), 400
        
        # 如果是字符串，尝试解析为JSON
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                return jsonify({'error': f'JSON 格式错误: {str(e)}'}), 400
        
        # 批量添加变量
        simulator.add_variables_from_json(json_data)
        
        return jsonify({
            'success': True,
            'message': f'成功添加 {len(json_data)} 个变量',
            'variables': simulator.get_variable_info()
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/list_variables', methods=['GET'])
def list_variables():
    """列出所有变量"""
    try:
        simulator = get_user_simulator()
        
        return jsonify({
            'success': True,
            'variables': simulator.get_variable_info()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_simulation', methods=['POST'])
def run_simulation():
    """运行模拟"""
    try:
        simulator = get_user_simulator()
        
        data = request.json
        formula_str = data.get('formula')  # 例如: "var_A + var_B * var_C"
        result_name = data.get('result_name', 'Result')
        n_samples = data.get('n_samples', 1000000)
        cdf_fit_degree = data.get('cdf_fit_degree', 5)  # CDF拟合次数
        use_custom_function = data.get('use_custom_function', False)
        custom_function_code = data.get('custom_function_code', '')
        
        # 创建函数
        var_names = list(simulator.variables.keys())
        
        if use_custom_function and custom_function_code:
            # 使用自定义 Python 函数
            # 创建一个安全的命名空间
            safe_namespace = {
                'np': np,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'log10': np.log10,
                'ln': np.log,  # ln 是自然对数的别名
                'sqrt': np.sqrt,
                'abs': np.abs,
                'pow': np.power,
            }
            
            # 执行自定义函数代码
            exec(custom_function_code, safe_namespace)
            
            # 获取函数（假设函数名为 custom_formula）
            if 'custom_formula' not in safe_namespace:
                return jsonify({'error': 'Custom function must be named "custom_formula"'}), 400
            
            formula_func = safe_namespace['custom_formula']
        else:
            # 使用简单公式
            formula_code = formula_str
            
            # 支持的数学函数
            formula_code = formula_code.replace('sin(', 'np.sin(')
            formula_code = formula_code.replace('cos(', 'np.cos(')
            formula_code = formula_code.replace('tan(', 'np.tan(')
            formula_code = formula_code.replace('exp(', 'np.exp(')
            formula_code = formula_code.replace('log(', 'np.log(')
            formula_code = formula_code.replace('log10(', 'np.log10(')
            formula_code = formula_code.replace('ln(', 'np.log(')
            formula_code = formula_code.replace('sqrt(', 'np.sqrt(')
            
            # 将变量名转换为字典访问形式
            for var_name in var_names:
                formula_code = formula_code.replace(var_name, f"vars['{var_name}']")
            
            # 创建lambda函数
            formula_func = eval(f"lambda vars: {formula_code}", {'np': np})
        
        # 运行模拟（包含CDF拟合）
        simulator.run_simulation(formula_func, result_name, n_samples, formula_str, cdf_fit_degree)
        
        return jsonify({
            'success': True,
            'message': 'Simulation completed successfully',
            'result_name': result_name
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/analyze_result', methods=['POST'])
def analyze_result():
    """分析结果"""
    try:
        simulator = get_user_simulator()
        
        data = request.json
        confidence_levels = data.get('confidence_levels', [0.95, 0.99])
        
        analysis = simulator.analyze_result(confidence_levels)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/generate_charts', methods=['POST'])
def generate_charts():
    """生成选中的图表"""
    try:
        simulator = get_user_simulator()
        
        data = request.json
        chart_types = data.get('chart_types', [])  # ['result_plot', 'var_distributions', 'pareto', 'tornado']
        
        charts = {}
        
        # 结果分布图
        if 'result_plot' in chart_types:
            if 'result_plot' in simulator.chart_cache:
                charts['result_plot'] = simulator.chart_cache['result_plot']
            else:
                img_base64 = simulator.plot_result()
                simulator.chart_cache['result_plot'] = img_base64
                charts['result_plot'] = img_base64
        
        # 变量分布图
        if 'var_distributions' in chart_types:
            var_charts = {}
            for var_name, var in simulator.variables.items():
                cache_key = f'var_dist_{var_name}'
                if cache_key in simulator.chart_cache:
                    var_charts[var_name] = simulator.chart_cache[cache_key]
                else:
                    img_base64 = var.plot_distribution(show_samples=True)
                    simulator.chart_cache[cache_key] = img_base64
                    var_charts[var_name] = img_base64
            charts['var_distributions'] = var_charts
        
        # Pareto 图
        if 'pareto' in chart_types:
            if 'pareto' in simulator.chart_cache:
                charts['pareto'] = simulator.chart_cache['pareto']
            else:
                img_base64 = simulator.plot_pareto_chart()
                simulator.chart_cache['pareto'] = img_base64
                charts['pareto'] = img_base64
        
        # Tornado 图
        if 'tornado' in chart_types:
            if 'tornado' in simulator.chart_cache:
                charts['tornado'] = simulator.chart_cache['tornado']
            else:
                img_base64 = simulator.plot_tornado_chart()
                simulator.chart_cache['tornado'] = img_base64
                charts['tornado'] = img_base64
        
        return jsonify({
            'success': True,
            'charts': charts
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/generate_report', methods=['POST'])
def generate_report_api():
    """生成报告（仅文本格式）"""
    try:
        simulator = get_user_simulator()
        
        data = request.json
        confidence_levels = data.get('confidence_levels', [0.95, 0.99])
        
        # 使用 monte_carlo.py 中的报告生成方法
        report = simulator.generate_report(confidence_levels)
        
        return jsonify({
            'success': True,
            'report': report,
            'type': 'text'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    """重置模拟器"""
    try:
        # 重置当前用户的模拟器
        if 'user_id' in session:
            user_id = session['user_id']
            user_simulators[user_id] = MonteCarloSimulator()
        
        return jsonify({
            'success': True,
            'message': 'Simulator reset successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)