from flask import Flask, request, jsonify, send_from_directory, session, send_file
from flask_cors import CORS
from monte_carlo import MonteCarloSimulator
import json
import traceback
import numpy as np
import secrets
import uuid
from datetime import datetime, timedelta
import zipfile
import io
import base64

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


# ==================== 新增：下载 ZIP 功能 ====================

def format_cdf_fit_details(fit_result, fit_type):
    """格式化 CDF 拟合详情"""
    lines = []
    lines.append("="*70)
    lines.append(f"  CDF {'Forward' if fit_type == 'forward' else 'Inverse'} Fit Details")
    lines.append("="*70)
    lines.append("")
    lines.append(f"Fit Type: {'Result → CDF' if fit_type == 'forward' else 'CDF → Result'}")
    lines.append(f"Degree: {fit_result['degree']}")
    lines.append(f"R² Score: {fit_result['r_squared']:.8f}")
    lines.append(f"RMSE: {fit_result['rmse']:.8f}")
    lines.append("")
    lines.append("Polynomial Coefficients (highest to lowest degree):")
    
    coeffs = fit_result['coefficients']
    for i, coeff in enumerate(coeffs):
        power = fit_result['degree'] - i
        lines.append(f"  {'a' if fit_type == 'forward' else 'b'}{power} = {coeff:.15e}")
    
    lines.append("")
    lines.append("Polynomial Formula:")
    
    if fit_type == 'forward':
        func_parts = []
        for i, coeff in enumerate(coeffs):
            power = fit_result['degree'] - i
            if power == 0:
                func_parts.append(f"{coeff:.6e}")
            elif power == 1:
                func_parts.append(f"{coeff:.6e}*x")
            else:
                func_parts.append(f"{coeff:.6e}*x^{power}")
        lines.append(f"  CDF(x) = {' + '.join(func_parts).replace('+ -', '- ')}")
    else:
        func_parts = []
        for i, coeff in enumerate(coeffs):
            power = fit_result['degree'] - i
            if power == 0:
                func_parts.append(f"{coeff:.6e}")
            elif power == 1:
                func_parts.append(f"{coeff:.6e}*p")
            else:
                func_parts.append(f"{coeff:.6e}*p^{power}")
        lines.append(f"  x(p) = {' + '.join(func_parts).replace('+ -', '- ')}")
    
    lines.append("")
    lines.append("="*70)
    
    return "\n".join(lines)


def format_sensitivity_csv(sensitivity_results):
    """格式化敏感性分析为 CSV"""
    lines = []
    lines.append("Rank,Variable,Correlation,Abs_Correlation,P_Value")
    
    # 按绝对相关系数排序
    sorted_vars = sorted(sensitivity_results.items(), 
                        key=lambda x: x[1]['abs_correlation'], 
                        reverse=True)
    
    for rank, (var_name, data) in enumerate(sorted_vars, 1):
        lines.append(f"{rank},{var_name},{data['correlation']:.6f},"
                    f"{data['abs_correlation']:.6f},{data['p_value']:.6e}")
    
    return "\n".join(lines)


@app.route('/api/download_csv_zip', methods=['POST'])
def download_csv_zip():
    """下载 CSV ZIP 包"""
    try:
        simulator = get_user_simulator()
        
        if simulator.result is None:
            return jsonify({'error': '请先运行模拟'}), 400
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            # 添加 CSV
            csv_content = simulator.export_to_csv()
            zf.writestr('raw_data.csv', csv_content)
        
        zip_buffer.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'monte_carlo_data_csv_{timestamp}.zip'
        
        return send_file(zip_buffer, mimetype='application/zip',
                        as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/download_json_zip', methods=['POST'])
def download_json_zip():
    """下载 JSON ZIP 包"""
    try:
        simulator = get_user_simulator()
        
        if simulator.result is None:
            return jsonify({'error': '请先运行模拟'}), 400
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            # 添加 JSON
            json_content = simulator.export_to_json()
            zf.writestr('data.json', json_content)
        
        zip_buffer.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'monte_carlo_data_json_{timestamp}.zip'
        
        return send_file(zip_buffer, mimetype='application/zip',
                        as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/download_report_zip', methods=['POST'])
def download_report_zip():
    """下载分析报告 ZIP 包"""
    try:
        simulator = get_user_simulator()
        
        if simulator.result is None:
            return jsonify({'error': '请先运行模拟'}), 400
        
        data = request.json
        confidence_levels = data.get('confidence_levels', [0.95, 0.99])
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            # 主报告
            report = simulator.generate_report(confidence_levels)
            zf.writestr('analysis_report.txt', report)
            
            # CDF Forward Fit 详细信息
            if simulator.cdf_fit_result:
                forward_fit = format_cdf_fit_details(simulator.cdf_fit_result, 'forward')
                zf.writestr('cdf_forward_fit.txt', forward_fit)
            
            # CDF Inverse Fit 详细信息
            if simulator.cdf_inverse_fit_result:
                inverse_fit = format_cdf_fit_details(simulator.cdf_inverse_fit_result, 'inverse')
                zf.writestr('cdf_inverse_fit.txt', inverse_fit)
        
        zip_buffer.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'monte_carlo_report_{timestamp}.zip'
        
        return send_file(zip_buffer, mimetype='application/zip',
                        as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/download_full_zip', methods=['POST'])
def download_full_zip():
    """下载完整报告（含图表）ZIP 包"""
    try:
        simulator = get_user_simulator()
        
        if simulator.result is None:
            return jsonify({'error': '请先运行模拟'}), 400
        
        data = request.json
        confidence_levels = data.get('confidence_levels', [0.95, 0.99])
        
        # 如果图表还没生成，先生成
        if 'result_plot' not in simulator.chart_cache:
            simulator.chart_cache['result_plot'] = simulator.plot_result()
        if 'pareto' not in simulator.chart_cache and simulator.sensitivity_results:
            simulator.chart_cache['pareto'] = simulator.plot_pareto_chart()
        if 'tornado' not in simulator.chart_cache and simulator.sensitivity_results:
            simulator.chart_cache['tornado'] = simulator.plot_tornado_chart()
        
        # 生成变量分布图
        for var_name, var in simulator.variables.items():
            cache_key = f'var_dist_{var_name}'
            if cache_key not in simulator.chart_cache:
                simulator.chart_cache[cache_key] = var.plot_distribution()
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            # 文本报告
            report = simulator.generate_report(confidence_levels)
            zf.writestr('analysis_report.txt', report)
            
            # 图表 PNG 文件
            if 'result_plot' in simulator.chart_cache:
                img_bytes = base64.b64decode(simulator.chart_cache['result_plot'])
                zf.writestr('charts/result_distribution.png', img_bytes)
            
            if 'pareto' in simulator.chart_cache:
                img_bytes = base64.b64decode(simulator.chart_cache['pareto'])
                zf.writestr('charts/pareto_chart.png', img_bytes)
            
            if 'tornado' in simulator.chart_cache:
                img_bytes = base64.b64decode(simulator.chart_cache['tornado'])
                zf.writestr('charts/tornado_chart.png', img_bytes)
            
            # 变量分布图
            for var_name, var in simulator.variables.items():
                cache_key = f'var_dist_{var_name}'
                if cache_key in simulator.chart_cache:
                    img_bytes = base64.b64decode(simulator.chart_cache[cache_key])
                    zf.writestr(f'charts/variables/{var_name}_distribution.png', img_bytes)
        
        zip_buffer.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'monte_carlo_full_{timestamp}.zip'
        
        return send_file(zip_buffer, mimetype='application/zip',
                        as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)