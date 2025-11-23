        // 显示本地状态（在特定区域）
        function showLocalStatus(elementId, message, type = 'info', showSpinner = false) {
            const statusDiv = document.getElementById(elementId);
            
            let spinnerHtml = '';
            if (showSpinner) {
                spinnerHtml = '<div class="spinner"></div>';
            }
            
            statusDiv.innerHTML = `${spinnerHtml}<span>${message}</span>`;
            statusDiv.className = `local-status ${type}`;
            statusDiv.style.display = 'flex';
        }

        // 隐藏本地状态
        function hideLocalStatus(elementId) {
            const statusDiv = document.getElementById(elementId);
            statusDiv.style.display = 'none';
        }

        // Toggle collapsible sections
        function toggleCollapsible(header) {
            header.classList.toggle('active');
            const content = header.nextElementSibling;
            content.classList.toggle('show');
        }

        // Toggle formula input type
        document.querySelectorAll('input[name="formulaType"]').forEach(radio => {
            radio.addEventListener('change', function() {
                if (this.value === 'simple') {
                    // 显示 Simple Formula 输入框和示例
                    document.getElementById('simpleFormulaDiv').style.display = 'block';
                    document.getElementById('advancedFormulaDiv').style.display = 'none';
                    document.getElementById('simpleFormulaExamples').style.display = 'block';
                    document.getElementById('advancedFormulaExamples').style.display = 'none';
                } else {
                    // 显示 Advanced Formula 输入框和示例
                    document.getElementById('simpleFormulaDiv').style.display = 'none';
                    document.getElementById('advancedFormulaDiv').style.display = 'block';
                    document.getElementById('simpleFormulaExamples').style.display = 'none';
                    document.getElementById('advancedFormulaExamples').style.display = 'block';
                }
            });
        });

        function showStatus(message, type = 'info') {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = type;
            status.style.display = 'block';
            setTimeout(() => {
                status.style.display = 'none';
            }, 5000);
        }

        function addVariablesFromJson() {
            const jsonInput = document.getElementById('jsonInput').value.trim();
            
            if (!jsonInput) {
                showLocalStatus('addVariableStatus', 'Please enter JSON variable definition', 'error', false);
                showStatus('Please enter JSON variable definition', 'error');
                return;
            }

            // Show loading state
            showLocalStatus('addVariableStatus', 'Adding variables...', 'info', true);
            showStatus('Adding variables...', 'info');

            fetch('/api/add_variables_json', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ variables: jsonInput })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showLocalStatus('addVariableStatus', `✅ Successfully added ${Object.keys(JSON.parse(jsonInput)).length} variable(s)`, 'success', false);
                    showStatus(data.message, 'success');
                    updateVariableList(data.variables);
                } else {
                    showLocalStatus('addVariableStatus', '❌ Error: ' + data.error, 'error', false);
                    showStatus('Error: ' + data.error, 'error');
                }
            })
            .catch(error => {
                showLocalStatus('addVariableStatus', '❌ Request failed: ' + error.message, 'error', false);
                showStatus('Request failed: ' + error.message, 'error');
            });
        }

        function updateVariableList(variables) {
            const listDiv = document.getElementById('variableList');
            const itemsDiv = document.getElementById('variableItems');
            
            if (Object.keys(variables).length === 0) {
                listDiv.style.display = 'none';
                return;
            }

            listDiv.style.display = 'block';
            itemsDiv.innerHTML = '';

            for (const [name, info] of Object.entries(variables)) {
                const item = document.createElement('div');
                item.className = 'variable-item';
                
                let paramStr = '';
                if (info.dist_type === 'norm' || info.dist_type === 'normal') {
                    paramStr = `μ=${info.dist_params[0].toFixed(3)}, σ=${info.dist_params[1].toFixed(3)}`;
                } else if (info.dist_type === 't') {
                    paramStr = `df=${info.dist_params[0].toFixed(1)}, loc=${info.dist_params[1].toFixed(3)}, scale=${info.dist_params[2].toFixed(3)}`;
                }

                // Format range
                let rangeStr = '';
                const minVal = info.min_value;
                const maxVal = info.max_value;
                
                if (minVal === null && maxVal === null) {
                    rangeStr = '(-∞, +∞)';
                } else if (minVal === null) {
                    rangeStr = `(-∞, ${maxVal.toFixed(3)}]`;
                } else if (maxVal === null) {
                    rangeStr = `[${minVal.toFixed(3)}, +∞)`;
                } else {
                    rangeStr = `[${minVal.toFixed(3)}, ${maxVal.toFixed(3)}]`;
                }

                item.innerHTML = `
                    <div>
                        <strong>${name}</strong>
                        <span class="badge badge-info">${info.dist_type}</span>
                    </div>
                    <div class="variable-info">${paramStr}</div>
                    <div class="variable-info" style="color: #666; font-size: 0.9em;">Range: ${rangeStr}</div>
                `;
                itemsDiv.appendChild(item);
            }
        }

        function runSimulation() {
            const formulaType = document.querySelector('input[name="formulaType"]:checked').value;
            const resultName = document.getElementById('resultName').value;
            const nSamples = parseInt(document.getElementById('nSamples').value);
            const cdfDegree = parseInt(document.getElementById('cdfDegree').value);

            let requestData = {
                result_name: resultName,
                n_samples: nSamples,
                cdf_fit_degree: cdfDegree
            };

            if (formulaType === 'simple') {
                requestData.formula = document.getElementById('formula').value;
                requestData.use_custom_function = false;
            } else {
                requestData.custom_function_code = document.getElementById('customFunction').value;
                requestData.use_custom_function = true;
            }

            // Estimate running time (based on sample count)
            // Assume ~1 second per 500,000 samples
            const estimatedTime = Math.max(2, (nSamples / 500000));

            // Show progress bar
            const statusHtml = `
                <div style="flex: 1;">
                    <div style="margin-bottom: 5px; font-size: 0.9em; color: #5F5F5F;">
                        Running simulation... (${nSamples.toLocaleString()} samples)
                    </div>
                    <div class="progress-container">
                        <div class="progress-text" id="progressText">0%</div>
                        <div class="progress-bar" id="progressBar" style="width: 0%;"></div>
                    </div>
                </div>
            `;
            
            const statusDiv = document.getElementById('simulationStatus');
            statusDiv.innerHTML = statusHtml;
            statusDiv.className = 'local-status info';
            statusDiv.style.display = 'flex';

            showStatus('Simulation in progress, please wait...', 'info');

            // Progress bar animation
            let progress = 0;
            const progressInterval = setInterval(() => {
                // Use non-linear growth (fast at start, slow at end)
                if (progress < 30) {
                    progress += 2;
                } else if (progress < 60) {
                    progress += 1;
                } else if (progress < 90) {
                    progress += 0.5;
                } else if (progress < 95) {
                    progress += 0.2;
                }
                
                progress = Math.min(progress, 95); // Max 95%, wait for actual completion
                
                const progressBar = document.getElementById('progressBar');
                const progressText = document.getElementById('progressText');
                if (progressBar && progressText) {
                    progressBar.style.width = progress + '%';
                    progressText.textContent = Math.floor(progress) + '%';
                }
            }, estimatedTime * 1000 / 50); // Update in 50 steps

            const startTime = Date.now();

            fetch('/api/run_simulation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                clearInterval(progressInterval);
                const duration = ((Date.now() - startTime) / 1000).toFixed(1);
                
                if (data.success) {
                    // Set to 100% when complete
                    const progressBar = document.getElementById('progressBar');
                    const progressText = document.getElementById('progressText');
                    if (progressBar && progressText) {
                        progressBar.style.width = '100%';
                        progressText.textContent = '100%';
                    }
                    
                    // Show success message after 0.3s delay
                    setTimeout(() => {
                        showLocalStatus('simulationStatus', 
                            `✅ Simulation complete! Time: ${duration}s | Samples: ${nSamples.toLocaleString()} | CDF degree: ${cdfDegree}`, 
                            'success', false);
                    }, 300);
                    
                    showStatus(`Simulation completed successfully in ${duration}s`, 'success');
                } else {
                    showLocalStatus('simulationStatus', '❌ Error: ' + data.error, 'error', false);
                    showStatus('Simulation failed: ' + data.error, 'error');
                }
            })
            .catch(error => {
                clearInterval(progressInterval);
                showLocalStatus('simulationStatus', '❌ Request failed: ' + error.message, 'error', false);
                showStatus('Request failed: ' + error.message, 'error');
            });
        }

        function generateCharts() {
            const chartTypes = [];
            document.querySelectorAll('.checkbox-item input[type="checkbox"]:checked').forEach(cb => {
                chartTypes.push(cb.value);
            });

            if (chartTypes.length === 0) {
                showLocalStatus('chartStatus', '⚠️ Please select at least one chart type', 'warning', false);
                return;
            }

            // Show loading state
            showLocalStatus('chartStatus', `Generating ${chartTypes.length} chart(s)...`, 'info', true);
            showStatus('Generating charts, please wait...', 'info');

            const startTime = Date.now();

            fetch('/api/generate_charts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chart_types: chartTypes })
            })
            .then(response => response.json())
            .then(data => {
                const duration = ((Date.now() - startTime) / 1000).toFixed(1);
                
                if (data.success) {
                    showLocalStatus('chartStatus', `✅ Charts generated! Time: ${duration}s`, 'success', false);
                    showStatus(`Successfully generated ${chartTypes.length} chart(s)`, 'success');
                    displayCharts(data.charts);
                } else {
                    showLocalStatus('chartStatus', '❌ Error: ' + data.error, 'error', false);
                    showStatus('Chart generation failed: ' + data.error, 'error');
                }
            })
            .catch(error => {
                showLocalStatus('chartStatus', '❌ Request failed: ' + error.message, 'error', false);
                showStatus('Request failed: ' + error.message, 'error');
            });
        }

        function displayCharts(charts) {
            const resultsDiv = document.getElementById('chartResults');
            resultsDiv.innerHTML = '';
            resultsDiv.style.display = 'block';

            if (charts.result_plot) {
                const title = document.createElement('h3');
                title.textContent = 'Result Distribution';
                title.style.marginTop = '20px';
                resultsDiv.appendChild(title);

                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + charts.result_plot;
                img.className = 'chart-image';
                resultsDiv.appendChild(img);
            }

            if (charts.var_distributions) {
                const title = document.createElement('h3');
                title.textContent = 'Variable Distributions';
                title.style.marginTop = '20px';
                resultsDiv.appendChild(title);

                for (const [varName, imgData] of Object.entries(charts.var_distributions)) {
                    const varTitle = document.createElement('h4');
                    varTitle.textContent = varName;
                    varTitle.style.marginTop = '15px';
                    resultsDiv.appendChild(varTitle);

                    // 添加容器确保宽度一致
                    const imgContainer = document.createElement('div');
                    imgContainer.style.width = '100%';
                    imgContainer.style.display = 'block';
                    
                    const img = document.createElement('img');
                    img.src = 'data:image/png;base64,' + imgData;
                    img.className = 'chart-image';
                    img.style.width = '100%';
                    img.style.height = 'auto';
                    img.style.display = 'block';
                    
                    imgContainer.appendChild(img);
                    resultsDiv.appendChild(imgContainer);
                }
            }

            if (charts.pareto) {
                const title = document.createElement('h3');
                title.textContent = 'Pareto Chart';
                title.style.marginTop = '20px';
                resultsDiv.appendChild(title);

                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + charts.pareto;
                img.className = 'chart-image';
                resultsDiv.appendChild(img);
            }

            if (charts.tornado) {
                const title = document.createElement('h3');
                title.textContent = 'Tornado Chart';
                title.style.marginTop = '20px';
                resultsDiv.appendChild(title);

                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + charts.tornado;
                img.className = 'chart-image';
                resultsDiv.appendChild(img);
            }
        }

        function generateReport() {
            // Show loading state
            showLocalStatus('reportStatus', 'Generating report...', 'info', true);
            showStatus('Generating report, please wait...', 'info');

            const startTime = Date.now();

            fetch('/api/generate_report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            })
            .then(response => response.json())
            .then(data => {
                const duration = ((Date.now() - startTime) / 1000).toFixed(1);
                
                if (data.success) {
                    showLocalStatus('reportStatus', `✅ Report generated! Time: ${duration}s`, 'success', false);
                    showStatus('Report generated successfully', 'success');
                    displayReport(data.report);
                } else {
                    showLocalStatus('reportStatus', '❌ Error: ' + data.error, 'error', false);
                    showStatus('Report generation failed: ' + data.error, 'error');
                }
            })
            .catch(error => {
                showLocalStatus('reportStatus', '❌ Request failed: ' + error.message, 'error', false);
                showStatus('Request failed: ' + error.message, 'error');
            });
        }

        function displayReport(report) {
            const resultsDiv = document.getElementById('reportResult');
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<pre>' + report + '</pre>';
        }

        function resetSimulator() {
            if (!confirm('Are you sure you want to reset all data?')) {
                return;
            }

            showStatus('Resetting...', 'info');

            fetch('/api/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('Simulator reset successfully', 'success');
                    document.getElementById('jsonInput').value = '';
                    document.getElementById('variableList').style.display = 'none';
                    document.getElementById('chartResults').style.display = 'none';
                    document.getElementById('reportResult').style.display = 'none';
                    
                    // Hide all local status
                    hideLocalStatus('addVariableStatus');
                    hideLocalStatus('simulationStatus');
                    hideLocalStatus('chartStatus');
                    hideLocalStatus('reportStatus');
                } else {
                    showStatus('Reset failed: ' + data.error, 'error');
                }
            })
            .catch(error => {
                showStatus('Request failed: ' + error.message, 'error');
            });
        }

        // ==================== 下载功能 ====================
        
        // 切换下拉菜单显示
        function toggleDownloadMenu() {
            const menu = document.getElementById('downloadMenu');
            menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
        }

        // 点击页面其他地方关闭下拉菜单
        document.addEventListener('click', function(event) {
            const btn = document.getElementById('downloadBtn');
            const menu = document.getElementById('downloadMenu');
            if (btn && menu && !btn.contains(event.target) && !menu.contains(event.target)) {
                menu.style.display = 'none';
            }
        });

        // 通用下载函数
        async function downloadFile(endpoint, filename) {
            showLocalStatus('reportStatus', '📥 Preparing download...', 'info', true);
            showStatus('Preparing download, please wait...', 'info');

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });

                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.error || 'Download failed');
                }

                // 获取文件 blob
                const blob = await response.blob();
                
                // 创建下载链接
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename || 'download.zip';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);

                showLocalStatus('reportStatus', '✅ Download complete!', 'success', false);
                showStatus('Download completed successfully', 'success');
                
                // 3秒后隐藏状态
                setTimeout(() => {
                    hideLocalStatus('reportStatus');
                }, 3000);

            } catch (error) {
                showLocalStatus('reportStatus', '❌ Download failed: ' + error.message, 'error', false);
                showStatus('Download failed: ' + error.message, 'error');
            }

            // 关闭下拉菜单
            document.getElementById('downloadMenu').style.display = 'none';
        }

        // 下载 CSV
        function downloadCSV() {
            const now = new Date();
            const timestamp = `${now.getFullYear()}${String(now.getMonth()+1).padStart(2,'0')}${String(now.getDate()).padStart(2,'0')}_${String(now.getHours()).padStart(2,'0')}${String(now.getMinutes()).padStart(2,'0')}${String(now.getSeconds()).padStart(2,'0')}`;
            downloadFile('/api/download_csv_zip', `monte_carlo_data_csv_${timestamp}.zip`);
        }

        // 下载 JSON
        function downloadJSON() {
            const now = new Date();
            const timestamp = `${now.getFullYear()}${String(now.getMonth()+1).padStart(2,'0')}${String(now.getDate()).padStart(2,'0')}_${String(now.getHours()).padStart(2,'0')}${String(now.getMinutes()).padStart(2,'0')}${String(now.getSeconds()).padStart(2,'0')}`;
            downloadFile('/api/download_json_zip', `monte_carlo_data_json_${timestamp}.zip`);
        }

        // 下载报告
        function downloadReport() {
            const now = new Date();
            const timestamp = `${now.getFullYear()}${String(now.getMonth()+1).padStart(2,'0')}${String(now.getDate()).padStart(2,'0')}_${String(now.getHours()).padStart(2,'0')}${String(now.getMinutes()).padStart(2,'0')}${String(now.getSeconds()).padStart(2,'0')}`;
            downloadFile('/api/download_report_zip', `monte_carlo_report_${timestamp}.zip`);
        }

        // 下载完整包
        function downloadFull() {
            const now = new Date();
            const timestamp = `${now.getFullYear()}${String(now.getMonth()+1).padStart(2,'0')}${String(now.getDate()).padStart(2,'0')}_${String(now.getHours()).padStart(2,'0')}${String(now.getMinutes()).padStart(2,'0')}${String(now.getSeconds()).padStart(2,'0')}`;
            downloadFile('/api/download_full_zip', `monte_carlo_full_${timestamp}.zip`);
        }

        // ==================== 复制功能 ====================
        
        // 复制示例代码到剪贴板
        function copyExample(button, targetId) {
            // 获取目标输入框
            const targetElement = document.getElementById(targetId);
            if (!targetElement) {
                showStatus('Target input not found', 'error');
                return;
            }

            // 获取按钮所在的 example-code 容器
            const exampleCode = button.closest('.example-code');
            if (!exampleCode) {
                showStatus('Example code not found', 'error');
                return;
            }

            // 获取 pre > code 中的文本内容
            const codeElement = exampleCode.querySelector('pre code');
            if (!codeElement) {
                showStatus('Code content not found', 'error');
                return;
            }

            const codeText = codeElement.textContent.trim();

            // 复制到剪贴板
            navigator.clipboard.writeText(codeText).then(() => {
                // 填充到目标输入框
                targetElement.value = codeText;

                // 根据内容自动调整输入框高度
                adjustTextareaHeight(targetElement, codeText);

                // 更新按钮状态
                const originalHTML = button.innerHTML;
                button.innerHTML = '✓ Copied!';
                button.classList.add('copied');

                // 显示成功消息
                showStatus('Example copied to input field', 'success');

                // 折叠示例区域
                const collapsibleHeader = button.closest('.collapsible').querySelector('.collapsible-header');
                if (collapsibleHeader && collapsibleHeader.classList.contains('active')) {
                    toggleCollapsible(collapsibleHeader);
                }

                // 延迟聚焦，等待折叠动画完成
                setTimeout(() => {
                    // 聚焦到输入框
                    targetElement.focus();
                    
                    // 滚动到输入框位置
                    targetElement.scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'center' 
                    });
                }, 350); // 等待折叠动画（300ms）完成

                // 2秒后恢复按钮
                setTimeout(() => {
                    button.innerHTML = originalHTML;
                    button.classList.remove('copied');
                }, 2000);
            }).catch(err => {
                showStatus('Copy failed: ' + err.message, 'error');
            });
        }

        // 根据内容自动调整 textarea 高度
        function adjustTextareaHeight(element, content) {
            // 只对 textarea 元素调整高度，input 元素不需要调整
            if (element.tagName.toLowerCase() !== 'textarea') {
                return;
            }
            
            // 计算行数
            const lines = content.split('\n').length;
            
            // 根据不同的 textarea 设置不同的最小高度
            let minLines, maxLines;
            
            if (element.id === 'jsonInput') {
                // JSON 输入框：与 Python 函数一致
                minLines = 5;
                maxLines = 25;
            } else if (element.id === 'customFunction') {
                // Python 函数输入框
                minLines = 5;
                maxLines = 25;
            } else {
                // 其他 textarea
                minLines = 5;
                maxLines = 25;
            }
            
            // 计算目标行数（不增加缓冲，精确匹配内容）
            const targetLines = Math.min(Math.max(lines, minLines), maxLines);
            
            // 每行大约 20px（根据 font-size 和 line-height）
            const lineHeight = 20;
            const newHeight = targetLines * lineHeight;
            
            // 设置新高度
            element.style.height = newHeight + 'px';
            element.style.minHeight = (minLines * lineHeight) + 'px';
        }
