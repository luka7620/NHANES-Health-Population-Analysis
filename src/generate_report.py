import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_results():
    """
    加载各种聚类算法的结果
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(current_dir), "results")
    
    results = {}
    algorithms = ['kmeans', 'hierarchical', 'dbscan', 'spectral']
    
    for algorithm in algorithms:
        # 检查结果文件是否存在
        result_path = os.path.join(results_dir, f"{algorithm}_results.csv")
        if os.path.exists(result_path):
            results[algorithm] = pd.read_csv(result_path)
            print(f"已加载 {algorithm} 聚类结果")
        else:
            print(f"警告: {algorithm} 聚类结果文件不存在")
    
    return results, results_dir

def extract_recommendations(results_dir, algorithms):
    """
    从健康管理建议文件中提取关键信息
    """
    recommendations = {}
    
    for algorithm in algorithms:
        rec_path = os.path.join(results_dir, f"{algorithm}_health_recommendations.txt")
        if os.path.exists(rec_path):
            with open(rec_path, 'r', encoding='utf-8') as f:
                content = f.read()
                recommendations[algorithm] = content
            print(f"已加载 {algorithm} 健康管理建议")
        else:
            print(f"警告: {algorithm} 健康管理建议文件不存在")
    
    return recommendations

def compare_algorithms(results):
    """
    比较不同聚类算法的结果
    """
    comparison = {}
    
    for algorithm, df in results.items():
        if 'Cluster' in df.columns:
            clusters = df['Cluster'].values
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            noise_count = list(clusters).count(-1) if -1 in clusters else 0
            noise_ratio = noise_count / len(clusters)
            
            comparison[algorithm] = {
                'n_clusters': n_clusters,
                'noise_count': noise_count,
                'noise_ratio': noise_ratio,
                'total_samples': len(clusters)
            }
    
    return comparison

def visualize_comparison(comparison, results_dir):
    """
    可视化不同算法的比较结果
    """
    plt.figure(figsize=(12, 8))
    
    # 簇数量比较
    algorithms = list(comparison.keys())
    n_clusters = [comp['n_clusters'] for comp in comparison.values()]
    noise_ratios = [comp['noise_ratio'] * 100 for comp in comparison.values()]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    rects1 = ax1.bar(x - width/2, n_clusters, width, label='群体数量')
    ax1.set_ylabel('群体数量')
    ax1.set_title('聚类算法比较')
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, noise_ratios, width, label='噪声点比例 (%)', color='orange')
    ax2.set_ylabel('噪声点比例 (%)')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(results_dir, 'algorithm_comparison.png'), dpi=300)
    plt.close()

def generate_html_report(results, comparison, recommendations, results_dir):
    """
    生成HTML格式的总结报告
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>NHANES健康群体聚类分析报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .footer {{ margin-top: 50px; border-top: 1px solid #ddd; padding-top: 20px; color: #777; }}
            img {{ max-width: 100%; height: auto; }}
            pre {{ background-color: #f9f9f9; padding: 15px; border: 1px solid #ddd; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>NHANES健康群体聚类分析报告</h1>
            <p>生成时间: {now}</p>
            
            <div class="section">
                <h2>算法比较</h2>
                <table>
                    <tr>
                        <th>算法</th>
                        <th>群体数量</th>
                        <th>样本总数</th>
                        <th>噪声点数量</th>
                        <th>噪声点比例</th>
                    </tr>
    """
    
    for algorithm, comp in comparison.items():
        html_content += f"""
                    <tr>
                        <td>{algorithm}</td>
                        <td>{comp['n_clusters']}</td>
                        <td>{comp['total_samples']}</td>
                        <td>{comp['noise_count']}</td>
                        <td>{comp['noise_ratio']:.2%}</td>
                    </tr>
        """
    
    html_content += """
                </table>
                <img src="algorithm_comparison.png" alt="算法比较" />
            </div>
            
            <div class="section">
                <h2>可视化结果</h2>
    """
    
    for algorithm in comparison.keys():
        html_content += f"""
                <h3>{algorithm} 聚类结果</h3>
                <img src="{algorithm}_pca_visualization.png" alt="{algorithm} 聚类可视化" />
                <p>不同颜色表示不同的健康群体。通过PCA降维将高维健康指标映射到二维空间。</p>
        """
    
    html_content += """
            </div>
            
            <div class="section">
                <h2>健康群体分析与管理建议</h2>
    """
    
    for algorithm, content in recommendations.items():
        html_content += f"""
                <h3>{algorithm} 聚类健康群体分析</h3>
                <pre>{content}</pre>
        """
    
    html_content += """
            </div>
            
            <div class="section">
                <h2>聚类结果总结</h2>
                <p>本项目使用多种聚类算法对NHANES健康数据集进行了分析，主要发现如下：</p>
                <ol>
                    <li>K-means算法产生了形状较为均匀的健康群体，适合识别主要健康类型</li>
                    <li>层次聚类能够发现健康群体间的层次关系，更好地理解健康状况的渐变过程</li>
                    <li>DBSCAN算法能够识别不规则形状的健康群体，并标记异常健康状况为噪声点</li>
                    <li>谱聚类算法能够捕捉数据的整体结构特性，适合发现复杂拓扑结构的健康群体</li>
                </ol>
                <p>通过分析这些群体的特征，我们可以为不同人群提供个性化的健康管理建议，提高健康水平。</p>
            </div>
            
            <div class="footer">
                <p>NHANES健康群体聚类分析与个性化健康管理系统 | 生成时间: {now}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML报告
    with open(os.path.join(results_dir, 'clustering_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已生成: {os.path.join(results_dir, 'clustering_report.html')}")

def main():
    print("开始生成聚类分析报告...")
    
    # 加载聚类结果
    results, results_dir = load_results()
    
    if not results:
        print("错误: 没有找到任何聚类结果文件")
        return
    
    # 提取健康管理建议
    recommendations = extract_recommendations(results_dir, results.keys())
    
    # 比较不同算法
    comparison = compare_algorithms(results)
    
    # 可视化比较
    visualize_comparison(comparison, results_dir)
    
    # 生成HTML报告
    generate_html_report(results, comparison, recommendations, results_dir)
    
    print("聚类分析报告生成完成!")

if __name__ == "__main__":
    main() 