import os
import time
import platform
import psutil
from datetime import datetime
from extended_clustering import main as run_extended_clustering
from generate_report import main as generate_report

def print_header(title, char='=', width=70):
    """打印美观的标题栏"""
    print(char * width)
    print(f"{title:^{width}}")
    print(char * width)

def print_section(title, char='-', width=70):
    """打印区域标题"""
    print(f"\n{char * 3} {title} {char * (width - len(title) - 5)}")

def get_system_info():
    """获取系统信息"""
    system = platform.system()
    release = platform.release()
    processor = platform.processor()
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024 ** 3)
    
    return {
        "system": f"{system} {release}",
        "processor": processor,
        "memory": f"{memory_gb:.1f} GB",
        "python": platform.python_version(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == "__main__":
    # 系统标题
    print_header("NHANES健康群体聚类分析与个性化健康管理系统", "=", 70)
    
    # 系统信息
    try:
        sys_info = get_system_info()
        print(f"\n运行环境:")
        print(f"📅 日期时间: {sys_info['time']}")
        print(f"💻 操作系统: {sys_info['system']}")
        print(f"🔧 处理器: {sys_info['processor']}")
        print(f"🧠 系统内存: {sys_info['memory']}")
        print(f"🐍 Python版本: {sys_info['python']}")
    except:
        # 忽略系统信息获取错误，不影响主程序运行
        pass
    
    # 确保结果目录存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(current_dir), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"\n📁 创建结果目录: {results_dir}")
    else:
        print(f"\n📁 结果将保存至: {results_dir}")
    
    # 第一阶段：聚类分析
    total_start_time = time.time()
    print_section("第一阶段：健康群体聚类分析", "=")
    
    stage1_start = time.time()
    run_extended_clustering()
    stage1_end = time.time()
    stage1_time = stage1_end - stage1_start
    
    # 第二阶段：生成报告
    print_section("第二阶段：生成聚类分析报告", "=")
    
    stage2_start = time.time()
    generate_report()
    stage2_end = time.time()
    stage2_time = stage2_end - stage2_start
    
    # 总结信息
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print_section("分析完成摘要", "=")
    print(f"✅ 聚类分析阶段用时: {stage1_time:.2f}秒")
    print(f"✅ 报告生成阶段用时: {stage2_time:.2f}秒")
    print(f"✅ 全部处理总用时: {total_time:.2f}秒")
    
    # 分析结果位置
    print_section("分析结果", "-")
    print(f"📊 CSV聚类结果文件:")
    print(f"  - {os.path.join(results_dir, 'kmeans_results.csv')}")
    print(f"  - {os.path.join(results_dir, 'hierarchical_results.csv')}")
    print(f"  - {os.path.join(results_dir, 'dbscan_results.csv')}")
    print(f"  - {os.path.join(results_dir, 'spectral_results.csv')}")
    
    print(f"\n📈 聚类可视化文件:")
    print(f"  - {os.path.join(results_dir, 'kmeans_pca_visualization.png')}")
    print(f"  - {os.path.join(results_dir, 'hierarchical_pca_visualization.png')}")
    print(f"  - {os.path.join(results_dir, 'dbscan_pca_visualization.png')}")
    print(f"  - {os.path.join(results_dir, 'spectral_pca_visualization.png')}")
    
    print(f"\n📋 健康管理建议文件:")
    print(f"  - {os.path.join(results_dir, 'kmeans_health_recommendations.txt')}")
    print(f"  - {os.path.join(results_dir, 'hierarchical_health_recommendations.txt')}")
    print(f"  - {os.path.join(results_dir, 'dbscan_health_recommendations.txt')}")
    print(f"  - {os.path.join(results_dir, 'spectral_health_recommendations.txt')}")
    
    print(f"\n📑 综合报告文件:")
    print(f"  - {os.path.join(results_dir, 'clustering_report.html')}")
    
    # 建议操作
    print_section("后续操作建议", "-")
    print(f"💡 请使用浏览器打开以下HTML文件查看完整交互式分析报告:")
    print(f"  👉 {os.path.join(results_dir, 'clustering_report.html')}")
    print(f"💡 此报告包含所有聚类算法的比较、可视化及健康管理建议")
    
    print_header("分析完成", "=", 70) 