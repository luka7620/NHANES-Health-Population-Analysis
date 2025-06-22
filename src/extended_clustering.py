import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
# 导入谱聚类功能
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spectral_clustering import spectral_clustering

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data(sample_size=5000):
    """
    加载NHANES处理后的数据集
    """
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取数据文件路径
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    data_path = os.path.join(data_dir, "nhanes_processed.csv")
    
    print("正在加载数据...")
    df = pd.read_csv(data_path, nrows=sample_size)
    print(f"数据加载完成，形状: {df.shape}")
    return df

def preprocess_data(df):
    """
    预处理数据：选择特征、处理缺失值、标准化
    """
    print("开始数据预处理...")
    
    # 选择有用的特征列 (年龄、性别、BMI、血压、血糖、胆固醇等)
    # 核心特征必须有
    must_have_features = [
        'RIDAGEYR',     # 年龄
        'RIAGENDR',     # 性别
    ]
    
    # 其他健康指标
    health_indicators = [
        'BMXBMI',       # BMI
        'BPXSY1',       # 收缩压
        'BPXDI1',       # 舒张压
        'LBXTC',        # 总胆固醇
        'LBXTR',        # 甘油三酯
        'SMQ020',       # 吸烟状况
        'ALQ101'        # 饮酒状况
    ]
    
    # 可选的血糖和胆固醇指标
    optional_glucose = ['LBXGLU', 'LBXGH', 'LBXGLT']  # 不同形式的血糖指标
    optional_cholesterol = ['LBXHDL', 'LBDLDL']  # 不同形式的胆固醇指标
    
    # 确保必选特征存在
    for feature in must_have_features:
        if feature not in df.columns:
            raise ValueError(f"必需的特征 {feature} 不存在于数据集中")
    
    # 添加可用的健康指标
    available_features = must_have_features.copy()
    available_features.extend([col for col in health_indicators if col in df.columns])
    
    # 尝试添加血糖和胆固醇指标
    for glucose in optional_glucose:
        if glucose in df.columns:
            available_features.append(glucose)
            print(f"选择了血糖指标: {glucose}")
            break
    
    for cholesterol in optional_cholesterol:
        if cholesterol in df.columns:
            available_features.append(cholesterol)
            print(f"选择了胆固醇指标: {cholesterol}")
            break
    
    print(f"选择的特征: {available_features}")
    
    # 选择数据
    data = df[available_features].copy()
    
    # 检查缺失值
    print("缺失值统计:")
    print(data.isnull().sum())
    
    # 去除含有缺失值的行
    data = data.dropna()
    print(f"处理缺失值后的数据形状: {data.shape}")
    
    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return data, scaled_data, available_features

def reduce_dimensions(scaled_data):
    """
    使用PCA降维，便于可视化
    """
    print("正在进行PCA降维...")
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    print(f"PCA解释方差比例: {pca.explained_variance_ratio_}")
    
    return reduced_data

def kmeans_clustering(scaled_data, n_clusters=5):
    """
    K均值聚类
    """
    print(f"\n🔍 执行K均值聚类分析 (k={n_clusters}) " + "="*40)
    print(f"  · 算法特点: 简单高效，适合发现球状簇")
    print(f"  · 优势: 计算速度快，易于理解和实现")
    print(f"  · 劣势: 对初始质心敏感，只能发现凸形状的簇")
    
    # 开始计时
    start_time = time.time()
    
    # 执行聚类
    print(f"\n  开始训练K均值模型...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(scaled_data)
    
    # 计算性能指标
    end_time = time.time()
    silhouette_avg = silhouette_score(scaled_data, clusters)
    
    # 统计每个簇的样本数
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # 输出结果
    print(f"\n  📊 聚类结果:")
    print(f"    - 总样本数: {len(clusters)}")
    print(f"    - 聚类数量: {n_clusters}")
    for i in range(n_clusters):
        print(f"    - 群体 {i+1}: {cluster_counts.get(i, 0)}个样本 ({cluster_counts.get(i, 0)/len(clusters)*100:.1f}%)")
    print(f"\n  📈 性能指标:")
    print(f"    - 轮廓系数: {silhouette_avg:.4f}  (范围: -1到1, 越大越好)")
    print(f"    - 运行时间: {end_time - start_time:.2f}秒")
    
    return clusters

def hierarchical_clustering(scaled_data, n_clusters=5):
    """
    层次聚类
    """
    print(f"\n🔍 执行层次聚类分析 (k={n_clusters}) " + "="*40)
    print(f"  · 算法特点: 自底向上逐步合并样本，形成层次结构")
    print(f"  · 优势: 不需要预先指定聚类数量，能够发现嵌套关系")
    print(f"  · 劣势: 计算复杂度高，难以处理大规模数据集")
    
    # 开始计时
    start_time = time.time()
    
    # 执行聚类
    print(f"\n  开始训练层次聚类模型...")
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = hierarchical.fit_predict(scaled_data)
    
    # 计算性能指标
    end_time = time.time()
    silhouette_avg = silhouette_score(scaled_data, clusters)
    
    # 统计每个簇的样本数
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # 输出结果
    print(f"\n  📊 聚类结果:")
    print(f"    - 总样本数: {len(clusters)}")
    print(f"    - 聚类数量: {n_clusters}")
    for i in range(n_clusters):
        print(f"    - 群体 {i+1}: {cluster_counts.get(i, 0)}个样本 ({cluster_counts.get(i, 0)/len(clusters)*100:.1f}%)")
    print(f"\n  📈 性能指标:")
    print(f"    - 轮廓系数: {silhouette_avg:.4f}  (范围: -1到1, 越大越好)")
    print(f"    - 运行时间: {end_time - start_time:.2f}秒")
    
    return clusters

def dbscan_clustering(scaled_data, eps=0.5, min_samples=5):
    """
    DBSCAN聚类
    """
    print(f"\n🔍 执行DBSCAN密度聚类分析 " + "="*40)
    print(f"  · 算法特点: 基于密度的空间聚类，能识别任意形状的簇")
    print(f"  · 优势: 能发现任意形状的簇，自动识别噪声点，不需要预先指定簇数量")
    print(f"  · 劣势: 对参数敏感，处理不同密度的簇效果较差")
    print(f"  · 参数设置: eps={eps} (邻域半径), min_samples={min_samples} (最小样本数)")
    
    # 开始计时
    start_time = time.time()
    
    # 执行聚类
    print(f"\n  开始训练DBSCAN模型...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)
    
    # 计算性能指标
    end_time = time.time()
    
    # 统计聚类结果
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    noise_count = np.sum(clusters == -1)
    noise_ratio = noise_count / len(clusters)
    
    # 计算轮廓系数(如果有效)
    has_valid_silhouette = len(np.unique(clusters)) > 1 and (len(clusters) - noise_count) > 1
    if has_valid_silhouette:
        # 计算无噪声点的轮廓系数
        valid_indices = clusters != -1
        if np.sum(valid_indices) > 1:
            try:
                silhouette_avg = silhouette_score(scaled_data[valid_indices], 
                                                 clusters[valid_indices])
            except:
                silhouette_avg = "无法计算"
        else:
            silhouette_avg = "无法计算"
    else:
        silhouette_avg = "无法计算"
        
    # 统计每个簇的样本数
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # 输出结果
    print(f"\n  📊 聚类结果:")
    print(f"    - 总样本数: {len(clusters)}")
    print(f"    - 自动识别的群体数量: {n_clusters}")
    print(f"    - 噪声点: {noise_count}个样本 ({noise_ratio*100:.1f}%)")
    
    # 显示各个簇的样本数量和百分比
    for i in [k for k in cluster_counts.keys() if k != -1]:
        print(f"    - 群体 {i+1}: {cluster_counts.get(i, 0)}个样本 ({cluster_counts.get(i, 0)/len(clusters)*100:.1f}%)")
    
    print(f"\n  📈 性能指标:")
    print(f"    - 轮廓系数: {silhouette_avg if isinstance(silhouette_avg, str) else silhouette_avg:.4f}")
    print(f"    - 噪声点比例: {noise_ratio:.4f} (0-1之间，越小越好)")
    print(f"    - 运行时间: {end_time - start_time:.2f}秒")
    
    # 分析结果质量
    if noise_ratio > 0.5:
        print(f"\n  ⚠️ 警告: DBSCAN生成过多噪声点 ({noise_ratio*100:.1f}%)，聚类效果不佳")
        print(f"     建议尝试增加eps值或减少min_samples值")
    elif n_clusters < 2:
        print(f"\n  ⚠️ 警告: DBSCAN未能识别足够的群体，建议调整参数")
    elif n_clusters > 10:
        print(f"\n  ⚠️ 提示: DBSCAN识别了较多({n_clusters})个群体，可能需要增加min_samples")
    
    return clusters

def visualize_clusters(data, reduced_data, clusters, algorithm_name, features):
    """
    可视化聚类结果
    """
    # 创建结果目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(current_dir), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 将聚类标签添加到原始数据中
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    
    # 保存聚类结果
    result_path = os.path.join(results_dir, f"{algorithm_name}_results.csv")
    data_with_clusters.to_csv(result_path, index=False)
    
    # 1. PCA可视化
    plt.figure(figsize=(12, 10))
    
    # 创建一个colormap
    unique_clusters = np.unique(clusters)
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_clusters)))
    
    # 绘制散点图
    for i, cluster in enumerate(unique_clusters):
        if cluster == -1:  # DBSCAN的噪声点
            plt.scatter(
                reduced_data[clusters == cluster, 0],
                reduced_data[clusters == cluster, 1],
                s=30, c='black', marker='x', label=f'噪声'
            )
        else:
            plt.scatter(
                reduced_data[clusters == cluster, 0],
                reduced_data[clusters == cluster, 1],
                s=30, c=[colors[i]], label=f'群体 {cluster+1}'
            )
    
    plt.title(f'{algorithm_name} 聚类结果可视化 (PCA降维)', fontsize=15)
    plt.xlabel('主成分1', fontsize=12)
    plt.ylabel('主成分2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{algorithm_name}_pca_visualization.png"), dpi=300)
    plt.close()
    
    # 分析群体特征
    analyze_clusters(data_with_clusters, algorithm_name, features)
    
def analyze_clusters(data_with_clusters, algorithm_name, features):
    """
    分析各群体特征并生成健康管理建议
    """
    # 创建结果目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(current_dir), "results")
    
    # 计算每个群体的特征统计
    stats_path = os.path.join(results_dir, f"{algorithm_name}_cluster_statistics.csv")
    cluster_stats = data_with_clusters.groupby('Cluster').agg(['mean', 'std']).round(2)
    cluster_stats.to_csv(stats_path)
    
    # 为每个群体生成健康管理建议
    with open(os.path.join(results_dir, f"{algorithm_name}_health_recommendations.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== {algorithm_name} 聚类健康群体分析 ===\n\n")
        
        for cluster in sorted(data_with_clusters['Cluster'].unique()):
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
            
            # 跳过可能的噪声点
            if cluster == -1:
                f.write(f"群体: 噪声点 (样本数: {len(cluster_data)})\n")
                f.write("描述: 这些样本被识别为异常值，不属于任何主要健康群体\n")
                f.write("建议: 需要个体化评估\n\n")
                continue
            
            try:
                mean_age = cluster_data['RIDAGEYR'].mean()
                mean_gender = cluster_data['RIAGENDR'].mean()  # 1=男性, 2=女性
                gender_str = "男性为主" if mean_gender < 1.5 else "女性为主"
                
                # BMI和血压
                has_bmi = 'BMXBMI' in cluster_data.columns
                has_bp = 'BPXSY1' in cluster_data.columns and 'BPXDI1' in cluster_data.columns
                
                mean_bmi = cluster_data['BMXBMI'].mean() if has_bmi else "未知"
                mean_systolic = cluster_data['BPXSY1'].mean() if has_bp else "未知"
                mean_diastolic = cluster_data['BPXDI1'].mean() if has_bp else "未知"
                
                # 根据特征值识别健康群体类型
                group_type = ""
                health_issues = []
                recommendations = []
                
                # 识别年龄组
                if mean_age < 30:
                    group_type = "青年健康群体"
                    recommendations.extend([
                        "1. 建立良好的生活习惯，包括规律作息、均衡饮食",
                        "2. 定期体检，建立健康档案",
                        "3. 增加体育锻炼，增强体质",
                        "4. 注意心理健康，保持积极心态"
                    ])
                elif mean_age > 65:
                    group_type = "老年健康群体"
                    recommendations.extend([
                        "1. 定期体检，特别关注心脑血管健康",
                        "2. 适度锻炼，如散步、太极等低强度运动",
                        "3. 保持社交活动，预防认知功能下降",
                        "4. 合理用药，避免药物相互作用",
                        "5. 防跌倒措施，保障居家安全"
                    ])
                else:
                    group_type = "中年健康群体"
                    recommendations.extend([
                        "1. 定期体检，关注慢性病风险",
                        "2. 保持规律作息，避免过度劳累",
                        "3. 均衡饮食，控制热量摄入",
                        "4. 坚持锻炼，每周至少150分钟中等强度活动"
                    ])
                
                # 评估BMI
                if has_bmi:
                    bmi_status = ""
                    if isinstance(mean_bmi, (int, float)) and mean_bmi < 18.5:
                        bmi_status = "体重过轻"
                        health_issues.append("体重过轻")
                        recommendations.append("5. 增加营养摄入，适当增加体重")
                    elif isinstance(mean_bmi, (int, float)) and mean_bmi < 24:
                        bmi_status = "正常体重"
                    elif isinstance(mean_bmi, (int, float)) and mean_bmi < 28:
                        bmi_status = "超重"
                        health_issues.append("超重")
                        recommendations.append("5. 控制饮食，增加运动，维持健康体重")
                    elif isinstance(mean_bmi, (int, float)):
                        bmi_status = "肥胖"
                        health_issues.append("肥胖")
                        recommendations.extend([
                            "5. 制定健康减重计划，控制热量摄入",
                            "6. 增加体育活动，每周至少150分钟中等强度运动",
                            "7. 咨询营养师，制定个性化饮食计划"
                        ])
                    
                    if bmi_status:
                        group_type += f" - {bmi_status}"
                
                # 评估血压
                if has_bp:
                    bp_status = ""
                    if (isinstance(mean_systolic, (int, float)) and isinstance(mean_diastolic, (int, float)) and 
                        (mean_systolic >= 140 or mean_diastolic >= 90)):
                        bp_status = "高血压风险"
                        health_issues.append("高血压风险")
                        recommendations.extend([
                            "8. 限制钠盐摄入，每天摄入量控制在5g以下",
                            "9. 增加钾的摄入，多吃新鲜蔬果",
                            "10. 戒烟限酒，避免过度劳累",
                            "11. 学习血压自我监测，保持记录"
                        ])
                    elif (isinstance(mean_systolic, (int, float)) and isinstance(mean_diastolic, (int, float)) and 
                          (mean_systolic <= 90 or mean_diastolic <= 60)):
                        bp_status = "低血压风险"
                        health_issues.append("低血压风险")
                        recommendations.extend([
                            "8. 避免突然站立，预防体位性低血压",
                            "9. 保持水分摄入，防止脱水",
                            "10. 适当增加盐分摄入（在医生指导下）"
                        ])
                    
                    if bp_status:
                        group_type += f" - {bp_status}"
                
                # 输出到文件
                f.write(f"群体 {cluster+1}: {group_type} (样本数: {len(cluster_data)})\n\n")
                f.write("特征概况:\n")
                f.write(f"- 年龄均值: {mean_age:.1f} 岁\n")
                f.write(f"- 性别分布: {gender_str}\n")
                
                if has_bmi:
                    f.write(f"- BMI均值: {mean_bmi:.1f}\n")
                
                if has_bp:
                    f.write(f"- 收缩压均值: {mean_systolic:.1f} mmHg\n")
                    f.write(f"- 舒张压均值: {mean_diastolic:.1f} mmHg\n")
                
                f.write("\n潜在健康问题:\n")
                if health_issues:
                    for issue in health_issues:
                        f.write(f"- {issue}\n")
                else:
                    f.write("- 无明显健康风险\n")
                
                f.write("\n健康管理建议:\n")
                for rec in recommendations:
                    f.write(f"{rec}\n")
                f.write("\n" + "=" * 40 + "\n\n")
                
            except Exception as e:
                f.write(f"群体 {cluster+1}: 数据不足以进行详细分析\n")
                f.write(f"错误: {str(e)}\n\n")
    
    print(f"健康群体分析和建议已保存到: {os.path.join(results_dir, f'{algorithm_name}_health_recommendations.txt')}")

def main():
    # 加载数据
    df = load_data(sample_size=5000)
    
    # 预处理数据
    data, scaled_data, features = preprocess_data(df)
    
    # 降维用于可视化
    reduced_data = reduce_dimensions(scaled_data)
    
    # 1. K均值聚类
    kmeans_clusters = kmeans_clustering(scaled_data)
    visualize_clusters(data, reduced_data, kmeans_clusters, "kmeans", features)
    
    # 2. 层次聚类
    hierarchical_clusters = hierarchical_clustering(scaled_data)
    visualize_clusters(data, reduced_data, hierarchical_clusters, "hierarchical", features)
    
    # 3. DBSCAN聚类 - 自动调整参数
    # DBSCAN参数调整为更适合的值
    eps = 0.8  # 增加邻域半径
    min_samples = 15  # 减少最小样本数
    dbscan_clusters = dbscan_clustering(scaled_data, eps, min_samples)
    
    # 如果DBSCAN聚类结果不好（大部分为噪声点），尝试再次调整参数
    noise_ratio = np.sum(dbscan_clusters == -1) / len(dbscan_clusters)
    if noise_ratio > 0.5:  # 如果超过50%是噪声点
        print("DBSCAN生成过多噪声点，尝试调整参数...")
        eps = 1.0  # 进一步增加邻域半径
        min_samples = 10  # 进一步减少最小样本数
        dbscan_clusters = dbscan_clustering(scaled_data, eps, min_samples)
    
    visualize_clusters(data, reduced_data, dbscan_clusters, "dbscan", features)
    
    # 4. 谱聚类
    try:
        print(f"\n🔍 执行谱聚类分析 " + "="*48)
        print(f"  · 算法特点: 基于图论和流形学习的聚类方法，能够发现非凸形状的簇")
        print(f"  · 优势: 能处理复杂形状的簇，对噪声相对鲁棒")
        print(f"  · 劣势: 计算复杂度高，大数据集上性能较差")
        print(f"  · 实现方法: 使用最近邻图构建相似度矩阵，降低计算复杂度")
        
        print(f"\n  开始训练谱聚类模型...")
        spectral_clusters = spectral_clustering(scaled_data)
        
        # 统计每个簇的样本数
        unique, counts = np.unique(spectral_clusters, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        
        # 输出结果统计
        print(f"\n  📊 聚类结果:")
        print(f"    - 总样本数: {len(spectral_clusters)}")
        print(f"    - 聚类数量: {len(unique)}")
        
        for i in range(len(unique)):
            print(f"    - 群体 {i+1}: {cluster_counts.get(i, 0)}个样本 " + 
                  f"({cluster_counts.get(i, 0)/len(spectral_clusters)*100:.1f}%)")
        
        # 可视化结果
        visualize_clusters(data, reduced_data, spectral_clusters, "spectral", features)
        print(f"\n  ✅ 谱聚类执行成功并生成可视化结果")
    except Exception as e:
        print(f"\n  ❌ 谱聚类执行失败: {str(e)}")
        print(f"     可能原因: 内存不足或数据集过大，谱聚类的计算复杂度较高")
    
    print("\n所有聚类算法执行完毕，结果已保存到results目录")

if __name__ == "__main__":
    main() 