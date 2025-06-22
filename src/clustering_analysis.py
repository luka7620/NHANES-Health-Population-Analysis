import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_data():
    """
    加载NHANES处理后的数据集
    """
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取数据文件路径
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    data_path = os.path.join(data_dir, "nhanes_processed.csv")
    
    print("正在加载数据...")
    df = pd.read_csv(data_path)
    print(f"数据加载完成，形状: {df.shape}")
    return df


def preprocess_data(df):
    """
    预处理数据：选择特征、处理缺失值、标准化
    """
    print("开始数据预处理...")
    
    # 选择有用的特征列 (年龄、性别、BMI、血压、血糖、胆固醇等)
    # 定义关键健康指标
    # 年龄、性别必选
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
    
    # 可选的血糖和胆固醇指标 - 使用存在的列
    optional_glucose = ['LBXGLU', 'LBXGH', 'LBXGLT']  # 不同形式的血糖指标
    optional_cholesterol = ['LBXHDL', 'LBDLDL']  # 不同形式的胆固醇指标
    
    # 确保必选特征存在
    for feature in must_have_features:
        if feature not in df.columns:
            raise ValueError(f"必需的特征 {feature} 不存在于数据集中")
    
    # 添加可用的健康指标
    available_features = must_have_features.copy()
    available_features.extend([col for col in health_indicators if col in df.columns])
    
    # 尝试添加至少一个血糖指标和一个胆固醇指标
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
    
    # 去除含有缺失值的行 (或者可以进行插补)
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
    print(f"正在进行K均值聚类 (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(scaled_data)
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(scaled_data, clusters)
    print(f"K均值聚类轮廓系数: {silhouette_avg:.4f}")
    
    return clusters


def hierarchical_clustering(scaled_data, n_clusters=5):
    """
    层次聚类
    """
    print(f"正在进行层次聚类 (k={n_clusters})...")
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = hierarchical.fit_predict(scaled_data)
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(scaled_data, clusters)
    print(f"层次聚类轮廓系数: {silhouette_avg:.4f}")
    
    return clusters


def dbscan_clustering(scaled_data):
    """
    DBSCAN聚类
    """
    print("正在进行DBSCAN聚类...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    
    # 计算有效标签的轮廓系数（排除噪声点，标签为-1）
    if len(np.unique(clusters)) > 1 and -1 not in np.unique(clusters):
        silhouette_avg = silhouette_score(scaled_data, clusters)
        print(f"DBSCAN聚类轮廓系数: {silhouette_avg:.4f}")
    else:
        print("DBSCAN没有找到有效的聚类或者只有噪声点")
    
    return clusters


def optimized_spectral_clustering(scaled_data, n_clusters=5):
    """
    优化的谱聚类 - 使用近邻图和降低的样本量来提高性能
    """
    print(f"正在进行优化的谱聚类 (k={n_clusters})...")
    
    # 如果数据量很大，随机抽样减小计算量
    if len(scaled_data) > 10000:
        print("数据量较大，进行随机抽样...")
        sample_size = min(10000, int(len(scaled_data) * 0.3))
        indices = np.random.choice(len(scaled_data), sample_size, replace=False)
        sampled_data = scaled_data[indices]
    else:
        sampled_data = scaled_data
        indices = np.arange(len(scaled_data))
    
    start_time = time.time()
    
    # 使用优化参数的谱聚类
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels="discretize",
        random_state=42,
        affinity="nearest_neighbors",  # 使用近邻图而不是全连接图
        n_neighbors=10,  # 近邻数
        eigen_solver='arpack'  # 更高效的特征值求解器
    )
    
    # 对抽样数据进行聚类
    sample_clusters = spectral.fit_predict(sampled_data)
    
    # 如果使用了抽样，需要将标签扩展回原始数据集
    if len(scaled_data) > 10000:
        # 训练一个KNN分类器来预测其余数据点的簇
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(sampled_data, sample_clusters)
        
        # 预测所有数据点
        full_clusters = np.zeros(len(scaled_data), dtype=int)
        full_clusters[indices] = sample_clusters
        
        # 预测未抽样的数据点
        mask = np.ones(len(scaled_data), dtype=bool)
        mask[indices] = False
        if np.any(mask):
            full_clusters[mask] = knn.predict(scaled_data[mask])
        
        clusters = full_clusters
    else:
        clusters = sample_clusters
    
    end_time = time.time()
    print(f"谱聚类用时: {end_time - start_time:.2f}秒")
    
    # 计算轮廓系数
    if len(scaled_data) <= 10000:  # 仅当数据量适中时计算
        silhouette_avg = silhouette_score(scaled_data, clusters)
        print(f"谱聚类轮廓系数: {silhouette_avg:.4f}")
    
    return clusters


def visualize_clusters(data, reduced_data, clusters, cluster_name, features):
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
    result_path = os.path.join(results_dir, f"{cluster_name}_results.csv")
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
    
    plt.title(f'{cluster_name} 聚类结果可视化 (PCA降维)', fontsize=15)
    plt.xlabel('主成分1', fontsize=12)
    plt.ylabel('主成分2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{cluster_name}_pca_visualization.png"), dpi=300)
    plt.close()
    
    # 2. 每个群体的特征分布
    plt.figure(figsize=(15, 10))
    
    # 计算每个群体的特征均值
    cluster_means = data_with_clusters.groupby('Cluster').mean()
    
    # 为每个特征创建子图
    for i, feature in enumerate(features):
        plt.subplot(3, 4, i+1)
        for j, cluster in enumerate(cluster_means.index):
            if cluster == -1:
                label = '噪声'
            else:
                label = f'群体 {cluster+1}'
            plt.bar(
                x=j, 
                height=cluster_means.loc[cluster, feature],
                width=0.6,
                color=colors[j % len(colors)],
                label=label if i == 0 else ""
            )
        plt.title(feature)
        plt.xticks(np.arange(len(cluster_means)), 
                   [f'{c+1}' if c != -1 else '噪声' for c in cluster_means.index])
    
    plt.suptitle(f"{cluster_name} - 各群体特征均值分布", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(results_dir, f"{cluster_name}_feature_distribution.png"), dpi=300)
    plt.close()


def analyze_health_groups(data, clusters, cluster_name):
    """
    分析各健康群体的特征和提供健康管理建议
    """
    # 创建结果目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(current_dir), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    
    # 计算每个群体的特征统计
    cluster_stats = data_with_clusters.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])
    
    # 保存群体统计结果
    stats_path = os.path.join(results_dir, f"{cluster_name}_cluster_statistics.csv")
    cluster_stats.to_csv(stats_path)
    
    # 根据特征统计识别不同健康群体并生成管理建议
    health_groups = []
    recommendations = []
    
    # 群体数量
    n_clusters = len(data_with_clusters['Cluster'].unique())
    
    # 创建描述和建议的文本文件
    with open(os.path.join(results_dir, f"{cluster_name}_health_recommendations.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== {cluster_name} 聚类健康群体分析 ===\n\n")
        
        for cluster in sorted(data_with_clusters['Cluster'].unique()):
            # 跳过可能的噪声点
            if cluster == -1:
                f.write("群体: 噪声点\n")
                f.write("描述: 这些样本被识别为异常值，不属于任何主要健康群体\n")
                f.write("建议: 需要个体化评估\n\n")
                continue
                
            # 获取当前群体的数据
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
            
            # 提取关键特征均值
            try:
                mean_age = cluster_data['RIDAGEYR'].mean()
                mean_bmi = cluster_data['BMXBMI'].mean() if 'BMXBMI' in cluster_data.columns else "未知"
                mean_systolic = cluster_data['BPXSY1'].mean() if 'BPXSY1' in cluster_data.columns else "未知"
                mean_diastolic = cluster_data['BPXDI1'].mean() if 'BPXDI1' in cluster_data.columns else "未知"
                
                # 根据特征值识别健康群体类型
                group_type = ""
                recommendation = ""
                
                # 年龄分组判断
                if mean_age < 30:
                    group_type = "青年健康群体"
                    recommendation = (
                        "1. 建立良好的生活习惯，包括规律作息、均衡饮食\n"
                        "2. 定期体检，建立健康档案\n"
                        "3. 增加体育锻炼，增强体质\n"
                        "4. 培养健康的社交关系，减轻心理压力"
                    )
                elif mean_age > 65:
                    group_type = "老年健康群体"
                    recommendation = (
                        "1. 定期体检，特别关注心脑血管健康\n"
                        "2. 适度锻炼，如散步、太极等低强度运动\n"
                        "3. 保持社交活动，预防认知功能下降\n"
                        "4. 合理用药，避免药物相互作用\n"
                        "5. 防跌倒措施，保障居家安全"
                    )
                
                # BMI判断
                if mean_bmi != "未知":
                    if mean_bmi > 30:
                        group_type += " - 肥胖健康群体" if group_type else "肥胖健康群体"
                        recommendation += (
                            "\n6. 制定健康减重计划，控制热量摄入\n"
                            "7. 增加体育活动，每周至少150分钟中等强度运动\n"
                            "8. 咨询营养师，制定个性化饮食计划\n"
                            "9. 定期监测血压、血糖和血脂"
                        )
                
                # 血压判断
                if mean_systolic != "未知" and mean_diastolic != "未知":
                    if mean_systolic >= 140 or mean_diastolic >= 90:
                        group_type += " - 高血压风险群体" if group_type else "高血压风险群体"
                        recommendation += (
                            "\n10. 限制钠盐摄入，每天摄入量控制在5g以下\n"
                            "11. 增加钾的摄入，多吃新鲜蔬果\n"
                            "12. 戒烟限酒，避免过度劳累\n"
                            "13. 学习血压自我监测，保持记录\n"
                            "14. 遵医嘱服药，不自行调整药物"
                        )
                
                # 如果没有明确特征，可能是一般健康群体
                if not group_type:
                    group_type = f"普通健康群体 {cluster+1}"
                    recommendation = (
                        "1. 保持健康生活方式，均衡饮食\n"
                        "2. 规律运动，每周至少150分钟中等强度活动\n"
                        "3. 保证充足睡眠，减轻压力\n"
                        "4. 定期体检，预防疾病"
                    )
                
                # 输出到文件
                f.write(f"群体 {cluster+1}: {group_type}\n")
                f.write(f"年龄均值: {mean_age:.1f} 岁\n")
                f.write(f"BMI均值: {mean_bmi}\n")
                f.write(f"收缩压均值: {mean_systolic}\n")
                f.write(f"舒张压均值: {mean_diastolic}\n")
                f.write("\n健康管理建议:\n")
                f.write(recommendation)
                f.write("\n\n")
                
            except Exception as e:
                f.write(f"群体 {cluster+1}: 数据不足以进行详细分析\n")
                f.write(f"错误: {str(e)}\n\n")
    
    print(f"健康群体分析和建议已保存到: {os.path.join(results_dir, f'{cluster_name}_health_recommendations.txt')}")


def main():
    # 加载数据
    df = load_data()
    
    # 预处理数据
    data, scaled_data, features = preprocess_data(df)
    
    # 降维用于可视化
    reduced_data = reduce_dimensions(scaled_data)
    
    # 1. K均值聚类
    kmeans_clusters = kmeans_clustering(scaled_data)
    visualize_clusters(data, reduced_data, kmeans_clusters, "kmeans", features)
    analyze_health_groups(data, kmeans_clusters, "kmeans")
    
    # 2. 层次聚类
    hierarchical_clusters = hierarchical_clustering(scaled_data)
    visualize_clusters(data, reduced_data, hierarchical_clusters, "hierarchical", features)
    analyze_health_groups(data, hierarchical_clusters, "hierarchical")
    
    # 3. DBSCAN聚类
    dbscan_clusters = dbscan_clustering(scaled_data)
    visualize_clusters(data, reduced_data, dbscan_clusters, "dbscan", features)
    analyze_health_groups(data, dbscan_clusters, "dbscan")
    
    # 4. 优化的谱聚类
    spectral_clusters = optimized_spectral_clustering(scaled_data)
    visualize_clusters(data, reduced_data, spectral_clusters, "spectral", features)
    analyze_health_groups(data, spectral_clusters, "spectral")
    
    print("所有聚类算法执行完毕，结果已保存到results目录")


if __name__ == "__main__":
    main() 