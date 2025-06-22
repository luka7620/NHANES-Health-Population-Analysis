# NHANES健康群体聚类分析与个性化健康管理系统

## 项目简介

本项目是一个基于NHANES（美国国家健康与营养调查）数据的健康群体聚类分析系统。通过应用多种机器学习聚类算法，对健康数据进行分析，从而实现人群健康特征分类和个性化健康管理建议生成。

## 功能特点

- **多种聚类算法**：实现了K-means、层次聚类(Hierarchical)、DBSCAN和谱聚类(Spectral)多种算法
- **数据可视化**：生成直观的聚类结果可视化图表，帮助理解不同健康群体的特征
- **个性化健康建议**：基于聚类结果为不同健康群体生成定制化健康管理建议
- **综合分析报告**：自动生成HTML格式的交互式分析报告，便于查阅和分享研究结果

## 系统要求

- Python 3.8或更高版本
- 足够的内存处理大规模数据集(建议8GB以上)
- 支持Windows、macOS和Linux系统

## 安装说明

1. 克隆项目到本地

```bash
git clone https://github.com/yourusername/nhanes-clustering.git
cd nhanes-clustering
```

2. 创建并激活虚拟环境(建议)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

3. 安装依赖包

```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保数据文件位于`data`目录下
2. 运行主程序

```bash
python src/main.py
```

3. 分析完成后，系统将在`results`目录中生成以下文件：
   - CSV格式的聚类结果
   - 聚类可视化图像
   - 健康管理建议文本
   - 综合HTML分析报告

4. 使用浏览器打开`results/clustering_report.html`查看完整分析报告

## 项目结构

```
.
├── data/                # NHANES数据集
├── results/             # 分析结果和报告
├── src/                 # 源代码
│   ├── main.py          # 主程序入口
│   ├── extended_clustering.py    # 聚类算法实现
│   ├── clustering_analysis.py    # 聚类分析逻辑
│   ├── data_exploration.py       # 数据探索模块
│   ├── spectral_clustering.py    # 谱聚类实现
│   └── generate_report.py        # 报告生成模块
├── .venv/               # Python虚拟环境(可选)
└── README.md            # 项目说明文档
```

## 注意事项

- 首次运行时间可能较长，取决于数据集大小和计算机性能
- 确保有足够的磁盘空间存储结果文件，尤其是可视化图表
- 建议在虚拟环境中运行，避免依赖包冲突

## 扩展与定制

- 可在`src/extended_clustering.py`中添加新的聚类算法
- 可在`src/generate_report.py`中自定义报告格式和内容
- 可根据需要调整聚类参数以适应不同数据集特点

## 联系与支持

如有问题或建议，请通过以下方式联系：

- 提交Issues: [https://github.com/yourusername/nhanes-clustering/issues](https://github.com/yourusername/nhanes-clustering/issues)
- 邮件联系: your.email@example.com

## 许可证

本项目采用MIT许可证，详情请参阅LICENSE文件。 