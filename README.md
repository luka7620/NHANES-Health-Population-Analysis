# NHANES 医学数据仓库与数据挖掘系统

本项目用于“数据仓库与数据挖掘”课程大作业，医学场景为：基于 NHANES 公开脱敏健康数据，构建心代谢风险数据仓库，完成聚合分析、Apriori 关联规则挖掘和决策树分类预测，并通过 Streamlit 系统展示结果。

## 作业要求覆盖

| 要求 | 项目实现 |
| --- | --- |
| 医学领域数据 | NHANES 健康、体检、实验室、问卷、用药公开脱敏数据 |
| 数据导入 | Streamlit 上传 CSV 或默认读取 `data/nhanes_processed.csv` |
| 数据预处理 | 按 `SEQN` 汇总受试者，清洗未知编码和异常值，构造 BMI、血压、糖尿病、胆固醇、饮食、活动和用药特征 |
| 数据仓库 | SQLite：ODS 原始层、DWD 清洗宽表、ADS 聚合表、DM 挖掘结果表 |
| 聚合分析 | 按年龄、性别、BMI、风险等级统计高风险率、糖尿病率、高血压率、高胆固醇率等 |
| 关联规则 | Apriori 挖掘生活方式、检查指标、疾病风险、用药标签之间的规则 |
| 分类算法 | 决策树预测心代谢高风险人群，输出 Accuracy、Precision、Recall、F1、混淆矩阵、特征重要性 |
| 系统展示 | `streamlit run src/app.py` 启动完整 Web 系统 |
| 报告材料 | 自动生成 HTML、Markdown、Word 报告和数据仓库设计文档 |

## 环境安装

```bash
pip install -r requirements.txt
```

## 启动 Streamlit 系统

```bash
streamlit run src/app.py
```

系统页面包含：

- 数据导入
- 预处理与数据仓库
- 聚合分析
- 关联规则
- 分类预测
- 结果导出
- 系统说明

## 命令行一键复现

```bash
python src/main.py
```

可选参数：

```bash
python src/main.py --min-support 0.08 --min-confidence 0.55 --max-depth 5
```

## 主要输出

运行后生成：

- `results/nhanes_health_subjects.csv`：预处理后的受试者级数据
- `results/nhanes_health_warehouse.db`：SQLite 数据仓库
- `results/aggregation_by_*.csv`：聚合分析结果
- `results/association_rules.csv`：Apriori 关联规则
- `results/classification_metrics.csv`：分类模型评价指标
- `results/classification_confusion_matrix.png`：混淆矩阵
- `results/classification_feature_importance.png`：特征重要性
- `results/health_mining_report.html`：自动 HTML 报告
- `docs/data_warehouse_design.md`：数据仓库设计文档
- `docs/course_report.md`、`docs/course_report.docx`、`docs/course_report.pdf`：课程大作业报告

## 项目结构

```text
data/                         数据集
docs/                         课程报告和数据仓库设计文档
results/                      算法结果、图表、SQLite 仓库
src/app.py                    Streamlit 系统界面
src/main.py                   命令行入口
src/health_mining_pipeline.py 数据预处理、仓库、聚合、关联规则、分类、报告生成
```

## 数据说明

数据来源为 CDC NHANES 公开数据。仓库中的合并数据使用 `SEQN` 作为脱敏受试者编号，不包含姓名、电话、地址等直接身份信息。系统默认分析成人受试者，并围绕心代谢风险进行特征工程和数据挖掘。

## 原有聚类模块

仓库中仍保留原有聚类分析代码和结果文件，包括 K-means、层次聚类、DBSCAN、谱聚类等；本次课程作业的默认入口已切换为聚合分析、关联规则和分类预测这三类硬性要求。
