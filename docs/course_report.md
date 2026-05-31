# 基于 NHANES 数据的心代谢风险分层、关联规则与分类预测系统

## 摘要

本项目围绕医学健康管理场景，使用 NHANES 公开脱敏数据，构建心代谢风险数据仓库，实现数据预处理、聚合分析、Apriori 关联规则挖掘和决策树分类预测，并通过 Streamlit 系统展示结果。

## 数据集与场景

- 数据来源：CDC NHANES 公开调查数据，仓库内已提供合并后的 `data/nhanes_processed.csv`。
- 分析对象：清洗后成人受试者 6,113 条。
- 医学场景：识别具有肥胖、高血压、糖尿病、高胆固醇、不良生活方式等因素的心代谢高风险人群。

## 数据预处理与数据仓库

- 按 `SEQN` 将多行用药记录规约为受试者级宽表。
- 将 7、9、7777、9999 等 NHANES 未知或拒答编码转为空值。
- 按医学合理范围剔除异常值，并生成年龄组、BMI 分层、血压、糖化血红蛋白、胆固醇、饮食、活动和用药标签。
- SQLite 数据仓库路径：`results/nhanes_health_warehouse.db`。

## 聚合分析

| age_group | participants | mean_age | mean_bmi | mean_systolic_bp | mean_hba1c | mean_total_cholesterol | high_risk_rate | diabetes_rate | hypertension_rate | high_cholesterol_rate | obesity_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18-29 | 1300.0000 | 22.7346 | 27.2027 | 113.6036 | 5.2452 | 169.8885 | 0.2177 | 0.0262 | 0.1562 | 0.4346 | 0.2615 |
| 30-44 | 1547.0000 | 37.0853 | 29.4526 | 117.2592 | 5.5081 | 190.5947 | 0.3904 | 0.0905 | 0.3400 | 0.5701 | 0.3911 |
| 45-59 | 1425.0000 | 51.8211 | 29.6671 | 123.1523 | 5.9016 | 198.0849 | 0.5439 | 0.1986 | 0.5754 | 0.7025 | 0.3937 |
| 60+ | 1841.0000 | 70.0141 | 28.8981 | 131.8001 | 6.0430 | 186.8756 | 0.8452 | 0.3265 | 0.7865 | 0.7463 | 0.3509 |

| bmi_group | participants | mean_age | mean_bmi | mean_systolic_bp | mean_hba1c | mean_total_cholesterol | high_risk_rate | diabetes_rate | hypertension_rate | high_cholesterol_rate | obesity_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Normal | 1761.0000 | 44.0301 | 22.3790 | 119.2667 | 5.4382 | 181.5395 | 0.2334 | 0.0852 | 0.3390 | 0.4696 | 0.0000 |
| Obese | 2120.0000 | 48.5311 | 36.2875 | 125.0223 | 5.9530 | 190.0575 | 0.8491 | 0.2557 | 0.6283 | 0.7561 | 1.0000 |
| Overweight | 2110.0000 | 49.4133 | 27.4571 | 122.1024 | 5.6989 | 188.4517 | 0.4635 | 0.1716 | 0.4872 | 0.6365 | 0.0152 |
| Underweight | 122.0000 | 40.8361 | 17.4607 | 118.9235 | 5.3533 | 178.4262 | 0.2377 | 0.0328 | 0.3279 | 0.4016 | 0.0000 |

## 关联规则算法

- Apriori 默认参数：最小支持度 0.08，最小置信度 0.55，最大项集长度 3。
- 规则重点筛选高风险、糖尿病、高血压、高胆固醇和典型用药相关后件。

| antecedent | consequent | support | confidence | lift | antecedent_support | consequent_support |
| --- | --- | --- | --- | --- | --- | --- |
| Age=60+ + BMI=Obese | Risk=High | 0.1029 | 0.9890 | 1.8787 | 0.1040 | 0.5264 |
| BMI=Obese + Diabetes=Yes | Risk=High | 0.0872 | 0.9834 | 1.8681 | 0.0887 | 0.5264 |
| Rx=Antihypertensive + Rx=Statin | Hypertension=Yes | 0.1251 | 0.9096 | 1.8554 | 0.1376 | 0.4903 |
| Age=60+ + Diabetes=Yes | Risk=High | 0.0955 | 0.9717 | 1.8459 | 0.0983 | 0.5264 |
| BMI=Obese + Diabetes=Prediabetes | Risk=High | 0.0824 | 0.9674 | 1.8376 | 0.0852 | 0.5264 |
| BMI=Obese + Hypertension=Yes | Risk=High | 0.2099 | 0.9632 | 1.8297 | 0.2179 | 0.5264 |
| Diabetes=Yes + Smoking=Ever | Risk=High | 0.0803 | 0.9609 | 1.8253 | 0.0836 | 0.5264 |
| Diabetes=Yes + Hypertension=Yes | Risk=High | 0.1229 | 0.9591 | 1.8220 | 0.1281 | 0.5264 |
| Activity=SedentaryHigh + BMI=Obese | Risk=High | 0.0872 | 0.9586 | 1.8210 | 0.0910 | 0.5264 |
| Diabetes=Yes + Diet=FiberLow | Risk=High | 0.0906 | 0.9535 | 1.8113 | 0.0950 | 0.5264 |

## 分类算法

- 算法：决策树（信息增益/熵准则），训练集与测试集比例 7:3。
- Accuracy：0.850
- Precision：0.868
- Recall：0.842
- F1：0.855

### 重要特征

| feature | importance |
| --- | --- |
| hba1c | 0.3147 |
| bmi | 0.2535 |
| age_group_60+ | 0.1269 |
| ever_smoker | 0.1014 |
| sedentary_minutes | 0.0604 |
| age | 0.0547 |
| fiber_g | 0.0368 |
| systolic_bp | 0.0238 |
| sodium_mg | 0.0097 |
| triglycerides | 0.0095 |

## 系统开发

- 命令行入口：`python src/main.py`。
- Streamlit 入口：`streamlit run src/app.py`。
- 系统模块：数据导入、数据预处理、数据仓库、聚合分析、关联规则、分类预测、结果导出。

## 测试用例

| 测试项 | 操作 | 预期结果 |
| --- | --- | --- |
| 数据导入 | 运行默认数据或上传 CSV | 展示样本量和字段数 |
| 预处理 | 点击运行分析 | 生成受试者级清洗数据 |
| 数据仓库 | 查看结果文件 | SQLite 数据库包含 ODS/DWD/ADS/DM 表 |
| 聚合分析 | 查看聚合页面 | 年龄、性别、BMI 分组指标可视化 |
| 关联规则 | 调整支持度/置信度并运行 | 生成规则表，包含支持度、置信度、提升度 |
| 分类预测 | 查看分类页面 | 输出准确率、召回率、F1 和混淆矩阵 |

## 总结与展望

项目完成了医学数据从导入、清洗、仓库分层、算法挖掘到系统展示的闭环。后续可进一步引入抽样权重、更多 NHANES 周期数据和临床验证规则，提高医学解释性。

## 参考文献

- CDC National Health and Nutrition Examination Survey (NHANES).
- Han, J., Kamber, M., Pei, J. Data Mining: Concepts and Techniques.
- scikit-learn documentation: DecisionTreeClassifier.
- Streamlit documentation.