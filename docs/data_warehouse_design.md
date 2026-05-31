# NHANES 心代谢风险数据仓库设计

## 分层设计

| 层级 | 表名 | 说明 |
| --- | --- | --- |
| ODS | `ods_nhanes_subject_raw` | 按 SEQN 汇总后的原始受试者记录，保留人口学、体检、实验室、问卷、用药字段。 |
| DWD | `dwd_health_subjects` | 清洗、规约和医学特征工程后的受试者宽表。 |
| ADS | `ads_aggregation_by_age_group` | 按年龄分组的健康风险聚合指标。 |
| ADS | `ads_aggregation_by_gender` | 按性别分组的健康风险聚合指标。 |
| ADS | `ads_aggregation_by_bmi_group` | 按 BMI 分层的健康风险聚合指标。 |
| DM | `dm_association_rules` | Apriori 关联规则结果。 |
| DM | `dm_classification_metrics` | 决策树分类评价指标。 |
| DM | `dm_classification_feature_importance` | 决策树特征重要性。 |

## DWD 核心字段

| 字段 | 含义 |
| --- | --- |
| `SEQN` | NHANES 脱敏受试者编号 |
| `age` | 年龄 |
| `gender` | 性别 |
| `age_group` | 年龄分组 |
| `bmi` | 体质指数 |
| `bmi_group` | BMI 分层 |
| `systolic_bp` | 平均收缩压 |
| `diastolic_bp` | 平均舒张压 |
| `hba1c` | 糖化血红蛋白 |
| `glucose` | 清洗或派生后的分析字段 |
| `total_cholesterol` | 总胆固醇 |
| `hdl_cholesterol` | HDL 胆固醇 |
| `triglycerides` | 清洗或派生后的分析字段 |
| `calories` | 清洗或派生后的分析字段 |
| `sodium_mg` | 清洗或派生后的分析字段 |
| `sugar_g` | 清洗或派生后的分析字段 |
| `fiber_g` | 清洗或派生后的分析字段 |
| `sedentary_minutes` | 清洗或派生后的分析字段 |
| `income_poverty_ratio` | 清洗或派生后的分析字段 |
| `diabetes_self_report` | 清洗或派生后的分析字段 |
| `hypertension_self_report` | 清洗或派生后的分析字段 |
| `cholesterol_self_report` | 清洗或派生后的分析字段 |
| `ever_smoker` | 清洗或派生后的分析字段 |
| `regular_alcohol` | 清洗或派生后的分析字段 |
| `vigorous_activity` | 清洗或派生后的分析字段 |
| `moderate_activity` | 清洗或派生后的分析字段 |
| `waist_high` | 清洗或派生后的分析字段 |
| `obesity` | 清洗或派生后的分析字段 |
| `hypertension` | 是否高血压风险 |
| `diabetes` | 是否糖尿病风险 |
| `prediabetes` | 清洗或派生后的分析字段 |
| `high_cholesterol` | 是否高胆固醇风险 |
| `high_sodium` | 清洗或派生后的分析字段 |
| `high_sugar` | 清洗或派生后的分析字段 |
| `low_fiber` | 清洗或派生后的分析字段 |
| `sedentary_high` | 清洗或派生后的分析字段 |
| `older_adult` | 清洗或派生后的分析字段 |
| `rx_insulin` | 清洗或派生后的分析字段 |
| `rx_metformin` | 清洗或派生后的分析字段 |
| `rx_statin` | 清洗或派生后的分析字段 |
| `rx_antihypertensive` | 清洗或派生后的分析字段 |
| `drug_count` | 清洗或派生后的分析字段 |
| `medications` | 清洗或派生后的分析字段 |
| `medication_indications` | 清洗或派生后的分析字段 |
| `risk_score` | 心代谢风险因素计数 |
| `risk_level` | 心代谢风险分层 |
| `high_cardiometabolic_risk` | 分类目标：是否高心代谢风险 |

## 数据流

1. ODS：读取 `data/nhanes_processed.csv`，按 `SEQN` 汇总重复的用药记录。
2. DWD：清洗未知编码、异常范围和缺失值，生成 BMI、血压、糖尿病、胆固醇、饮食与用药特征。
3. ADS：按年龄、性别、BMI 和风险等级进行聚合分析。
4. DM：执行 Apriori 关联规则和决策树分类，结果写回 SQLite 与 CSV。