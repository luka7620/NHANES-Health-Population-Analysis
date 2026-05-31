import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from health_mining_pipeline import (
    RESULTS_DIR,
    add_health_features,
    aggregate_subject_level,
    apriori_rules,
    build_aggregation_tables,
    load_raw_data,
    run_classification,
    run_pipeline,
    transaction_from_row,
)


st.set_page_config(
    page_title="NHANES Medical Data Mining System",
    page_icon=None,
    layout="wide",
)


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def show_dataframe(df: pd.DataFrame, height: int = 360) -> None:
    if df.empty:
        st.info("暂无数据，请先运行分析。")
    else:
        st.dataframe(df, use_container_width=True, height=height)


def metric_percent(label: str, value: float) -> None:
    st.metric(label, f"{value:.1%}" if pd.notna(value) else "N/A")


def prepare_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".csv"
    temp_dir = Path(tempfile.gettempdir()) / "nhanes_health_mining"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"uploaded_dataset{suffix}"
    temp_path.write_bytes(uploaded_file.getvalue())
    return temp_path


def current_output_paths() -> dict:
    return {
        "processed": RESULTS_DIR / "nhanes_health_subjects.csv",
        "rules": RESULTS_DIR / "association_rules.csv",
        "metrics": RESULTS_DIR / "classification_metrics.csv",
        "confusion": RESULTS_DIR / "classification_confusion_matrix.csv",
        "importance": RESULTS_DIR / "classification_feature_importance.csv",
        "warehouse": RESULTS_DIR / "nhanes_health_warehouse.db",
        "html": RESULTS_DIR / "health_mining_report.html",
        "docx": Path(__file__).resolve().parents[1] / "docs" / "course_report.docx",
        "pdf": Path(__file__).resolve().parents[1] / "docs" / "course_report.pdf",
        "md": Path(__file__).resolve().parents[1] / "docs" / "course_report.md",
    }


def sidebar_controls():
    st.sidebar.header("运行参数")
    uploaded = st.sidebar.file_uploader("上传 NHANES 合并 CSV", type=["csv"])
    data_path = None
    if uploaded is not None:
        data_path = prepare_uploaded_file(uploaded)
        st.sidebar.success(f"已加载上传文件：{uploaded.name}")
    else:
        st.sidebar.caption("未上传时使用 data/nhanes_processed.csv")

    min_support = st.sidebar.slider("Apriori 最小支持度", 0.01, 0.30, 0.08, 0.01)
    min_confidence = st.sidebar.slider("Apriori 最小置信度", 0.30, 0.95, 0.55, 0.05)
    max_depth = st.sidebar.slider("决策树最大深度", 2, 10, 5, 1)
    nrows = st.sidebar.number_input("快速测试行数（0 表示全量）", min_value=0, value=0, step=1000)
    run_clicked = st.sidebar.button("运行完整分析", type="primary", use_container_width=True)
    return data_path, min_support, min_confidence, max_depth, int(nrows) or None, run_clicked


def run_analysis(data_path: Path, min_support: float, min_confidence: float, max_depth: int, nrows):
    with st.spinner("正在执行数据预处理、数据仓库写入、聚合分析、关联规则和分类模型..."):
        artifacts = run_pipeline(
            data_path=data_path,
            min_support=min_support,
            min_confidence=min_confidence,
            max_depth=max_depth,
            nrows=nrows,
        )
    st.session_state["last_summary"] = artifacts.summary
    st.success("分析完成，结果已写入 results 和 docs 目录。")


def display_import_module(data_path: Path, nrows):
    st.subheader("数据导入")
    raw = load_raw_data(data_path=data_path, nrows=nrows)
    col1, col2, col3 = st.columns(3)
    col1.metric("原始行数", f"{len(raw):,}")
    col2.metric("字段数", f"{raw.shape[1]:,}")
    col3.metric("受试者数", f"{raw['SEQN'].nunique():,}" if "SEQN" in raw.columns else "N/A")
    st.caption("原始数据来自 NHANES 公开脱敏数据。系统按 SEQN 汇总重复用药记录，构建受试者级医学数据宽表。")
    show_dataframe(raw.head(200), height=420)


def display_preprocess_module(data_path: Path, nrows):
    st.subheader("数据预处理与数据仓库")
    raw = load_raw_data(data_path=data_path, nrows=nrows)
    subject = aggregate_subject_level(raw)
    processed_preview = add_health_features(subject)
    paths = current_output_paths()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("受试者级记录", f"{len(subject):,}")
    col2.metric("成人分析记录", f"{len(processed_preview):,}")
    col3.metric("派生字段数", f"{processed_preview.shape[1]:,}")
    col4.metric("数据仓库", "已生成" if paths["warehouse"].exists() else "待生成")

    st.markdown("**预处理步骤**")
    st.write(
        "清洗未知编码和异常值，按 SEQN 汇总用药记录，生成年龄组、BMI 分层、血压、糖化血红蛋白、胆固醇、饮食、活动和药物相关标签。"
    )
    st.markdown("**仓库分层**")
    st.write("ODS 原始受试者表，DWD 清洗宽表，ADS 聚合统计表，DM 关联规则与分类模型结果表。")
    show_dataframe(processed_preview.head(200), height=420)


def display_aggregation_module():
    st.subheader("聚合分析")
    paths = current_output_paths()
    processed = load_csv_if_exists(paths["processed"])
    if processed.empty:
        st.info("请先点击侧边栏“运行完整分析”。")
        return
    aggregations = {
        "年龄分组": load_csv_if_exists(RESULTS_DIR / "aggregation_by_age_group.csv"),
        "性别": load_csv_if_exists(RESULTS_DIR / "aggregation_by_gender.csv"),
        "BMI 分层": load_csv_if_exists(RESULTS_DIR / "aggregation_by_bmi_group.csv"),
        "风险等级": load_csv_if_exists(RESULTS_DIR / "aggregation_by_risk_level.csv"),
    }
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("成人样本", f"{len(processed):,}")
    col2.metric("高风险比例", f"{processed['high_cardiometabolic_risk'].mean():.1%}")
    col3.metric("平均 BMI", f"{processed['bmi'].mean():.1f}")
    col4.metric("平均收缩压", f"{processed['systolic_bp'].mean():.1f}")

    left, right = st.columns(2)
    with left:
        image = RESULTS_DIR / "aggregation_risk_by_age.png"
        if image.exists():
            st.image(str(image), caption="按年龄分组的高心代谢风险比例")
    with right:
        image = RESULTS_DIR / "aggregation_risk_by_bmi.png"
        if image.exists():
            st.image(str(image), caption="按 BMI 分层的高心代谢风险比例")

    selected = st.selectbox("查看聚合表", list(aggregations.keys()))
    show_dataframe(aggregations[selected], height=320)


def display_association_module(data_path: Path, min_support: float, min_confidence: float, nrows):
    st.subheader("关联规则")
    paths = current_output_paths()
    rules = load_csv_if_exists(paths["rules"])
    if st.button("仅重新运行关联规则", use_container_width=False):
        raw = load_raw_data(data_path=data_path, nrows=nrows)
        processed = add_health_features(aggregate_subject_level(raw))
        transactions = [transaction_from_row(row) for _, row in processed.iterrows()]
        _, rules = apriori_rules(
            transactions,
            min_support=min_support,
            min_confidence=min_confidence,
            max_len=3,
        )
        rules.to_csv(paths["rules"], index=False, encoding="utf-8-sig")
        st.success("关联规则已更新。")

    if rules.empty:
        st.info("暂无规则，请先运行完整分析，或降低支持度/置信度。")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("规则数", f"{len(rules):,}")
    col2.metric("最高置信度", f"{rules['confidence'].max():.3f}")
    col3.metric("最高提升度", f"{rules['lift'].max():.3f}")
    st.caption("规则格式为 antecedent -> consequent，重点筛选风险分层、疾病风险和典型用药相关后件。")
    show_dataframe(rules, height=520)


def display_classification_module(data_path: Path, max_depth: int, nrows):
    st.subheader("分类预测")
    paths = current_output_paths()
    metrics = load_csv_if_exists(paths["metrics"])
    importance = load_csv_if_exists(paths["importance"])
    confusion = load_csv_if_exists(paths["confusion"])

    if st.button("仅重新训练分类模型", use_container_width=False):
        raw = load_raw_data(data_path=data_path, nrows=nrows)
        processed = add_health_features(aggregate_subject_level(raw))
        metrics_dict, confusion_df, importance, tree_text = run_classification(processed, max_depth=max_depth)
        pd.DataFrame([metrics_dict]).to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
        confusion_df.to_csv(paths["confusion"], encoding="utf-8-sig")
        importance.to_csv(paths["importance"], index=False, encoding="utf-8-sig")
        (RESULTS_DIR / "classification_tree_rules.txt").write_text(tree_text, encoding="utf-8")
        metrics = pd.DataFrame([metrics_dict])
        confusion = confusion_df.reset_index()
        st.success("分类模型已更新。")

    if metrics.empty:
        st.info("暂无分类结果，请先运行完整分析。")
        return

    row = metrics.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{row['accuracy']:.3f}")
    col2.metric("Precision", f"{row['precision']:.3f}")
    col3.metric("Recall", f"{row['recall']:.3f}")
    col4.metric("F1", f"{row['f1']:.3f}")

    left, right = st.columns(2)
    with left:
        image = RESULTS_DIR / "classification_confusion_matrix.png"
        if image.exists():
            st.image(str(image), caption="决策树混淆矩阵")
        else:
            show_dataframe(confusion, height=240)
    with right:
        image = RESULTS_DIR / "classification_feature_importance.png"
        if image.exists():
            st.image(str(image), caption="决策树特征重要性")
    show_dataframe(importance, height=360)


def display_export_module():
    st.subheader("结果展示与导出")
    paths = current_output_paths()
    exports = [
        ("预处理后数据", paths["processed"]),
        ("SQLite 数据仓库", paths["warehouse"]),
        ("关联规则 CSV", paths["rules"]),
        ("分类指标 CSV", paths["metrics"]),
        ("HTML 分析报告", paths["html"]),
        ("Word 大作业报告", paths["docx"]),
        ("PDF 大作业报告", paths["pdf"]),
        ("Markdown 大作业报告", paths["md"]),
    ]
    for label, path in exports:
        exists = path.exists()
        col1, col2, col3 = st.columns([2, 4, 2])
        col1.write(label)
        col2.code(str(path), language=None)
        if exists:
            col3.download_button(
                "下载",
                data=path.read_bytes(),
                file_name=path.name,
                use_container_width=True,
            )
        else:
            col3.write("未生成")


def display_help_module():
    st.subheader("系统说明")
    st.write(
        "本系统面向课程大作业要求，围绕 NHANES 医学健康数据实现数据导入、预处理、数据仓库、聚合分析、关联规则、分类预测和可视化输出。"
    )
    st.markdown(
        """
**运行方式**

1. 命令行复现：`python src/main.py`
2. Web 系统：`streamlit run src/app.py`

**核心算法**

- 聚合分析：按年龄、性别、BMI 和风险等级统计高风险率、糖尿病率、高血压率、高胆固醇率等指标。
- 关联规则：Apriori 挖掘生活方式、体检指标、疾病风险和用药标签之间的规则。
- 分类算法：决策树预测是否属于心代谢高风险人群，并输出准确率、召回率、F1 和特征重要性。

**数据脱敏**

NHANES 数据使用公开受试者编号 `SEQN`，不包含姓名、电话、地址等直接身份标识。本系统不新增任何个人身份信息。
"""
    )


def main():
    st.title("NHANES 医学数据仓库与数据挖掘系统")
    st.caption("心代谢风险分层、关联规则挖掘与高风险分类预测")

    data_path, min_support, min_confidence, max_depth, nrows, run_clicked = sidebar_controls()
    if run_clicked:
        run_analysis(data_path, min_support, min_confidence, max_depth, nrows)

    tabs = st.tabs(["数据导入", "预处理与仓库", "聚合分析", "关联规则", "分类预测", "结果导出", "系统说明"])
    with tabs[0]:
        display_import_module(data_path, nrows)
    with tabs[1]:
        display_preprocess_module(data_path, nrows)
    with tabs[2]:
        display_aggregation_module()
    with tabs[3]:
        display_association_module(data_path, min_support, min_confidence, nrows)
    with tabs[4]:
        display_classification_module(data_path, max_depth, nrows)
    with tabs[5]:
        display_export_module()
    with tabs[6]:
        display_help_module()


if __name__ == "__main__":
    main()
