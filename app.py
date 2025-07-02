import streamlit as st
import pandas as pd
import openpyxl
import pickle
from modules import data_cleaning, eda, visualization, run_query, model_trainer

st.set_page_config(page_title="EasyML Studio", layout="wide")

st.title("📊 EasyML Studio")
st.subheader("No Code Data Analysis & Machine Learning Platform!")

st.write("Welcome to EasyML Studio! This app simplifies data analysis and machine learning with a user-friendly interface.")
st.write("Upload your dataset, perform data cleaning, and get your model trained effortlessly.")
st.write("Use the tabs below to navigate through different features.")

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁 Upload", "🧹 Data Cleaning", "📊 EDA", "📈 Visualization", "🧮 Work with SQL", "🧠 Model Training"
 ])

# File upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])

    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()

if uploaded_file and "raw_df" not in st.session_state:
    if uploaded_file.name.endswith('.csv'):
        st.session_state["raw_df"] = pd.read_csv(uploaded_file)
    else:
        st.session_state["raw_df"] = pd.read_excel(uploaded_file)

# Determine working DataFrame
working_df = st.session_state.get("cleaned_df", st.session_state.get("raw_df", None))

with tab1:
    st.subheader("📁 Upload Dataset")
    if uploaded_file:
        if 'cleaned_df' in st.session_state:
            st.subheader("Preview of Cleaned Dataset")
        else:
            st.subheader("Preview of Uploaded Dataset")
        st.dataframe(working_df)
    else:
        st.info("👈 Upload a file to begin")

with tab2:
    if working_df is not None:
        st.header("🧹 Data Cleaning")
        display_df = st.session_state.get("cleaned_df", working_df)
        st.dataframe(display_df)
        data_cleaning.clean_data(working_df)

with tab3:
    if working_df is not None:
        st.header("📊 Exploratory Data Analysis (EDA)")
        st.dataframe(working_df)
        eda.run_eda(working_df)

with tab4:
    if working_df is not None:
        st.header("📈 Visualizations")
        st.dataframe(working_df)
        visualization.plot_visuals(working_df)

with tab5:
    if working_df is not None:
        st.header("🧮 SQL Query")
        st.dataframe(working_df)
        run_query.run_query(working_df)

with tab6:
    if working_df is not None:
        st.header("🧠 Model Training")
        st.dataframe(working_df)
        target_column = st.selectbox("🎯 Select the target column", working_df.columns)

        if st.button("Train Model"):
            result = model_trainer.train_model(working_df, target_column)
            evaluation = result["evaluation"]
            task_type = result["task_type"]
            best_model = result["best_model"]
            best_model_name = model_trainer.select_best_model_name(evaluation, task_type)

            st.session_state["best_model"] = best_model
            st.session_state["best_model_name"] = best_model_name

            st.subheader("📈 Evaluation Results")
            eval_df = pd.DataFrame.from_dict(evaluation, orient="index")
            st.dataframe(eval_df.style.highlight_max(axis=0), use_container_width=True)

        if "best_model" in st.session_state and "best_model_name" in st.session_state:
            st.subheader("⬇️ Download Best Model")
            st.success(f"Best Model: `{st.session_state['best_model_name']}`")
            st.download_button(
                label=f"💾 Download {st.session_state['best_model_name']}",
                data=model_trainer.get_serialized_model(st.session_state["best_model"]),
                file_name=f"{st.session_state['best_model_name'].replace(' ', '_').lower()}.pkl",
                mime="application/octet-stream"
            )
