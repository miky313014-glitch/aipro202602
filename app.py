import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 設定頁面標題
st.set_page_config(page_title="酒類資料集預測儀表板", layout="wide")

# 模擬 sklearn 的 Bunch 物件結構，以便不改動下游程式碼
class DataBunch:
    def __init__(self, target):
        self.target = target

# 載入資料集
@st.cache_data
def load_data():
    df = pd.read_csv('wine.csv')
    wine = DataBunch(df['target'].values)
    return wine, df

wine_data, df_wine = load_data()

# --- Sidebar ---
st.sidebar.header("模型設定")
model_option = st.sidebar.selectbox(
    "選擇預測模型",
    ("KNN", "羅吉斯迴歸", "Random Forest", "XGBoost")
)

st.sidebar.markdown("---")
st.sidebar.header("資料集資訊")
st.sidebar.info(f"""
**資料集名稱：** 酒類 (Wine)
**樣本總數：** {df_wine.shape[0]}
**特徵數量：** {df_wine.shape[1] - 1}
**類別數量：** {len(np.unique(wine_data.target))}
""")

# --- Main Area ---
st.title("🍷 酒類資料集預測儀表板")

col1, col2 = st.columns(2)

with col1:
    st.subheader("資料集前 5 筆內容")
    st.dataframe(df_wine.head())

with col2:
    st.subheader("特徵統計值")
    st.dataframe(df_wine.describe())

st.markdown("---")

# 模型檔案路徑映射
MODEL_PATHS = {
    "KNN": "k-nearest_neighbors_model.joblib",
    "羅吉斯迴歸": "logistic_regression_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "XGBoost": "xgboost_model.joblib"
}

if st.button("進行預測"):
    # 資料準備
    X = df_wine.drop('target', axis=1)
    y = df_wine['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 載入預訓練模型
    model_path = MODEL_PATHS.get(model_option)
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            
            # 預測
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # 顯示結果
            st.success(f"### 預測完成！(使用預訓練模型：{os.path.basename(model_path)})")
            st.metric(label="模型準確度 (Accuracy)", value=f"{acc:.2%}")
            
            st.subheader("分類報告 (Classification Report)")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        except Exception as e:
            st.error(f"載入模型時發生錯誤：{e}")
    else:
        st.error(f"找不到預訓練模型檔案：{model_path}")
