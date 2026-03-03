import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 頁面標題與設定
st.set_page_config(page_title="Wine 機器學習分類器", layout="wide")

# 1. 載入資料集
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return wine, df

wine_data, df = load_data()

# 2. 左側 Sidebar
st.sidebar.header("模型與資訊")

# 模型選擇下拉選單
model_option = st.sidebar.selectbox(
    "請選擇機器學習模型：",
    ("KNN (K-最近鄰)", "Logistic Regression (羅吉斯迴歸)", "XGBoost", "Random Forest (隨機森林)")
)

st.sidebar.markdown("---")
st.sidebar.subheader("「酒類」資料集資訊")
st.sidebar.info(f"""
**資料集名稱：** Wine (Scikit-learn 內建)
**樣本數：** {df.shape[0]}
**特徵數：** {df.shape[1] - 1}
**類別數：** {len(np.unique(wine_data.target))} (Class 0, 1, 2)
""")

# 3. 右側 Main 區域
st.title("🍷 Wine 資料集分類分析")
st.write(f"當前選擇的模型: **{model_option}**")

# 數據預覽
st.subheader("📊 資料前 5 筆預覽")
st.dataframe(df.head())

# 統計值資訊
st.subheader("📈 特徵統計數值")
st.write(df.describe())

st.markdown("---")

# 4. 預測與結果
st.subheader("🤖 模型預測")

# 準備訓練資料
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型對應表
models = {
    "KNN (K-最近鄰)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression (羅吉斯迴歸)": LogisticRegression(max_iter=10000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "Random Forest (隨機森林)": RandomForestClassifier(n_estimators=100)
}

if st.button("🚀 開始進行預測"):
    with st.spinner('模型訓練中，請稍候...'):
        # 選擇並訓練模型
        model = models[model_option]
        model.fit(X_train, y_train)
        
        # 預測
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # 顯示結果
        st.success("✅ 預測完成！")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="準確度 (Accuracy)", value=f"{acc:.2%}")
        
        with col2:
            st.write("**測試集前 5 筆預測結果 vs 實際結果：**")
            results_df = pd.DataFrame({
                '實際類別': y_test[:5].values,
                '預測類別': y_pred[:5]
            })
            st.table(results_df)

        # 視覺化簡單說明 (特徵重要性 - 僅 RandomForest 與 XGBoost)
        if model_option in ["Random Forest (隨機森林)", "XGBoost"]:
            st.subheader("📌 特徵重要性 (Feature Importance)")
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
            st.bar_chart(feat_imp)
