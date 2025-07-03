
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(layout="wide")
st.title("FastTrack Logistics Feasibility Dashboard")

# Load data
df = pd.read_csv("fasttrack_synthetic_data.csv")

# Sidebar
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Data Visualisation", "Classification", "Clustering", "Association Rule Mining", "Regression"])

# Tab 1: Data Visualisation
if tab == "Data Visualisation":
    st.header("Descriptive Insights")
    st.write("This section provides summary statistics and key visual trends in the logistics data.")
    st.dataframe(df.head())

    # Example visualizations
    fig1, ax1 = plt.subplots()
    sns.histplot(df["cur_delivery_time_hr"], kde=True, bins=30, ax=ax1)
    ax1.set_title("Distribution of Current Delivery Time (Hours)")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(x="industry", y="cur_cost_aed", data=df, ax=ax2)
    ax2.set_title("Cost per Parcel by Industry")
    st.pyplot(fig2)

    st.bar_chart(df["origin_city"].value_counts())

    st.line_chart(df.groupby("roi_months")["cur_cost_aed"].mean())

# Tab 2: Classification
elif tab == "Classification":
    st.header("Classification Models")
    st.write("Choose a model to classify switch likelihood (1–5 scale).")

    features = ['cur_delivery_time_hr', 'cur_cost_aed', 'fuel_cost', 'driver_wage']
    X = df[features]
    y = df["switch_likelihood"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_choice = st.selectbox("Select Classifier", ["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"])
    if model_choice == "KNN":
        model = KNeighborsClassifier()
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = GradientBoostingClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    if st.checkbox("Show Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

    if st.checkbox("Show ROC Curve"):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=model.classes_[1])
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"{model_choice} (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Model does not support probability scores.")

# Tab 3: Clustering
elif tab == "Clustering":
    st.header("Customer Segmentation using K-Means")
    st.write("Adjust cluster number to explore customer segments.")

    k = st.slider("Select number of clusters", 2, 10, 3)
    cluster_features = ['cur_cost_aed', 'avg_distance_km', 'pct_same_day']
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[cluster_features])

    st.dataframe(df[['industry', 'origin_city', 'cur_cost_aed', 'cluster']].head())

    fig, ax = plt.subplots()
    sns.scatterplot(x="avg_distance_km", y="cur_cost_aed", hue="cluster", data=df, palette="tab10", ax=ax)
    st.pyplot(fig)

    st.download_button("Download Clustered Data", df.to_csv(index=False), file_name="clustered_data.csv")

# Tab 4: Association Rule Mining
elif tab == "Association Rule Mining":
    st.header("Association Rule Mining on Adoption Drivers")
    st.write("Discover patterns in feature adoption behavior.")

    transactions = df["adoption_drivers"].dropna().str.split(", ").tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_ary, columns=te.columns_)

    min_support = st.slider("Min Support", 0.01, 0.5, 0.05)
    min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5)
    freq_items = apriori(trans_df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)

    top10_rules = rules.sort_values("confidence", ascending=False).head(10)
    st.dataframe(top10_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Tab 5: Regression
elif tab == "Regression":
    st.header("Cost Modeling with Regression")
    st.write("Explore how different models predict cost per delivery.")

    reg_features = ['avg_distance_km', 'fuel_cost', 'driver_wage', 'maint_cost_per_km']
    target = df['cur_cost_aed']
    X = df[reg_features]
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Linear": LogisticRegression(max_iter=200),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        st.write(f"{name} Regression R² Score: {score:.2f}")
