import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Page Configuration
st.set_page_config(page_title="Advertising MLR", layout="centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Title
st.markdown("""
<div class='card'> 
    <h1>Multiple Linear Regression</h1>
    <p>Predict <b>Sales</b> using <b>TV, Radio & Newspaper Advertising</b></p>
</div>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Advertising.csv")
    df.columns = df.columns.str.strip()
    return df
df = load_data()

# Dataset Preview
st.markdown("""
<div class="card">
    <h2>Dataset Preview</h2></div>""", unsafe_allow_html=True)

st.dataframe(df.head())
# Prepare Data
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Visualization
st.markdown("""
<div class="card">
    <h2>Actual vs Predicted Sales</h2>
</div>""", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(df['Radio'],df['Sales'],alpha=0.8)
ax.plot(df['Radio'],model.predict(scaler.transform(X)),color='red')

ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title("MLR Model Performance")
ax.grid(alpha=0.3)
st.pyplot(fig)

# PERFORMANCE 
st.markdown(f"""
<div class="card">
    <h2>Model Performance</h2>
    <p>
    <b>TV Coefficient:</b> {model.coef_[0]:.3f}<br>
    <b>Radio Coefficient:</b> {model.coef_[1]:.3f}<br>
    <b>Newspaper Coefficient:</b> {model.coef_[2]:.3f}<br>
    <b>Intercept:</b> {model.intercept_:.3f}<br><br>
    <b>RMSE:</b> {rmse:.2f}<br>
    <b>R² Score:</b> {r2:.3f}<br>
    <b>MAPE:</b> {mape:.2%}
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.markdown("""
<div class="card">
    <h2>Sales Prediction</h2>
</div>
""", unsafe_allow_html=True)

tv = st.slider("TV Advertising Budget", float(X['TV'].min()), float(X['TV'].max()), float(X['TV'].mean()))
radio = st.slider("Radio Advertising Budget", float(X['Radio'].min()), float(X['Radio'].max()), float(X['Radio'].mean()))
newspaper = st.slider("Newspaper Advertising Budget", float(X['Newspaper'].min()), float(X['Newspaper'].max()), float(X['Newspaper'].mean()))

input_df = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
input_scaled = scaler.transform(input_df)
predicted_sales = model.predict(input_scaled)[0]

st.markdown(
    f"""
    <div class="prediction-box">
        Predicted Sales: {predicted_sales:.2f}
    </div>
    """,
    unsafe_allow_html=True
)
