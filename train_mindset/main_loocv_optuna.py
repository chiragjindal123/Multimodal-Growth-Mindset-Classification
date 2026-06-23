import os
import json
from datetime import datetime
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import LeaveOneOut
from models_config import get_model_and_space  # 前面寫好的檔案
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 設定路徑與時間戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_path = "/data/feature_data"
output_dir = f"./output/0612_Bayesian_Optimization/{timestamp}_5modal"
os.makedirs(output_dir, exist_ok=True)

# 載入數據
label_csv = pd.read_csv(f"{data_path}/y_true_1140217.csv")[["面試代碼 (kkey)", "平均分數"]]
with open(f"{data_path}/interview_all_video_export_1140217.json", "r") as f:
    feature_data = json.load(f)

# 匹配數據
combined_data = []
for entry in feature_data:
    sid = entry["sid"]
    matching_row = label_csv[label_csv["面試代碼 (kkey)"] == sid]
    if not matching_row.empty:
        score = matching_row["平均分數"].values[0]
        for feature in entry["feature"]:
            combined_data.append({
                "features": feature,
                "sid": sid,
                "score": score
            })

# 提取特徵
text_features = np.array([entry["features"]["text"] for entry in combined_data])
audio_features = np.array([entry["features"]["audio"] for entry in combined_data])
eye_features = np.array([entry["features"]["eye_movement"] for entry in combined_data])
face_features = np.array([np.mean(entry["features"]["face"], axis=0) for entry in combined_data])
hrv_features = np.array([entry["features"]["heart_rate_variability"] for entry in combined_data])

# 處理 NaN 值
text_features = np.nan_to_num(text_features, nan=0.0)
audio_features = np.nan_to_num(audio_features, nan=0.0)
eye_features = np.nan_to_num(eye_features, nan=0.0)
face_features = np.nan_to_num(face_features, nan=0.0)
hrv_features = np.nan_to_num(hrv_features, nan=0.0)

# 定義標籤
def categorize(score):
    if 1 <= score <= 1.9:
        return 0
    elif 2 <= score <= 2.9:
        return 1
    elif 3 <= score <= 3.9:
        return 2
    elif 4 <= score <= 5:
        return 3
    return -1  # 如果分數不在範圍內，可以返回 -1

scores = np.array([entry["score"] for entry in combined_data])
categories = (scores >= 3).astype(int)  # 二分類
# categories = np.array([categorize(score) for score in scores]) # 四分類
sids = np.array([entry["sid"] for entry in combined_data])  # 提取 sid


scaler_text = StandardScaler().fit(text_features)
scaler_audio = StandardScaler().fit(audio_features)
scaler_face = StandardScaler().fit(face_features)
scaler_hrv = MinMaxScaler().fit(hrv_features)
scaler_eye = MinMaxScaler().fit(eye_features)
text_scaled = scaler_text.transform(text_features)
audio_scaled = scaler_audio.transform(audio_features)
face_scaled = scaler_face.transform(face_features)
hrv_scaled = scaler_hrv.transform(hrv_features)
eye_scaled = scaler_eye.transform(eye_features)

all_features = np.hstack([
    text_scaled, audio_scaled, face_scaled, hrv_scaled ,eye_scaled
])
# all_features = np.hstack([text_features, audio_features, eye_features, face_features, hrv_features])
print("All features shape:", all_features.shape)  # 應該是 (N, 768 + 193 + 7 + face_dim + HRV_dim)


# 🧪 輸入資料
X = all_features  # 你的特徵矩陣
y = categories      # 你的目標標籤

model_dir = os.path.join(output_dir, "models")
os.makedirs(model_dir, exist_ok=True)

# 要跑的模型名稱
model_names = [
    # "Logistic Regression", 
    # "SVM", "Random Forest", 
    "Gradient Boosting", "Decision Tree", 
    # "KNN", "MLP", "XGBoost", 
    # "SMOTEBoost", "RUSBoost"
]

# 儲存每個模型的結果
loo = LeaveOneOut()
loo_results = defaultdict(lambda: {"y_true": [], "y_pred": []})

# 🧪 主迴圈：LOOCV + Bayesian Optimization
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for model_name in model_names:
        def objective(trial):
            model_pipeline = get_model_and_space(model_name, trial)
            model_pipeline.fit(X_train, y_train)
            y_pred = model_pipeline.predict(X_train)
            return f1_score(y_train, y_pred, zero_division=1)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)

        # 用最佳參數重建模型，進行最終預測
        best_trial = study.best_trial
        best_pipeline = get_model_and_space(model_name, best_trial)
        best_pipeline.fit(X_train, y_train)
        y_pred = best_pipeline.predict(X_test)

        # 儲存預測
        loo_results[model_name]["y_true"].append(y_test[0])
        loo_results[model_name]["y_pred"].append(y_pred[0])

# 📊 統計與評估指標
summary = []
for model_name, result in loo_results.items():
    y_true = result["y_true"]
    y_pred = result["y_pred"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=1)
    rec = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    summary.append([model_name, acc, prec, rec, f1])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fixed", "Growth"])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix (LOOCV) - {model_name}")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_loocv_{model_name.replace(' ', '_')}.png"))
    plt.close()
    print(f"[✓] Confusion Matrix for {model_name} saved.")

# 儲存 CSV 結果
df_loocv_summary = pd.DataFrame(summary, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
df_loocv_summary.to_csv(os.path.join(output_dir, "loocv_model_performance_summary.csv"), index=False)
print("[✓] LOOCV summary saved.")
print(df_loocv_summary)

# 找出最佳模型（F1最高）
best_model_row = df_loocv_summary.sort_values(by="F1", ascending=False).iloc[0]
best_model_name = best_model_row["Model"]
print(f"\n🏆 Best LOOCV Model: {best_model_name} with F1-score = {best_model_row['F1']:.4f}")

# 用全部資料重新訓練最佳模型
def final_objective(trial):
    model = get_model_and_space(best_model_name, trial)
    model.fit(X, y)
    y_pred = model.predict(X)
    return f1_score(y, y_pred, zero_division=1)

study_final = optuna.create_study(direction="maximize")
study_final.optimize(final_objective, n_trials=10)

final_model = get_model_and_space(best_model_name, study_final.best_trial)
final_model.fit(X, y)

# 儲存模型
joblib.dump(final_model, os.path.join(model_dir, f"best_loocv_model_{best_model_name.replace(' ', '_')}.joblib"))
print(f"[✓] Best model saved to: {model_dir}")
