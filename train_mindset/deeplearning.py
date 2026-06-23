import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
from sklearn.model_selection import StratifiedKFold


# 設定路徑與時間戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_path = "/data/feature_data"
output_dir = f"./deeplearning_output/0603/{timestamp}_unimodal_face_lstm"
os.makedirs(output_dir, exist_ok=True)

# 載入數據
label_csv = pd.read_csv(f"{data_path}/y_true_1140217.csv")[["面試代碼 (kkey)", "平均分數"]]
with open(f"{data_path}/interview_all_video_export_1140217.json", "r") as f:
    feature_data = json.load(f)

# 匹配數據
combined_data = []
face_sequences = []
for entry in feature_data:
    sid = entry["sid"]
    matching_row = label_csv[label_csv["面試代碼 (kkey)"] == sid]
    if not matching_row.empty:
        score = matching_row["平均分數"].values[0]
        for feature in entry["feature"]:
            face_seq = feature["face"]
            if not face_seq or len(face_seq) == 0:
                continue  # 跳過空的序列
            face_sequences.append(np.array(face_seq))  # 保留 shape=(num_frames, 256)
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
print(f"face: {len(face_sequences)}")
print(f"face: {face_sequences[1].shape}")

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

# # -------- ① 只保留「類別 1 或 2」的樣本 ----------
# mask = np.isin(categories, [0, 1])

# # ★ 必須讓「每一筆資料對應的所有陣列」都用同一個 mask 過濾
# text_features  = text_features[mask]
# audio_features = audio_features[mask]
# eye_features   = eye_features[mask]
# face_features  = face_features[mask]
# hrv_features   = hrv_features[mask]
# scores         = scores[mask]
# sids           = sids[mask]

# categories = (categories[mask] == 1).astype(int)


# all_features = np.hstack([text_features, audio_features, eye_features, face_features, hrv_features])

# print("All features shape:", all_features.shape)  # 應該是 (N, 768 + 193 + 7 + face_dim + HRV_dim)

fused_features = face_sequences

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
# 將 face sequences 補零為相同長度 (LSTM 輸入要求)
face_sequences_padded = pad_sequences(face_sequences, padding='post', dtype='float32')  # shape = (samples, max_timesteps, 256)

# 模型輸出資料夾
output_face_lstm = output_dir

# 初始化交叉驗證
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {"y_true": [], "y_pred": []}

for fold, (train_idx, test_idx) in enumerate(kfold.split(face_sequences_padded, categories), 1):
    X_train, X_test = face_sequences_padded[train_idx], face_sequences_padded[test_idx]
    y_train, y_test = categories[train_idx], categories[test_idx]

    # 模型架構
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(None, 256)),  # 掩蔽 padding 值
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 訓練模型
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train,
              validation_split=0.2,
              epochs=1000,
              batch_size=16,
              callbacks=[early_stop],
              verbose=0)

    # 預測
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    results["y_true"].extend(y_test)
    results["y_pred"].extend(y_pred)

    print(f"[✓] Fold {fold} completed.")

# 評估指標
y_true = results["y_true"]
y_pred = results["y_pred"]
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=1)
rec = recall_score(y_true, y_pred, zero_division=1)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n📊 Face LSTM Performance:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# 混淆矩陣圖
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fixed", "Growth"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Face LSTM")
plt.tight_layout()
plt.savefig(os.path.join(output_face_lstm, "confusion_matrix_face_lstm.png"))
plt.close()

# 儲存模型
model.save(os.path.join(output_face_lstm, "face_lstm_model.h5"))
import pandas as pd

# 建立 Face LSTM 單一結果 DataFrame
face_lstm_summary = pd.DataFrame([{
    "Model": "Deep Learning (Face LSTM)",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1": f1
}])

# 儲存成獨立 CSV
face_lstm_summary.to_csv(os.path.join(output_face_lstm, "face_lstm_result.csv"), index=False)

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# print("\n🚀 Training Deep Learning Model with StratifiedKFold...")

# # 標準化特徵
# scaler_dl = StandardScaler()
# X_scaled_dl = scaler_dl.fit_transform(fused_features)

# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# dl_results = {"y_true": [], "y_pred": []}

# for fold, (train_idx, test_idx) in enumerate(kfold.split(X_scaled_dl, categories), 1):
#     X_train, X_test = X_scaled_dl[train_idx], X_scaled_dl[test_idx]
#     y_train, y_test = categories[train_idx], categories[test_idx]

#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dropout(0.2),
#         Dense(1, activation='sigmoid')
#     ])

#     model.compile(optimizer=Adam(learning_rate=0.001),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])

#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#     model.fit(X_train, y_train,
#               validation_split=0.2,
#               epochs=500,
#               batch_size=16,
#               callbacks=[early_stop],
#               verbose=0)

#     y_pred_prob = model.predict(X_test).flatten()
#     y_pred = (y_pred_prob > 0.5).astype(int)

#     dl_results["y_true"].extend(y_test)
#     dl_results["y_pred"].extend(y_pred)

#     print(f"[✓] Fold {fold} completed.")

# # 評估指標
# dl_y_true = dl_results["y_true"]
# dl_y_pred = dl_results["y_pred"]
# dl_acc = accuracy_score(dl_y_true, dl_y_pred)
# dl_prec = precision_score(dl_y_true, dl_y_pred, zero_division=1)
# dl_rec = recall_score(dl_y_true, dl_y_pred, zero_division=1)
# dl_f1 = f1_score(dl_y_true, dl_y_pred)
# dl_cm = confusion_matrix(dl_y_true, dl_y_pred)

# # 加入 summary
# dl_summary_row = pd.DataFrame([["Deep Learning (MLP)", dl_acc, dl_prec, dl_rec, dl_f1]],
#                               columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

# # 畫混淆矩陣
# disp = ConfusionMatrixDisplay(confusion_matrix=dl_cm, display_labels=["Fixed", "Growth"])
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix - Deep Learning (StratifiedKFold)")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "confusion_matrix_deep_learning.png"))
# plt.close()

# # 儲存模型
# model.save(os.path.join(output_dir, "model/best_deep_learning_model.h5"))
# joblib.dump(scaler_dl, os.path.join(output_dir, "model/scaler_dl.joblib"))

# print("\n[✓] Deep Learning model saved.")
# print(dl_summary_row)
# # 儲存 summary 成 CSV
# summary_path = os.path.join(output_dir, "deep_learning_model_performance_summary.csv")
# dl_summary_row.to_csv(summary_path, index=False)
# print(f"[✓] Model performance summary saved to: {summary_path}")

