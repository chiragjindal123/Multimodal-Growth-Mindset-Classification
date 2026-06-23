import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneOut, StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import mannwhitneyu, kruskal, normaltest, ttest_ind
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.feature_selection import RFE
from imblearn.under_sampling import RandomUnderSampler
from collections import defaultdict
import joblib
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
# imbalanced-learn imports for boosting
from imblearn.ensemble import RUSBoostClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


# 設定路徑與時間戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_path = "/data/feature_data"
output_dir = f"./output/all/{timestamp}_text"
os.makedirs(output_dir, exist_ok=True)

# 載入數據
label_csv = pd.read_csv(f"{data_path}/output_updated.csv")[["sid", "context_text", "value"]]
with open(f"{data_path}/interview_all_export_with_sentence_0626.json", "r", encoding="big5") as f:
    feature_data = json.load(f)

# print(len(feature_data))
# print(feature_data[0].keys())
# print(len(feature_data[0]['feature']))
# print(feature_data[0]['feature'][0].keys())
# print(len(feature_data[0]['feature'][0]['text']))
# print(feature_data[0]['feature'][0]['_context'])


# exit()

# 匹配數據
# combined_data = []
# for entry in feature_data:
#     sid = entry["sid"]
#     matching_row = label_csv[label_csv["sid"] == sid]
#     if not matching_row.empty:
#         score = matching_row["value"].values[0]
#         for feature in entry["feature"]:
#             combined_data.append({
#                 "features": feature,
#                 "sid": sid,
#                 "score": score
#             })

# key = (sid, _context), value = value
label_dict = {
    (row["sid"], row["context_text"]): row["value"]
    for _, row in label_csv.iterrows()
}

# 匹配資料並過濾 value != -1 的
combined_data = []
for entry in feature_data:
    sid = entry["sid"]
    matching_rows = label_csv[label_csv["sid"] == sid]

    if not matching_rows.empty:
        for i, feature in enumerate(entry["feature"]):
            context = feature.get("_context", {}).get("text", "").strip()
            matched = label_csv[
                (label_csv["sid"] == sid) &
                (label_csv["context_text"].str.strip() == context)
            ]
            # print(matched["value"].values[0])
            if not matched.empty:
                score = int(matched["value"].values[0])
                if score != -1:
                    combined_data.append({
                        "sid": sid,
                        "score": score,
                        "features": feature
                    })
                else:
                    print(f"value is -1 : {sid} - {context}")
            else:
                print(f"no match {sid}")

print(len(combined_data))
print("done")

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
categories = (scores >= 2).astype(int)  # 二分類
# categories = np.array([categorize(score) for score in scores]) # 四分類
# categories = scores # 四分類
sids = np.array([entry["sid"] for entry in combined_data])  # 提取 sid

# # -------- ① 只保留「類別 1 或 2」的樣本 ----------
# mask = np.isin(categories, [2, 3])

# # ★ 必須讓「每一筆資料對應的所有陣列」都用同一個 mask 過濾
# text_features  = text_features[mask]
# audio_features = audio_features[mask]
# eye_features   = eye_features[mask]
# face_features  = face_features[mask]
# hrv_features   = hrv_features[mask]
# scores         = scores[mask]
# sids           = sids[mask]

# categories = (categories[mask] == 3).astype(int)

'''
# 繪製數據分布圖
plt.figure(figsize=(12, 6))
ax = sns.histplot(scores, bins=20, stat='count', kde=False)
for patch in ax.patches:
    # bar 的高度
    height = patch.get_height()
    # bar 的 x 座標
    x = patch.get_x() + patch.get_width()/2.
    # 在 bar 上方一點點的位置加文字
    ax.text(x, height+0.5,        # y=高度+一點點位移
            f"{int(height)}",     # 顯示整數
            ha="center", va="bottom", fontsize=9)
plt.title("Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "score_distribution.png"))
plt.close()

# 正態性檢驗
stat, p = normaltest(scores)
with open(os.path.join(output_dir, "normality_test.txt"), "w") as f:
    f.write(f"Normality Test Statistic: {stat}, P-Value: {p}\n")

print(f"數據分布圖已儲存為 score_distribution.png，正態性檢驗結果存於 normality_test.txt")
'''

def perform_t_test_and_extract(features, labels, feature_name, writer):
    significant_indices = []
    for i in range(features.shape[1]):
        group1 = features[labels == 1, i]
        group2 = features[labels == 0, i]
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        writer.writerow([feature_name, f"Feature {i+1}", t_stat, p_value])
        if p_value < 0.05:
            significant_indices.append(i)
    return significant_indices

# Kruskal-Wallis H 檢定並提取顯著特徵
def perform_kruskal_test_and_extract(features, labels, feature_name, writer):
    significant_indices = []
    for i in range(features.shape[1]):
        group0 = features[labels == 0, i]
        group1 = features[labels == 1, i]
        group2 = features[labels == 2, i]
        group3 = features[labels == 3, i]
        stat, p_value = kruskal(group0, group1, group2, group3)
        writer.writerow([feature_name, f"Feature {i+1}", stat, p_value])
        if p_value < 0.05:  # 提取顯著特徵
            significant_indices.append(i)
    return significant_indices

# feature_significance = {}
# output_csv = os.path.join(output_dir, "t_test_results_all_data.csv")
# with open(output_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Feature Type", "Feature Index", "U-Statistic", "P-Value"])
#     # feature_significance["text"] = perform_kruskal_test_and_extract(text_features, categories, "Text Features", writer)
#     # feature_significance["audio"] = perform_kruskal_test_and_extract(audio_features, categories, "Audio Features", writer)
#     # feature_significance["eye"] = perform_kruskal_test_and_extract(eye_features, categories, "Eye Movement Features", writer)
#     # feature_significance["face"] = perform_kruskal_test_and_extract(face_features, categories, "Face Features", writer)
#     # feature_significance["hrv"] = perform_kruskal_test_and_extract(hrv_features, categories, "HRV Features", writer)
#     feature_significance["text"] = perform_t_test_and_extract(text_features, categories, "Text Features", writer)
#     feature_significance["audio"] = perform_t_test_and_extract(audio_features, categories, "Audio Features", writer)
#     feature_significance["eye"] = perform_t_test_and_extract(eye_features, categories, "Eye Movement Features", writer)
#     feature_significance["eye"] = perform_t_test_and_extract(eye_features, categories, "Eye Movement Features", writer)
#     feature_significance["face"] = perform_t_test_and_extract(face_features, categories, "Face Features", writer)
#     feature_significance["hrv"] = perform_t_test_and_extract(hrv_features, categories, "HRV Features", writer)

# print(f"Statistical analysis results saved to: {output_csv}")


# def extract_significant_features(features, significant_indices):
#     return features[:, significant_indices] if significant_indices else features

# X_filtered = {
#     "text": extract_significant_features(text_features, feature_significance["text"]),
#     "audio": extract_significant_features(audio_features, feature_significance["audio"]),
#     "eye": extract_significant_features(eye_features, feature_significance["eye"]),
#     "face": extract_significant_features(face_features, feature_significance["face"]),
#     "hrv": extract_significant_features(hrv_features, feature_significance["hrv"])
# }

# # 只計算每個模態的顯著特徵數量
# feature_counts = {key: X_filtered[key].shape[1] if X_filtered[key].size > 0 else 0 for key in X_filtered}

# # 印出每種類別的顯著特徵數
# print("\nSummary of significant features per modality:")
# for feature_type, count in feature_counts.items():
#     print(f"{feature_type}: {count} significant features")

# # 合併所有顯著特徵
# # fused_features = np.hstack([X_filtered[key] for key in X_filtered])
# fused_features = np.hstack([X_filtered[key] for key in X_filtered if X_filtered[key].size > 0])

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
    text_scaled, 
    audio_scaled, 
    face_scaled, 
    hrv_scaled ,
    eye_scaled
])
# all_features = np.hstack([text_features, audio_features, eye_features, face_features, hrv_features])
print("All features shape:", all_features.shape)  # 應該是 (N, 768 + 193 + 7 + face_dim + HRV_dim)

# base_estimator = SVC(kernel="linear")
# # base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
# # base_estimator = LogisticRegression(
# #     penalty='l2',      # or 'l1'，但要注意 solver 支持
# #     solver='liblinear',# 若用 l1 則可用 'liblinear' 或 'saga'
# #     random_state=42
# # )
# # base_estimator = XGBClassifier(
# #     n_estimators=100,
# #     random_state=42,
# #     importance_type='gain'    # 'gain' / 'weight' / 'cover' 皆可
# # )


# def perform_rfe(X, y, model, num_features=10):
#     """
#     使用 RFE 進行特徵選擇
#     X: 特徵矩陣 (可以是 numpy.ndarray 或 pandas.DataFrame)
#     y: 標籤
#     model: 基礎模型（如隨機森林、SVM）
#     num_features: 選擇的特徵數量
#     """
#     rfe = RFE(model, n_features_to_select=num_features)
#     rfe.fit(X, y)
#     selected_indices = np.where(rfe.support_)[0]
#     return selected_indices

# def perform_rfecv(X, y, model, min_features_to_select=1):
#     cv_strategy = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
#     rfecv = RFECV(
#         estimator=model,
#         step=1,
#         cv=cv_strategy,
#         scoring='accuracy',
#         min_features_to_select=min_features_to_select,
#         n_jobs=-1
#     )
#     rfecv.fit(X, y)
#     selected_indices = np.where(rfecv.support_)[0]
#     return selected_indices

# forced_text_block = text_features[:, forced_text_indices]
# # text_excluded = np.delete(text_features, forced_text_indices, axis=1)
# other_features = np.hstack([text_features, audio_features, face_features])

# # # Step 1: 對每個模態執行 RFE
# # selected_eye_indices = perform_rfe(eye_features, categories, base_estimator, num_features=13)
# # selected_hrv_indices = perform_rfe(hrv_features, categories, base_estimator, num_features=5)
# # selected_other_indices = perform_rfe(other_features, categories, base_estimator, num_features=50)

# # Step 1: 使用 RFECV 自動選擇特徵
# # selected_eye_indices = perform_rfecv(eye_features, categories, base_estimator, min_features_to_select=1)
# # selected_hrv_indices = perform_rfecv(hrv_features, categories, base_estimator, min_features_to_select=1)
# # selected_other_indices = perform_rfecv(other_features, categories, base_estimator, min_features_to_select=5)

# # Step 2: 生成最終特徵矩陣 (numpy.ndarray)
# fused_features = np.concatenate([
#     eye_features[:, selected_eye_indices],
#     hrv_features[:, selected_hrv_indices],
#     # forced_text_block,
#     other_features[:, selected_other_indices]
# ], axis=1)

# print(f"最終選擇的特徵數量: {fused_features.shape[1]}")
# print(f"最終特徵矩陣形狀: {fused_features.shape}")
# print(f"Fused significant features shape: {fused_features.shape}")

# selected_text_features = []
# selected_audio_features = []
# selected_face_features = []

# for idx in selected_other_indices:
#     if idx < text_dim:
#         # 屬於 text_features
#         selected_text_features.append(idx)
#     elif idx < text_dim + audio_dim:
#         # 屬於 audio_features
#         adjusted_idx = idx - text_dim
#         selected_audio_features.append(adjusted_idx)
#     else:
#         # 屬於 face_features
#         adjusted_idx = idx - text_dim - audio_dim
#         selected_face_features.append(adjusted_idx)

# # 依序列印出來
# print("Selected Text Feature Indices:", selected_text_features)
# print("Selected Audio Feature Indices:", selected_audio_features)
# print("Selected Face Feature Indices:", selected_face_features)

# # 建立 DataFrame
# df_selected = pd.DataFrame({
#     "eye_features": [selected_eye_indices.tolist()],
#     "hrv_features": [selected_hrv_indices.tolist()],
#     "text_indices": [selected_text_features],
#     "audio_indices": [selected_audio_features],
#     "face_indices": [selected_face_features]
# })

# # 輸出成 CSV
# df_selected.to_csv(os.path.join(output_dir, "selected_features_rfe.csv"), index=False)

fused_features = all_features
smote_boost = Pipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=1)),
    ('ada', AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=0.1,
        algorithm="SAMME",
        random_state=42
    ))
])

rus_boost = Pipeline([
    ('rus', RandomUnderSampler(random_state=42)),
    ('ada', AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=0.1,
        algorithm="SAMME",
        random_state=42
    ))
])

# 定義不同的機器學習模型

mlp = MLPClassifier(
    hidden_layer_sizes=(32, ),  # 根據數據選擇適當層數
    activation='relu',  
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    max_iter=500,
    early_stopping=True,
    learning_rate='adaptive',
    random_state=42
)

models = {
    "Logistic Regression": GridSearchCV(
        estimator=Pipeline([
            ("smote", SMOTE(k_neighbors=1, random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="liblinear", class_weight="balanced"))
        ]),
        param_grid={
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l1", "l2"]
        },
        cv=4
    ),

    "SVM": GridSearchCV(
        estimator=Pipeline([
            ("smote", SMOTE(k_neighbors=1, random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", SVC(class_weight="balanced"))
        ]),
        param_grid={
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["linear", "rbf", "poly"],
            "clf__gamma": ["scale", "auto"]
        },
        cv=4
    ),

    "Random Forest": GridSearchCV(
        estimator=Pipeline([
            ("smote", SMOTE(k_neighbors=1, random_state=42)),
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))
        ]),
        param_grid={
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20]
        },
        cv=4
    ),

    "Gradient Boosting": Pipeline([
        ("smote", SMOTE(k_neighbors=1, random_state=42)),
        ("clf", GradientBoostingClassifier(random_state=42))
    ]),

    "Decision Tree": Pipeline([
        ("smote", SMOTE(k_neighbors=1, random_state=42)),
        ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42))
    ]),

    "KNN": Pipeline([
        ("smote", SMOTE(k_neighbors=1, random_state=42)),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),

    "MLP": Pipeline([
        ("smote", SMOTE(k_neighbors=1, random_state=42)),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            learning_rate="adaptive",
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("smote", SMOTE(k_neighbors=1, random_state=42)),
        ("clf", XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric="logloss", random_state=42))
    ])
}

# models = {
#     "SVM": GridSearchCV(
#         SVC(class_weight="balanced"),
#         {"C":[0.1,1,10], "kernel":["linear","rbf","poly"], "gamma":["scale","auto"]},
#         cv=4
#     ),
#     # "Logistic Regression": GridSearchCV(
#     #     LogisticRegression(solver="liblinear", class_weight="balanced"),
#     #     {"C":[0.01,0.1,1,10], "penalty":["l1","l2"]},
#     #     cv=4
#     # ),
#     "Logistic Regression": GridSearchCV(
#         estimator=Pipeline([
#             ("smote", SMOTE(random_state=42, k_neighbors=1)),
#             ("scaler", StandardScaler()),
#             ("clf", LogisticRegression(solver="liblinear", class_weight="balanced"))
#         ]),
#         param_grid={
#             "clf__C": [0.01, 0.1, 1, 10],
#             "clf__penalty": ["l1", "l2"]
#         },
#         cv=4
#     ),
#     "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
#     "Random Forest": GridSearchCV(
#         RandomForestClassifier(class_weight="balanced", random_state=42),
#         {"n_estimators":[100,200], "max_depth":[None,10,20]},
#         cv=4
#     ),
#     "KNN": KNeighborsClassifier(),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=42),
#     "MLP": MLPClassifier(hidden_layer_sizes=(32,), early_stopping=True, random_state=42),
#     "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
#     # "RUSBoost": rus_boost,
#     # "SMOTEBoost": smote_boost

# }

# models = {
#     "SVM": GridSearchCV(
#         SVC(),
#         {"C":[0.1,1,10], "kernel":["linear","rbf","poly"], "gamma":["scale","auto"]},
#         cv=4
#     ),
#     "Logistic Regression": GridSearchCV(
#         LogisticRegression(solver="liblinear"),
#         {"C":[0.01,0.1,1,10], "penalty":["l1","l2"]},
#         cv=4
#     ),
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "Random Forest": GridSearchCV(
#         RandomForestClassifier(random_state=42),
#         {"n_estimators":[100,200], "max_depth":[None,10,20]},
#         cv=4
#     ),
#     "KNN": KNeighborsClassifier(),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=42),
#     "MLP": mlp,
#     "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
#     "RUSBoost": rus_boost,
#     "SMOTEBoost": smote_boost
# }

# 創建模型儲存的目錄
model_dir = os.path.join(output_dir, "models")
os.makedirs(model_dir, exist_ok=True)


# '''
# LOOCV
# '''
# loo = LeaveOneOut()
# loo_results = defaultdict(lambda: {"y_true": [], "y_pred": []})
# # sampler = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=3))
# # smote = SMOTE(random_state=42, k_neighbors=3)
# for train_idx, test_idx in loo.split(fused_features):
#     X_train, X_test = fused_features[train_idx], fused_features[test_idx]
#     y_train, y_test = categories[train_idx], categories[test_idx]
#     # smote = SMOTE(random_state=42, k_neighbors=1)
#     # X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#     # X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
#     # scaler = StandardScaler()
#     # X_train_scaled = scaler.fit_transform(X_train)
#     # X_test_scaled = scaler.transform(X_test)
#     # smote = SMOTE(random_state=42, k_neighbors=1)
#     # X_train_scaled, y_train_res = smote.fit_resample(X_train_scaled, y_train)


#     for model_name, model in models.items():
#         # fitted_model = model.fit(X_train_scaled, y_train)
#         # best_model = fitted_model.best_estimator_ if hasattr(fitted_model, "best_estimator_") else fitted_model
#         # y_pred = best_model.predict(X_test_scaled)
#         model.fit(X_train, y_train)
#         best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
#         y_pred = best_model.predict(X_test)
#         loo_results[model_name]["y_true"].append(y_test[0])
#         loo_results[model_name]["y_pred"].append(y_pred[0])

# # ==========================
# # 📊 統計並畫圖
# # ==========================
# summary = []
# for model_name, result in loo_results.items():
#     y_true = result["y_true"]
#     y_pred = result["y_pred"]

#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, zero_division=1)
#     rec = recall_score(y_true, y_pred, zero_division=1)
#     f1 = f1_score(y_true, y_pred)
#     cm = confusion_matrix(y_true, y_pred)

#     summary.append([model_name, acc, prec, rec, f1])

#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fixed", "Growth"])
#     disp.plot(cmap='Blues')
#     plt.title(f"Confusion Matrix (LOOCV) - {model_name}")
#     plt.savefig(os.path.join(output_dir, f"confusion_matrix_loocv_{model_name.replace(' ', '_')}.png"))
#     plt.close()
#     print(f"[✓] Confusion Matrix for {model_name} saved.")

# # 儲存成 DataFrame 與 CSV
# df_loocv_summary = pd.DataFrame(summary, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
# df_loocv_summary.to_csv(os.path.join(output_dir, "loocv_model_performance_summary.csv"), index=False)

# print("\n[✓] LOOCV summary saved.")
# print(df_loocv_summary)

# best_model_row = df_loocv_summary.sort_values(by="F1", ascending=False).iloc[0]
# best_model_name = best_model_row["Model"]
# print(f"\n🏆 Best LOOCV Model: {best_model_name} with F1-score = {best_model_row['F1']:.4f}")

# # 使用全部資料重新訓練最佳模型
# scaler_final = StandardScaler()
# X_final_scaled = scaler_final.fit_transform(fused_features)

# final_model_obj = models[best_model_name]
# final_model_fitted = final_model_obj.fit(X_final_scaled, categories)
# final_model = final_model_fitted.best_estimator_ if hasattr(final_model_fitted, "best_estimator_") else final_model_fitted

# # 儲存模型與 scaler
# joblib.dump(final_model, os.path.join(model_dir, f"best_loocv_model_{best_model_name.replace(' ', '_')}.joblib"))
# joblib.dump(scaler_final, os.path.join(model_dir, "scaler_loocv.joblib"))

# print(f"[✓] Best model and scaler saved to: {model_dir}")

from sklearn.model_selection import GroupKFold
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# 設定 GroupKFold 的參數
k = 5
gkf = GroupKFold(n_splits=k)

kfold_results = defaultdict(lambda: {"y_true": [], "y_pred": []})


for train_idx, test_idx in gkf.split(fused_features, categories, groups=sids):
    X_train, X_test = fused_features[train_idx], fused_features[test_idx]
    y_train, y_test = categories[train_idx], categories[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        y_pred = best_model.predict(X_test_scaled)
        kfold_results[model_name]["y_true"].extend(y_test)
        kfold_results[model_name]["y_pred"].extend(y_pred)

# 📊 統計與畫圖
summary = []
for model_name, result in kfold_results.items():
    y_true = result["y_true"]
    y_pred = result["y_pred"]

    acc = accuracy_score(y_true, y_pred)
    # prec = precision_score(y_true, y_pred, zero_division=1)
    # rec = recall_score(y_true, y_pred, zero_division=1)
    # f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    summary.append([model_name, acc, prec, rec, f1])
    labels = sorted(list(set(categories)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix (GroupKFold-{k}) - {model_name}")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_groupkfold_{model_name.replace(' ', '_')}.png"))
    plt.close()
    print(f"[✓] Confusion Matrix for {model_name} saved.")

# 儲存成 DataFrame 與 CSV
df_kfold_summary = pd.DataFrame(summary, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
df_kfold_summary.to_csv(os.path.join(output_dir, f"groupkfold_model_performance_summary.csv"), index=False)

print("\n[✓] GroupKFold summary saved.")
print(df_kfold_summary)

best_model_row = df_kfold_summary.sort_values(by="F1", ascending=False).iloc[0]
best_model_name = best_model_row["Model"]
print(f"\n🏆 Best GroupKFold Model: {best_model_name} with F1-score = {best_model_row['F1']:.4f}")

# 使用全部資料重新訓練最佳模型
scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(fused_features)

final_model_obj = models[best_model_name]
final_model_fitted = final_model_obj.fit(X_final_scaled, categories)
final_model = final_model_fitted.best_estimator_ if hasattr(final_model_fitted, "best_estimator_") else final_model_fitted

# 儲存模型與 scaler
joblib.dump(final_model, os.path.join(model_dir, f"best_groupkfold_model_{best_model_name.replace(' ', '_')}.joblib"))
joblib.dump(scaler_final, os.path.join(model_dir, "scaler_groupkfold.joblib"))

print(f"[✓] Best model and scaler saved to: {model_dir}")
