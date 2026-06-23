# import pandas as pd

# # 讀取兩個 CSV 檔
# df_text = pd.read_csv('output.csv')               # 包含 sid, context_text, value
# df_label = pd.read_csv('output_label.csv')        # 包含 Sid, Value

# # 將欄位名稱統一為相同（方便合併）
# df_label.rename(columns={'sid': 'sid', 'value': 'new_value'}, inplace=True)

# # 合併兩個資料表，根據 sid 對應
# df_merged = df_text.merge(df_label, on='sid', how='left')

# # 用新的 value 替換原本的 value 欄位
# df_merged['value'] = df_merged['new_value']

# # 丟掉暫時的 new_value 欄位
# df_merged.drop(columns=['new_value'], inplace=True)

# # 存成新的 CSV 檔案
# df_merged.to_csv('output_updated.csv', index=False)

# print("更新完成，檔案已存為 output_updated.csv")

import pandas as pd

# 讀取檔案
df = pd.read_csv('output_0704_redefined.csv')

# 移除 value 為 -1 的行
df_filtered = df[df["value"] != -1]

# 計算每個 sid 的平均值
df_avg = df_filtered.groupby("sid")["value"].mean().reset_index()

# 四捨五入到整數
df_avg["rounded_value"] = df_avg["value"].round()

# 計算每個 sid 最多數的類別（計數最多的值）
def get_most_frequent(x):
    counts = x.value_counts()
    return counts.idxmax()  # 取得出現最多的數字

# 使用 value_counts() 計算眾數
df_mode = df_filtered.groupby("sid")["value"].agg(get_most_frequent).reset_index()
df_mode.rename(columns={"value": "most_frequent_value"}, inplace=True)

# 合併平均值和眾數
df_final = pd.merge(df_avg, df_mode, on="sid")

# 顯示結果
print(df_final)

# 若要保存到新檔案，可以使用以下程式碼：
df_final.to_csv('output_with_averages_and_modes.csv', index=False)
