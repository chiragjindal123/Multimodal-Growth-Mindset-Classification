import os
import numpy as np
import pandas as pd

def count_all_modalities_per_sid(root_dir: str):
    """
    統計每個 sid 的五個 feature 長度，並檢查是否一致
    回傳 DataFrame，包含：sid, text, audio, face, hrv, eye, is_consistent
    """
    rows = []
    modalities = ['text', 'audio', 'face', 'hrv', 'eye']

    for sid in os.listdir(root_dir):
        sid_path = os.path.join(root_dir, sid)
        if not os.path.isdir(sid_path):
            continue

        feature_lengths = {}
        valid = True

        for m in modalities:
            npy_file = os.path.join(sid_path, f"x_split_{m}.npy")
            if not os.path.exists(npy_file):
                print(f"[!] 缺少 {m} feature in {sid}")
                valid = False
                feature_lengths[m] = -1
                continue

            data = np.load(npy_file, allow_pickle=True)
            feature_lengths[m] = len(data)

        is_consistent = len(set(feature_lengths.values())) == 1
        rows.append({
            'sid': sid,
            **feature_lengths,
            'is_consistent': is_consistent
        })

    df = pd.DataFrame(rows).sort_values("sid").reset_index(drop=True)
    return df


if __name__ == "__main__":
    root = "newlabel/"   # <<< 換成你的資料夾
    df_check = count_all_modalities_per_sid(root)

    print(df_check)
    inconsistent = df_check[~df_check['is_consistent']]
    if not inconsistent.empty:
        print("\n[!] 以下 sid 的 feature 長度不一致：")
        print(inconsistent)
    else:
        print("\n✅ 所有 sid 的五個 feature 長度一致")

    # 若需要存檔：
    df_check.to_csv("feature_modalities_check.csv", index=False, encoding="utf-8-sig")
