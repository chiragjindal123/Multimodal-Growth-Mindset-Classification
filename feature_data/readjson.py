import json

# 載入 JSON 檔案
with open('interview_all_export_with_sentence_0612.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# print(len(data[0]['feature']))
# print(data[0]['feature'][2]['eye_movement'])
# print(data[0]['feature'][2]['heart_rate_variability'])

total = 0
# 遍歷資料
print(len(data))
for item in data:
    sid = item.get('sid')
    print(f"SID: {sid}, {len(item['feature'])}")
    total += len(item['feature'])
    # features = item.get('feature', [])
    # for feature in features:
    #     context = feature.get('_context', {})
    #     context_text = context.get('text')
    #     print(f"  Context Text: {context_text}")
    # print("-" * 50)

print(f"total: {total}")