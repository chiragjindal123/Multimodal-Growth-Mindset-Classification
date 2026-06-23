import os
import shutil
import ujson
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm


facial_time_series = 120

class preprocessing:

    label_path = 'json_data/new_factor_questnaire_label.csv'
    target_dir = 'new_questionaire'
    raw_data_path = 'json_data/new_factor_questnaire_normal.json'
    feature_list = ['text', 'audio', 'face', 'heart_rate_variability', 'eye_movement']

    def __init__(self, raw_data_path:str, targetPath:str, labelPath: str):
        self.raw_data_path = raw_data_path
        self.target_dir = targetPath
        self.label_path = labelPath

    '''
    @load_normal_json: 
    @filepath: 設定儲存 feature 的路徑
    
    只吃 json 檔案格式為:
    {
        _id: 
        sid:
        feature:
        {
            0:
                _context
                _cal_time:
                text:
                {
                    ...
                    ...
                }
                audio_id:
                audio:
                {
                    ...
                }
                ....
            1:
            ...:
        }
        ...
    }
    (細節參照: 140.115.54.68:27017 features document 文件格式)
    '''

    def load_json_into_numpy(self, filepath: str):
        
        if filepath != '':
            self.raw_data_path = filepath
        
        label_file = pd.read_csv(self.label_path)
        print(label_file)

        def categorize(score):
            if 1 <= score <= 1.9:
                return 0
            elif 2 <= score <= 2.9:
                return 1
            elif 3 <= score <= 3.9:
                return 2
            elif 4 <= score <= 5:
                return 3
            return -1  # 若不在範圍內可忽略
        
        with open(self.raw_data_path, encoding='big5') as f:
            data = ujson.load(f)
            
        for record in tqdm(data):
            sender_id = record['sid']
            x_text = []
            x_audio = []
            x_face = []
            x_eye = []
            x_hrv = []
            
            for key, question in tqdm(enumerate(record['feature'])):
                # 0, 1, 2, ...
                if all(feature_name in question for feature_name in self.feature_list) and \
                        question['text'] and \
                            question['audio'] and \
                                question['face'] and \
                                    question['heart_rate_variability'] and \
                                        question['eye_movement']:
                                        
                        x_text.append(np.array(question['text']))
                        x_audio.append(np.array(question['audio']))
                        
                        
                        # model 的 input 是 (100, 256)，將所有的 face feature truncate or padding 成 (100, 256)
                        
                        length = len(question['face'])
                        if length < facial_time_series:
                            length = facial_time_series - length
                            for i in range(length):
                                question['face'].append(np.zeros(256))
                                
                        elif length > facial_time_series:
                            question['face'] = question['face'][:facial_time_series]
                        
                        x_face.append(np.array(question['face']))
                        x_hrv.append(np.array(question['heart_rate_variability']))
                        x_eye.append(np.array(question['eye_movement']))
                # else:
                #     print(f'{key} is not contains all feautures')
            
            
            # print("data: ", np.asarray(x_face).shape)
                    
            current_subject_save_path = self.target_dir + f"{sender_id}"
            
            
            if os.path.exists(current_subject_save_path):
                shutil.rmtree(current_subject_save_path)
            
            if sender_id in label_file['sid'].values:   
                for data in [x_text, x_audio, x_face, x_hrv, x_eye]:
                    for i in range(len(data)):
                        for j in range(len(data[i])):
                            if isinstance(data[i][j], dict):
                                data[i][j] = 0
                
                print(sender_id, np.array(x_text).shape)
                label_value = label_file.loc[label_file['sid'] == sender_id]['value'].values[0]
                # label_category = categorize(label_value)
                label_category = label_value
                if label_category == -1:
                    print(f"[!] Invalid score for SID {sender_id}, skipped.")
                    continue

                os.mkdir(current_subject_save_path)
                np.save(current_subject_save_path + f"/x_split_text", x_text, allow_pickle=True)
                np.save(current_subject_save_path + f"/x_split_audio", x_audio, allow_pickle=True)
                np.save(current_subject_save_path + f"/x_split_face", x_face, allow_pickle=True)
                np.save(current_subject_save_path + f"/x_split_hrv", x_hrv, allow_pickle=True)
                np.save(current_subject_save_path + f"/x_split_eye", x_eye, allow_pickle=True)
                np.save(current_subject_save_path + f"/y_label", label_category, allow_pickle=True)
                # print(label_file.loc[label_file['id'] == sender_id]['value'].values[0])
            else:
                print(f'Key {sender_id} doesn\'t exist')

    def load_json_into_numpy_redefined(self, filepath: str):

        if filepath != '':
            self.raw_data_path = filepath

        label_file = pd.read_csv(self.label_path)

        with open(self.raw_data_path, encoding='big5') as f:
            data = ujson.load(f)

        # 建立精確的 label 對應表: (sid, context) -> value
        label_dict = {}
        for _, row in label_file.iterrows():
            sid = str(row['sid']).strip()
            context = str(row['context_text']).strip()
            value = row['value']
            # if value == -1:
            #     continue  # 無效 label 略過
            label_dict[(sid, context)] = value

        for record in tqdm(data):
            sender_id = str(record['sid']).strip()
            features = record['feature']

            for idx, question in enumerate(features):
                context = str(question.get('_context', {}).get('text', '')).strip()
                label_key = (sender_id, context)

                if label_key not in label_dict:
                    print(f"label_key not in label_dict - {label_key}")
                    continue  # 該句子無對應有效 label，跳過

                if not all(f in question and question[f] for f in self.feature_list):
                    print("loss features")
                    continue  # 缺少必要特徵，跳過

                label = label_dict[label_key]

                # Prepare features
                x_text = np.array(question['text'])
                x_audio = np.array(question['audio'])

                face_data = question['face']
                if len(face_data) < facial_time_series:
                    face_data += [np.zeros(256)] * (facial_time_series - len(face_data))
                else:
                    face_data = face_data[:facial_time_series]
                x_face = np.array(face_data)

                x_hrv = np.array(question['heart_rate_variability'])
                x_eye = np.array(question['eye_movement'])

                # Save per sample
                save_path = os.path.join(self.target_dir, f"{sender_id}_{idx}")
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                os.makedirs(save_path)

                np.save(os.path.join(save_path, "x_split_text.npy"), x_text, allow_pickle=True)
                np.save(os.path.join(save_path, "x_split_audio.npy"), x_audio, allow_pickle=True)
                np.save(os.path.join(save_path, "x_split_face.npy"), x_face, allow_pickle=True)
                np.save(os.path.join(save_path, "x_split_hrv.npy"), x_hrv, allow_pickle=True)
                np.save(os.path.join(save_path, "x_split_eye.npy"), x_eye, allow_pickle=True)
                np.save(os.path.join(save_path, "y_label.npy"), label, allow_pickle=True)


if __name__ == '__main__':
    
    '''
    preprocessing features data in json_data directory.
    remark: data will generate in different date if you preprocess twice or more.
    '''
    encoder = preprocessing(raw_data_path='interview_all_export_with_sentence_0626.json', 
                            targetPath='redefined_sentence/',
                            labelPath='output_0704_redefined.csv')
    
    encoder.load_json_into_numpy_redefined(filepath=encoder.raw_data_path)
    # encoder.load_json_into_numpy(filepath=encoder.raw_data_path)
    