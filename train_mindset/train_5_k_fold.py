# using pytoch env

import os
import shutil
import ujson
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import itertools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
import statistics
from sklearn.metrics import f1_score, precision_score, recall_score

#from imblearn.under_sampling import RandomUnderSampler

NEW_DATASET_BINARY = 1
NEW_DATASET_TRIPLE = 8
NEW_DATASET = 7
OLD_DATASET_MATURE = 2
OLD_DATASET_TEENS = 3
OLD_DATASET_HYBIRD = 4
HYBIRD_DATASET = 5
OLD_QUESTIONAIRE_TEENS_DATASET_PAPER_VERSION = 6

# dataset_version = NEW_DATASET_BINARY
dataset_version = NEW_DATASET_BINARY
# isPredict = True
isPredict = False
predictModel = "/data/train_mindset/___Experiments___0716/2025_07_20_09_53_2462203_sentence_sid_0&1_2分類__FiveFusionModel2_Att_epochs=30_/model.pkl"
# "2023_06_17_0189696_old_factor_teens_5分類__FiveFusionModel2_Att_epoch=100"
# "2023_06_27_0527830_new_factor_triple_3分類__FiveFusionModel2_Att_epochs=10_with_attetnion_論文數據"
    
binary_classification = False
three_classification = False
four_classification = False   
    
facial_time_series = 8

# ---------------------------- 定義超參數 ----------------------------
epochs = 30
batch_size = 1
LR = 0.0001
optimizerName = "Adam"  # Adam, RMSprop, AdamDefault
loss_fn = torch.nn.CrossEntropyLoss()
oversampling = False
undersampling = False

useShuffleSplit = False
test_split = 0.25
shuffle_dataset = True
random.seed(12)
random_seed = 34

# get current time
nowTime = int(time.time())
struct_time = time.localtime(nowTime)
rand_number =  random.randint(1,100000)
# experimentPath = '/data/suicide/___Experiments___'
experimentPath = '/data/train_mindset/___Experiments___0716'
timeStringOnly = time.strftime("%Y_%m_%d_%H_%M_%S", struct_time)
timeString = timeStringOnly + str(rand_number) + '_'
torch.autograd.set_detect_anomaly(True)
    
if dataset_version == NEW_DATASET_BINARY:
    timeString += 'sentence_sid_0&1'    
    root_dirs = ['/data/feature_data/newlabel_sentence/']
    # root_dirs = ['/data/feature_data/redefined_sentence/']
    # root_dirs = ['/data/feature_data/newlabel/']
    binary_classification = True
    CATEGORY = 2
elif dataset_version == NEW_DATASET_TRIPLE:
    timeString += 'new_factor_triple'
    # root_dirs = ['/root/MatureDepressionAssessment/ru_test/new_questionaire_new_eye/']
    root_dirs = ['/data/feature_data/98data_sentence/']
    three_classification = True
    CATEGORY = 3
elif dataset_version == OLD_DATASET_MATURE:
    timeString += 'old_factor_mature'
    root_dirs = [ 'old_questionaire_mature/']
    binary_classification = True
    CATEGORY = 2
elif dataset_version == OLD_DATASET_TEENS:
    timeString += 'old_factor_teens'
    root_dirs = ['old_questionaire_teens/']
    if three_classification:
        CATEGORY = 3
    elif four_classification:
        CATEGORY = 4
    else:
        CATEGORY = 5
elif dataset_version == NEW_DATASET:
    timeString += 'people'
    root_dirs =  ['/data/feature_data/newlabel/']
    four_classification = True   
    CATEGORY = 4
elif dataset_version == OLD_DATASET_HYBIRD:
    timeString += 'old_factor_hybird'
    root_dirs = [ 'old_questionaire_mature/', 'old_questionaire_teens/',]
    CATEGORY = 5
elif dataset_version == OLD_QUESTIONAIRE_TEENS_DATASET_PAPER_VERSION:
    timeString += 'old_questionaire_teens_dataset_paper_version'
    CATEGORY = 5
else:
    timeString += 'hybird'
    root_dirs = ['new_questionaire/', 'old_questionaire_mature/', 'old_questionaire_teens/']    
    CATEGORY = 5

# root_dirs = ['old_questionaire_teens_14_face']
if oversampling:
    timeString +='_oversampling'
if undersampling:
    timeString +='_undersampling'

timeString += f'_{CATEGORY}分類_'
test_loss_record = []
test_accuracy_record = []


                                        
def accuracy(predictions, targets):
    _, predicted_labels = torch.max(predictions, dim=1)
    correct = (predicted_labels == targets).sum().item()
    total = targets.size(0)
    acc = correct / total
    return acc

def fix_None(data, fix:bool):
    if fix: # fix none and nan
        data = np.array(data, dtype=float)
        nan_indices = np.isnan(data).any(axis=1)
        data[nan_indices] = 0
        return data
    else:   # only check
        print(((data == None).any()))
        return data

class QuestionDatasetWithSID(Dataset):

    X_text, X_audio, X_face, X_hrv, X_eye_movement = [], [], [], [], []
    Y_label = []
    X_sid = []
    y_label_qun = [0 for i in range(CATEGORY)]
    sample_quntity = [0 for _ in range(CATEGORY)]
    
    # data loading
    def __init__(self):
        
        max_len = 0
        valid_sid_set = set()
        # 讀取 CSV 並篩選
        # df = pd.read_csv('/data/feature_data/output_with_averages_and_modes.csv')
        # valid_sid_set = set(df[df['most_frequent_value'].isin([2, 3])]['sid'].str.strip())
        for root_dir in root_dirs:
            for folder in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder)
            
                if os.path.isdir(folder_path):

                    sid = os.path.basename(folder_path)
                    # sid = sid.split('_')[0]
                    # if sid not in valid_sid_set:
                    #     continue
                    # print(os.path.join(folder_path, 'x_split_text.npy'))
                    x_text = np.load(os.path.join(folder_path, 'x_split_text.npy'), allow_pickle=True).astype(np.float32)
                    x_audio = np.load(os.path.join(folder_path, 'x_split_audio.npy'), allow_pickle=True).astype(np.float32)
                    x_face = np.load(os.path.join(folder_path, 'x_split_face.npy'), allow_pickle=True)
                    x_hrv = np.load(os.path.join(folder_path, 'x_split_hrv.npy'), allow_pickle=True)
                    x_eye_movement = np.load(os.path.join(folder_path, 'x_split_eye.npy'), allow_pickle=True).astype(np.float32)
                    y_label = np.load(os.path.join(folder_path, 'y_label.npy'), allow_pickle=True).astype(np.float32)
                    # print(y_label, y_label.shape)
                    # if int(y_label) not in [0, 1, 2, 3]:
                    #     continue
                    if int(y_label) not in [0, 1]:
                        continue

                    if binary_classification and int(y_label) > 0:
                        y_label = 1
                    elif binary_classification: y_label = 0

                    # #if three_classification and int(y_label) >= 3:
                    # if three_classification and dataset_version == NEW_DATASET_TRIPLE:
                    #     if int(y_label) > 2:
                    #         y_label = 2

                    self.sample_quntity[int(y_label)] += 1
                                    
                    target_shape = (facial_time_series, 256)
                    
                    ''' redefined sentence '''
                    # x_text = np.nan_to_num(np.array(x_text)).astype(np.float32)
                    # x_audio = np.nan_to_num(np.array(x_audio)).astype(np.float32)

                    # # Face padding
                    # if len(x_face) < facial_time_series:
                    #     x_face = np.pad(x_face, ((0, facial_time_series - len(x_face)), (0, 0)), mode='constant')
                    # else:
                    #     x_face = x_face[:facial_time_series]
                    # x_face = np.nan_to_num(np.array(x_face)).astype(np.float32)

                    # x_hrv = np.nan_to_num(np.array(x_hrv)).astype(np.float32)
                    # x_eye_movement = np.nan_to_num(np.array(x_eye_movement)).astype(np.float32)

                    # # Append to dataset
                    # self.X_text.append(x_text)
                    # self.X_audio.append(x_audio)
                    # self.X_face.append(x_face)
                    # self.X_hrv.append(x_hrv)
                    # self.X_eye_movement.append(x_eye_movement)
                    # self.Y_label.append(int(y_label))
                    # self.X_sid.append(sid)
                    # self.y_label_qun[int(y_label)] += 1

                    ''' other data '''
                    for i, x in enumerate(x_face):
                        # x = np.array(x)
                        
                        if len(x) > facial_time_series:
                            # x_face[i] = np.array(x_face[i])
                            if isinstance(x_face[i], np.ndarray): 
                                x_face = x_face.tolist()
                            if isinstance(x, np.ndarray):
                                x = x.tolist()
                            # print(type(x_face[i]), type(x[:14]))
                            x_face[i] = x[:facial_time_series]
                            
                        else:
                            x = np.array(x)
                            # 計算要填充的形狀
                            padding = [(0, max(0, target_shape[j] - x.shape[j])) for j in range(len(target_shape))]
                            # 填充數組
                            x_padded = np.pad(x, padding, mode='constant')
                            # 取出填充後的前 14 行，裁剪多餘的部分
                            if isinstance(x_face[i], np.ndarray): 
                                x_face = x_face.tolist()
                                
                            x_face[i] = x_padded[:target_shape[0], :]
                        
                        x_text[i] = np.nan_to_num(np.array(x_text[i]))
                        x_audio[i] = np.nan_to_num(np.array(x_audio[i]))
                        x_face[i] = np.nan_to_num(np.array((x_face[i])))
                        x_hrv[i] = np.nan_to_num(np.array((x_hrv[i])))
                        x_eye_movement[i] = np.nan_to_num(np.array((x_eye_movement[i])))
                        
                        self.X_text.append(x_text[i].astype(np.float32))
                        self.X_audio.append(x_audio[i].astype(np.float32))
                        self.X_face.append(x_face[i].astype(np.float32))
                        self.X_hrv.append(x_hrv[i].astype(np.float32))
                        self.X_eye_movement.append(x_eye_movement[i].astype(np.float32))
                        
                        self.Y_label.append(int(y_label))
                        self.X_sid.append(sid)
                        self.y_label_qun[int(y_label)] += 1
        
        
        # Replace Nan with 0
        self.X_text = np.nan_to_num(self.X_text)
        self.X_audio = np.nan_to_num(self.X_audio)
        self.X_face  = np.nan_to_num(self.X_face)
        self.X_eye_movement  = np.nan_to_num(self.X_eye_movement)
        self.X_hrv  = np.nan_to_num(self.X_hrv)
        
        self.X_text = np.array(self.X_text)
        self.X_audio = np.array(self.X_audio)
        self.X_face = np.array(self.X_face)
        self.X_hrv = np.array(self.X_hrv)
        self.X_eye_movement = np.array(self.X_eye_movement)
  
        # print(2 in self.Y_label)
        # print('y_label quentity: ', len(self.Y_label))
        # print(np.array([X_text, X_audio, X_face, X_hrv, X_eye_movement, Y]).shape[1])
        # print(self.X_sid)
        self.n_samples = len(self.X_text)
        
    # working for indexing
    def __getitem__(self, index):
        return self.X_sid[index], self.X_text[index], self.X_audio[index], self.X_face[index], self.X_hrv[index], \
            self.X_eye_movement[index],self.Y_label[index]
    
    # return the length of our dataset
    def __len__(self):
        return self.n_samples


class SequencePerSubjectDataset(Dataset):
    def __init__(self):
        self.data_by_sid = {}
        self.sids = []

        def safe_array(x, shape=None, fill=0.0, dtype=np.float32):
            try:
                x = np.array(x, dtype=dtype)
                if shape and x.shape != shape:
                    x = np.full(shape, fill, dtype=dtype)
            except:
                x = np.full(shape, fill, dtype=dtype)
            return np.nan_to_num(x)

        for root_dir in root_dirs:
            for folder in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder)
                if not os.path.isdir(folder_path):
                    continue

                sid = os.path.basename(folder_path).split('_')[0]

                try:
                    x_text = np.load(os.path.join(folder_path, 'x_split_text.npy'), allow_pickle=True)
                    x_audio = np.load(os.path.join(folder_path, 'x_split_audio.npy'), allow_pickle=True)
                    x_face = np.load(os.path.join(folder_path, 'x_split_face.npy'), allow_pickle=True)
                    x_hrv = np.load(os.path.join(folder_path, 'x_split_hrv.npy'), allow_pickle=True)
                    x_eye = np.load(os.path.join(folder_path, 'x_split_eye.npy'), allow_pickle=True)
                    y_label = np.load(os.path.join(folder_path, 'y_label.npy'), allow_pickle=True)
                except Exception as e:
                    print(f"Error loading {folder_path}: {e}")
                    continue
                
                seq_len = min(len(x_text), len(x_audio), len(x_face))
                if seq_len == 0:
                    print(f"[跳過] {sid} 的模態序列為空，已移除")
                    continue

                if binary_classification and int(y_label) > 1:
                    y_label = 1
                else: y_label = 0
                if three_classification and int(y_label) >= 3:
                    y_label = 2

                if sid not in self.data_by_sid:
                    self.data_by_sid[sid] = {
                        "text": [], "audio": [], "face": [], "hrv": [], "eye": [], "label": int(y_label)
                    }
                    self.sids.append(sid)  # ✅ 加入 subject id

                entry = self.data_by_sid[sid]

                for i in range(seq_len):
                    entry["text"].append(safe_array(x_text[i], shape=(768,)))
                    entry["audio"].append(safe_array(x_audio[i], shape=(193,)))
                    entry["face"].append(safe_array(x_face[i], shape=(8, 256)))  # (8幀, 256維)
                
                entry["hrv"] = safe_array(x_hrv, shape=(23,))
                entry["eye"] = safe_array(x_eye, shape=(13,))
                entry["label"] = int(y_label)

    def __getitem__(self, idx):
        sid  = self.sids[idx]
        item = self.data_by_sid[sid]
        T = len(item["text"])

        face_np = np.stack(item["face"], axis=0)  # (T, 8, 256)
        face_tensor = torch.from_numpy(face_np).float().mean(dim=1)  # (T, 256)

        return (
            sid,
            torch.from_numpy(np.stack(item["text"])).float(),   # (T, 768)
            torch.from_numpy(np.stack(item["audio"])).float(),  # (T, 193)
            face_tensor,                                        # (T, 2048)
            torch.from_numpy(item["hrv"]).float(),              # (23,)
            torch.from_numpy(item["eye"]).float(),              # (13,)
            torch.tensor(item["label"], dtype=torch.long),      # Label
            T                                                    # 句數
        )

    def __len__(self):
        return len(self.sids)

def collate_fn(batch):
    sids, texts, audios, faces, hrvs, eyes, labels, lengths = zip(*batch)
    max_len = max(lengths)

    def pad_tensor(seq_list, max_len):
        return torch.stack([
            F.pad(seq, (0, 0, 0, max_len - seq.shape[0])) for seq in seq_list
        ])

    return (
        list(sids),
        pad_tensor(texts, max_len),   # (B, max_T, 768)
        pad_tensor(audios, max_len),  # (B, max_T, 193)
        pad_tensor(faces, max_len),   # (B, max_T, 2048)
        torch.stack(hrvs),            # (B, 23)
        torch.stack(eyes),            # (B, 13)
        torch.tensor(labels),         # (B,)
        torch.tensor(lengths)         # (B,)
    )


class FiveFusionModel2_Five_Att(nn.Module):
    
    
    def __init__(self, n_text, n_audio, n_face, n_hrv, n_eye, lstm_hiddens, \
            text_lstm_layers, audio_lstm_layers, face_lstm_layers, hrv_lstm_layers, eye_lstm_layers, \
                n_output):
        
        super(FiveFusionModel2_Five_Att, self).__init__()
        
        self.text_lstm = nn.LSTM(input_size=n_text,
                            hidden_size=lstm_hiddens[0],
                            num_layers=text_lstm_layers,
                            bidirectional=True, )
        
        self.audio_lstm = nn.LSTM(input_size=n_audio,
                            hidden_size=lstm_hiddens[1],
                            num_layers=audio_lstm_layers,
                            bidirectional=True, )
        
        self.face_lstm = nn.LSTM(input_size=n_face,
                    hidden_size=lstm_hiddens[2],
                    num_layers=face_lstm_layers,
                    bidirectional=True, )
        
        self.hrv_lstm = nn.LSTM(input_size=n_hrv,
                    hidden_size=lstm_hiddens[3],
                    num_layers=hrv_lstm_layers,
                    bidirectional=True, )
        
        self.eye_lstm = nn.LSTM(input_size=n_eye,
                            hidden_size=lstm_hiddens[4],
                            num_layers=eye_lstm_layers,
                            bidirectional=True, )
        
        # self.linear1 = nn.Linear(sum(lstm_hiddens) * 2, n_output)
        
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(sum(lstm_hiddens) * 2, 1))
        self.w1 = nn.Parameter(torch.zeros(lstm_hiddens[0]* 2, 1))
        self.w2 = nn.Parameter(torch.zeros(lstm_hiddens[1]* 2, 1))
        self.w3 = nn.Parameter(torch.zeros(lstm_hiddens[2]* 2, 1))
        self.w4 = nn.Parameter(torch.zeros(lstm_hiddens[3]* 2, 1))
        self.w5 = nn.Parameter(torch.zeros(lstm_hiddens[4]* 2, 1))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(sum(lstm_hiddens) * 2, 64)
        self.fc = nn.Linear(64, n_output)
        
        # 对权重和偏置进行初始化
        # torch.nn.init.xavier_uniform_(self.linear1.weight)
        #3 torch.nn.init.zeros_(self.linear1.bias)
        
        for lstm in [self.text_lstm, self.audio_lstm,  self.face_lstm, self.hrv_lstm, self.eye_lstm]:
            for name, param in lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_text, x_audio, x_face, x_hrv, x_eye):
        
        x_text, _ = self.text_lstm(x_text)
        
        M1 = self.tanh1(x_text)  # [128, 32, 256]
        alpha1 = F.softmax(torch.matmul(M1, self.w1), dim=1).unsqueeze(-1)  # [128, 32, 1]
        x_text = x_text * alpha1  # [128, 32, 256]
        x_text = torch.sum(x_text, 1)  # [128, 256]
        
        
        x_audio, _ = self.audio_lstm(x_audio)
        M2 = self.tanh1(x_audio)  # [128, 32, 256]
        alpha2 = F.softmax(torch.matmul(M2, self.w2), dim=1).unsqueeze(-1)  # [128, 32, 1]
        x_audio = x_audio * alpha2  # [128, 32, 256]
        x_audio = torch.sum(x_audio, 1)  # [128, 256]
        
        
        x_face, _ = self.face_lstm(x_face)
        M3 = self.tanh1(x_face)  # [128, 32, 256]
        alpha3 = F.softmax(torch.matmul(M3, self.w3), dim=1).unsqueeze(-1)  # [128, 32, 1]
        x_face = x_face * alpha3  # [128, 32, 256]
        x_face = torch.sum(x_face, 1)  # [128, 256]
        
        x_hrv, _ = self.hrv_lstm(x_hrv)
        M4 = self.tanh1(x_hrv)  # [128, 32, 256]
        alpha4 = F.softmax(torch.matmul(M4, self.w4), dim=1).unsqueeze(-1)  # [128, 32, 1]
        x_hrv = x_hrv * alpha4  # [128, 32, 256]
        x_hrv = torch.sum(x_hrv, 1)  # [128, 256]
        
        x_eye, _ = self.eye_lstm(x_eye)
        M5 = self.tanh1(x_eye)  # [128, 32, 256]
        alpha5 = F.softmax(torch.matmul(M5, self.w5), dim=1).unsqueeze(-1)  # [128, 32, 1]
        x_eye = x_eye * alpha5  # [128, 32, 256]
        x_eye = torch.sum(x_eye, 1)  # [128, 256]
        
        encoding = torch.cat((x_text, x_audio, x_face, x_hrv, x_eye), dim=1)
        
        # print(encoding.shape)
        M = self.tanh1(encoding)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        # print('encoding shape: ', encoding.shape)
        out = encoding * alpha  # [128, 32, 256]
        # print('out shape: ', out.shape)
        out = torch.sum(out, 1)  # [128, 256]
        # print('out shape: ', out.shape)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        
        
        return out        

class FiveFusionModel_Att_sid(nn.Module):

    def __init__(
        self,
        n_text=768, n_audio=193, n_face=2048, n_hrv=23, n_eye=13,        
        lstm_hiddens=(256, 128, 128, 64, 32),                          
        text_lstm_layers=1, audio_lstm_layers=1,
        face_lstm_layers=1, hrv_lstm_layers=1, eye_lstm_layers=1,
        n_output=3, dropout_p=0.3
    ):
        super().__init__()

        # ─── 四個 Bi-LSTM ─────────────────────────────────────────────
        self.text_lstm  = nn.LSTM(n_text,  lstm_hiddens[0],
                                  text_lstm_layers,  bidirectional=True, batch_first=True)
        self.audio_lstm = nn.LSTM(n_audio, lstm_hiddens[1],
                                  audio_lstm_layers, bidirectional=True, batch_first=True)
        self.face_lstm  = nn.LSTM(n_face,  lstm_hiddens[2],
                                  face_lstm_layers,  bidirectional=True, batch_first=True)
        self.hrv_lstm   = nn.LSTM(n_hrv,   lstm_hiddens[3],
                                  hrv_lstm_layers,   bidirectional=True, batch_first=True)
        self.eye_lstm   = nn.LSTM(n_eye,   lstm_hiddens[4],
                                  eye_lstm_layers,   bidirectional=True, batch_first=True)

        # ─── 正規化 & Dropout ────────────────────────────────────────
        self.dropout   = nn.Dropout(dropout_p)
        self.F_total   = 2 * sum(lstm_hiddens)              # 2 × (256+128+128+64) = 1152
        self.layernorm = nn.LayerNorm(self.F_total)

        # ─── Self-Attention 向量 ──────────────────────────────────────
        self.tanh   = nn.Tanh()
        self.w_att  = nn.Parameter(torch.zeros(self.F_total, 1))

        # ─── 分類器 ──────────────────────────────────────────────────
        self.fc1 = nn.Linear(self.F_total, 64)
        self.fc2 = nn.Linear(64, n_output)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, x_text, x_audio, x_face, x_hrv, x_eye):
        # 1. Bi-LSTM
        x_text,  _ = self.text_lstm(x_text)
        x_audio, _ = self.audio_lstm(x_audio)
        x_face,  _ = self.face_lstm(x_face)
        x_hrv,   _ = self.hrv_lstm(x_hrv)
        x_eye,   _ = self.eye_lstm(x_eye)

        # 2. Dropout
        x_text  = self.dropout(x_text)
        x_audio = self.dropout(x_audio)
        x_face  = self.dropout(x_face)
        x_hrv   = self.dropout(x_hrv)
        x_eye   = self.dropout(x_eye)

        # 3. 時序拼接 (B, seq, F_total)
        encoding = torch.cat((x_text, x_audio, x_face, x_hrv, x_eye), dim=2)
        encoding = self.layernorm(encoding)

        # 4. Self-Attention over time
        M     = self.tanh(encoding)                          # (B, seq, F_total)
        alpha = torch.softmax(torch.matmul(M, self.w_att), dim=1)  # (B, seq, 1)
        out   = torch.sum(encoding * alpha, dim=1)           # (B, F_total)

        # 5. 分類
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

class FiveFusionModel2_Att(nn.Module):
    
    
    def __init__(self, n_text, n_audio, n_face, n_hrv, n_eye, lstm_hiddens, \
            text_lstm_layers, audio_lstm_layers, face_lstm_layers, hrv_lstm_layers, eye_lstm_layers, \
                n_output):
        
        super(FiveFusionModel2_Att, self).__init__()
        
        self.text_lstm = nn.LSTM(input_size=n_text,
                            hidden_size=lstm_hiddens[0],
                            num_layers=text_lstm_layers,
                            bidirectional=True, )
        
        self.audio_lstm = nn.LSTM(input_size=n_audio,
                            hidden_size=lstm_hiddens[1],
                            num_layers=audio_lstm_layers,
                            bidirectional=True, )
        
        self.face_lstm = nn.LSTM(input_size=n_face,
                    hidden_size=lstm_hiddens[2],
                    num_layers=face_lstm_layers,
                    bidirectional=True, )
        
        self.hrv_lstm = nn.LSTM(input_size=n_hrv,
                    hidden_size=lstm_hiddens[3],
                    num_layers=hrv_lstm_layers,
                    bidirectional=True, )
        
        self.eye_lstm = nn.LSTM(input_size=n_eye,
                            hidden_size=lstm_hiddens[4],
                            num_layers=eye_lstm_layers,
                            bidirectional=True, )
        
        # self.linear1 = nn.Linear(sum(lstm_hiddens) * 2, n_output)
        
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(sum(lstm_hiddens) * 2, 1))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(sum(lstm_hiddens) * 2, 64)
        self.fc = nn.Linear(64, n_output)
        
        # 对权重和偏置进行初始化
        # torch.nn.init.xavier_uniform_(self.linear1.weight)
        # torch.nn.init.zeros_(self.linear1.bias)
        
        for lstm in [self.text_lstm, self.audio_lstm,  self.face_lstm, self.hrv_lstm, self.eye_lstm]:
            for name, param in lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_text, x_audio, x_face, x_hrv, x_eye):
        
        x_text, _ = self.text_lstm(x_text)
        x_audio, _ = self.audio_lstm(x_audio)
        x_face, _ = self.face_lstm(x_face)
        x_hrv, _ = self.hrv_lstm(x_hrv)
        x_eye, _ = self.eye_lstm(x_eye)
        
        
        # print(x_face.shape)A
        encoding = torch.cat((x_text, x_audio, x_face, x_hrv, x_eye), dim=1)
        # encoding = self.linear1(encoding)
        # outputs = F.softmax(encoding, dim=1)
        
        M = self.tanh1(encoding)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = encoding * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [1, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        
        
        return out        
   
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
       
class FiveFusionModel_Mutihead(nn.Module):
    def __init__(self, n_text, n_audio, n_face, n_hrv, n_eye, linear_hiddens, lstm_hiddens, \
            text_lstm_layers, audio_lstm_layers, face_lstm_layers, hrv_lstm_layers, eye_lstm_layers, \
            n_output, n_head, d_model, d_k, d_v, dropout=0.1):
        
        super(FiveFusionModel_Mutihead, self).__init__()

        self.text_lstm = nn.LSTM(input_size=n_text,
                            hidden_size=lstm_hiddens[0],
                            num_layers=text_lstm_layers,
                            bidirectional=True,)
        
        self.audio_lstm = nn.LSTM(input_size=n_audio,
                            hidden_size=lstm_hiddens[1],
                            num_layers=audio_lstm_layers,
                            bidirectional=True,)
        
        self.face_lstm = nn.LSTM(input_size=n_face,
                            hidden_size=lstm_hiddens[2],
                            num_layers=face_lstm_layers,
                            bidirectional=True)
        
        self.hrv_lstm = nn.LSTM(input_size=n_hrv,
                            hidden_size=lstm_hiddens[3],
                            num_layers=hrv_lstm_layers,
                            bidirectional=True, )
        
        self.eye_lstm = nn.LSTM(input_size=n_eye,
                            hidden_size=lstm_hiddens[4],
                            num_layers=eye_lstm_layers,
                            bidirectional=True,)
        
        self.multihead_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        
        self.positionwise_feedforward = PositionwiseFeedForward(d_model, linear_hiddens[0], dropout=dropout)
        
        self.linear = nn.Linear(d_model, n_output)
        
        self.dropout = nn.Dropout(dropout)
        
        # 对权重和偏置进行初始化
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        
        for lstm in [self.text_lstm, self.audio_lstm, self.face_lstm, self.hrv_lstm, self.eye_lstm]:
            for name, param in lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
        

    def forward(self, x_text, x_audio, x_face, x_hrv, x_eye):
        
        x_text, _ = self.text_lstm(x_text)
        x_audio, _ = self.audio_lstm(x_audio)
        x_face, _ = self.face_lstm(x_face)
        x_hrv, _ = self.hrv_lstm(x_hrv)
        x_eye, _ = self.eye_lstm(x_eye)
        
        encoding = torch.cat((x_text, x_audio, x_face, x_hrv, x_eye), dim=1)
        encoding = encoding.view(-1, 2)

        encoding, _ = self.multihead_attention(encoding, encoding, encoding)
        
        encoding = self.positionwise_feedforward(encoding)
        
        encoding = self.dropout(encoding)
        
        outputs = self.linear(encoding)
        
        return outputs
   
class FiveDenseFusionModel(nn.Module):
    def __init__(self, n_text, n_audio, n_face, n_hrv, n_eye, linear_hiddens, lstm_hiddens, n_output):
        
        super(FiveDenseFusionModel, self).__init__()
        
        self.text_linear = nn.Linear(n_text, linear_hiddens[0])
        self.audio_linear = nn.Linear(n_audio, linear_hiddens[1])
        self.face_linear = nn.Linear(n_face, linear_hiddens[2])
        self.hrv_linear = nn.Linear(n_hrv, linear_hiddens[3])
        self.eye_linear = nn.Linear(n_eye, linear_hiddens[4])
        

        fusion_dim = sum(linear_hiddens)
        self.fusion_lstm1 = nn.LSTM(input_size=fusion_dim,
                            hidden_size=lstm_hiddens[0],
                            num_layers=1,
                            bidirectional=True,)
        self.fusion_lstm2 = nn.LSTM(input_size=lstm_hiddens[0]*2,
                            hidden_size=lstm_hiddens[1],
                            num_layers=1,
                            bidirectional=True,)
        
        self.fusion_linear = nn.Linear(lstm_hiddens[1]*2, n_output)
        
        self.dropout1 = nn.Dropout(p=.2)
        self.dropout2 = nn.Dropout(p=.2)
        
        # 对权重和偏置进行初始化
        torch.nn.init.xavier_uniform_(self.text_linear.weight)
        torch.nn.init.zeros_(self.text_linear.bias)
        torch.nn.init.xavier_uniform_(self.audio_linear.weight)
        torch.nn.init.zeros_(self.audio_linear.bias)
        torch.nn.init.xavier_uniform_(self.face_linear.weight)
        torch.nn.init.zeros_(self.face_linear.bias)
        torch.nn.init.xavier_uniform_(self.hrv_linear.weight)
        torch.nn.init.zeros_(self.hrv_linear.bias)
        torch.nn.init.xavier_uniform_(self.eye_linear.weight)
        torch.nn.init.zeros_(self.eye_linear.bias)
        torch.nn.init.xavier_uniform_(self.fusion_linear.weight)
        torch.nn.init.zeros_(self.fusion_linear.bias)
        
        for lstm in [self.fusion_lstm2, self.fusion_lstm2]:
            for name, param in lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
        
    def forward(self, x_text, x_audio, x_face, x_hrv, x_eye):
        
        # x_face = x_face.view(x_text.shape[0], -1)
        
        x_text = F.relu(self.text_linear(x_text))
        x_audio= F.relu(self.audio_linear(x_audio))
        x_face = F.relu(self.face_linear(x_face))
        x_hrv = F.relu(self.hrv_linear(x_hrv))
        x_eye = F.relu(self.eye_linear(x_eye))
        
        
        
        encoding = torch.cat((x_text, x_audio, x_face, x_hrv, x_eye), dim=1)
        # print(encoding.shape)
        # encoding = x_text
        encoding, _ = self.fusion_lstm1(encoding)
        encoding, _ = self.fusion_lstm2(encoding)
        encoding = F.relu(self.fusion_linear(encoding))
        outputs = F.softmax(encoding, dim=1)
        
        
        return outputs

class FivePureDenseFusionModel(nn.Module):
    def __init__(self, n_text, n_audio, n_face, n_hrv, n_eye, linear_hiddens, n_output):
        
        super(FivePureDenseFusionModel, self).__init__()
        
        self.text_linear = nn.Linear(n_text, linear_hiddens[0])
        self.audio_linear = nn.Linear(n_audio, linear_hiddens[1])
        self.face_linear = nn.Linear(n_face, linear_hiddens[2])
        self.hrv_linear = nn.Linear(n_hrv, linear_hiddens[3])
        self.eye_linear = nn.Linear(n_eye, linear_hiddens[4])
        

        fusion_dim = sum(linear_hiddens)
        
        self.fusion_linear = nn.Linear(fusion_dim, 512)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, n_output)
        
        self.dropout1 = nn.Dropout(p=.2)
        self.dropout2 = nn.Dropout(p=.2)
        
        # 对权重和偏置进行初始化
        torch.nn.init.xavier_uniform_(self.text_linear.weight)
        torch.nn.init.zeros_(self.text_linear.bias)
        torch.nn.init.xavier_uniform_(self.audio_linear.weight)
        torch.nn.init.zeros_(self.audio_linear.bias)
        torch.nn.init.xavier_uniform_(self.face_linear.weight)
        torch.nn.init.zeros_(self.face_linear.bias)
        torch.nn.init.xavier_uniform_(self.hrv_linear.weight)
        torch.nn.init.zeros_(self.hrv_linear.bias)
        torch.nn.init.xavier_uniform_(self.eye_linear.weight)
        torch.nn.init.zeros_(self.eye_linear.bias)
        torch.nn.init.xavier_uniform_(self.fusion_linear.weight)
        torch.nn.init.zeros_(self.fusion_linear.bias)
        

    def forward(self, x_text, x_audio, x_face, x_hrv, x_eye):
        
        # x_face = x_face.view(x_text.shape[0], -1)
        
        x_text = F.relu(self.text_linear(x_text))
        x_audio= F.relu(self.audio_linear(x_audio))
        x_face = F.relu(self.face_linear(x_face))
        x_hrv = F.relu(self.hrv_linear(x_hrv))
        x_eye = F.relu(self.eye_linear(x_eye))
        
        
        
        encoding = torch.cat((x_text, x_audio, x_face, x_hrv, x_eye), dim=1)
        encoding = F.relu(self.fusion_linear(encoding))
        encoding = F.relu(self.linear1(encoding))
        encoding = F.relu(self.linear2(encoding))
        outputs = F.softmax(encoding, dim=1)
        
        return outputs


class DenseLayer(nn.Module):
    def __init__(self, dim):
        super(DenseLayer, self).__init__()
        self.layer = nn.Linear(dim, dim)
        
    def forward(self, x):
        return F.relu(self.layer(x))
class LSTMLayer(nn.Module):
    def __init__(self, input, dim):
        super(LSTMLayer, self).__init__()
        self.layer = nn.LSTM(input, dim, batch_first=True)
        
    def forward(self, x):
        output, _ = self.layer(x)
        return output
    
class PaperFusionModel(nn.Module):
    def __init__(self, category):
        super(PaperFusionModel, self).__init__()
        self.text_dense = DenseLayer(771)
        self.audio_dense = DenseLayer(193)
        self.face_lstm = LSTMLayer(256*facial_time_series, 256)
        self.hrv_dense = DenseLayer(23)
        self.eye_dense = DenseLayer(13)

        self.fc1 = nn.Linear(771+193+256+23+13, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, category)

    def forward(self, x_text, x_audio, x_face, x_hrv, x_eye):
        
        text_output = self.text_dense(x_text)
        audio_output = self.audio_dense(x_audio)
        face_output = self.face_lstm(x_face)
        hrv_output = self.hrv_dense(x_hrv)
        eye_movement_output = self.eye_dense(x_eye)

        combinedInput = torch.cat((text_output, audio_output, face_output, hrv_output, eye_movement_output), dim=1)

        x = F.relu(self.fc1(combinedInput))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2)
        x = F.softmax(self.fc3(x), dim=1)

        return x
    
class FiveFusionModel2_NoAtt(nn.Module):
    def __init__(self, n_text, n_audio, n_face, n_hrv, n_eye, lstm_hiddens,
                 text_lstm_layers, audio_lstm_layers, face_lstm_layers, hrv_lstm_layers, eye_lstm_layers,
                 n_output):
        super(FiveFusionModel2_NoAtt, self).__init__()

        self.text_lstm = nn.LSTM(input_size=n_text, hidden_size=lstm_hiddens[0],
                                 num_layers=text_lstm_layers, bidirectional=True, batch_first=True)
        self.audio_lstm = nn.LSTM(input_size=n_audio, hidden_size=lstm_hiddens[1],
                                  num_layers=audio_lstm_layers, bidirectional=True, batch_first=True)
        self.face_lstm = nn.LSTM(input_size=n_face, hidden_size=lstm_hiddens[2],
                                 num_layers=face_lstm_layers, bidirectional=True, batch_first=True)
        self.hrv_lstm = nn.LSTM(input_size=n_hrv, hidden_size=lstm_hiddens[3],
                                num_layers=hrv_lstm_layers, bidirectional=True, batch_first=True)
        self.eye_lstm = nn.LSTM(input_size=n_eye, hidden_size=lstm_hiddens[4],
                                num_layers=eye_lstm_layers, bidirectional=True, batch_first=True)

        self.fc1 = nn.Linear(sum(lstm_hiddens) * 2, 64)
        self.fc2 = nn.Linear(64, n_output)

        for lstm in [self.text_lstm, self.audio_lstm, self.face_lstm, self.hrv_lstm, self.eye_lstm]:
            for name, param in lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_text, x_audio, x_face, x_hrv, x_eye):
        # 只取 LSTM 最後一個時間步的 hidden state（雙向 -> concat）
        _, (h_n_text, _) = self.text_lstm(x_text)
        _, (h_n_audio, _) = self.audio_lstm(x_audio)
        _, (h_n_face, _) = self.face_lstm(x_face)
        _, (h_n_hrv, _) = self.hrv_lstm(x_hrv)
        _, (h_n_eye, _) = self.eye_lstm(x_eye)

        # h_n 是 (num_layers * num_directions, batch, hidden_size)
        # 我們只取最後一層，將 forward 和 backward concat 起來
        def get_bidirectional_last(h_n):
            res = torch.cat((h_n[0], h_n[1]), dim=-1)
            if res.dim() == 1:
                res = res.unsqueeze(0)  # 變成 [1, hidden_size * 2]
            return res

        x_text = get_bidirectional_last(h_n_text)
        x_audio = get_bidirectional_last(h_n_audio)
        x_face = get_bidirectional_last(h_n_face)
        x_hrv = get_bidirectional_last(h_n_hrv)
        x_eye = get_bidirectional_last(h_n_eye)

        # Concatenate all features
        fused = torch.cat((x_text, x_audio, x_face, x_hrv, x_eye), dim=1)

        out = F.relu(self.fc1(fused))
        out = self.fc2(out)
        return out


class UniModalModel_Att(nn.Module):
    def __init__(self, input_size, lstm_hidden, lstm_layers, n_output):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        self.hidden_size = lstm_hidden * 2
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(self.hidden_size, 1))
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc = nn.Linear(64, n_output)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        h, _ = self.lstm(x)                       # (B, T, H)
        M = self.tanh1(h)                         # (B, T, H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # (B, T, 1)
        out = (h * alpha).sum(1)                  # (B, H)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class NoTextModel(nn.Module):
    def __init__(self,
                 n_audio, n_face, n_hrv, n_eye,
                 lstm_hiddens,  # list/tuple 長度=4
                 audio_lstm_layers, face_lstm_layers,
                 hrv_lstm_layers, eye_lstm_layers,
                 n_output):
        super().__init__()

        self.audio_lstm = nn.LSTM(n_audio, lstm_hiddens[0], audio_lstm_layers, bidirectional=True, batch_first=True)
        self.face_lstm  = nn.LSTM(n_face,  lstm_hiddens[1], face_lstm_layers,  bidirectional=True, batch_first=True)
        self.hrv_lstm   = nn.LSTM(n_hrv,   lstm_hiddens[2], hrv_lstm_layers,   bidirectional=True, batch_first=True)
        self.eye_lstm   = nn.LSTM(n_eye,   lstm_hiddens[3], eye_lstm_layers,   bidirectional=True, batch_first=True)

        total_hidden = sum(lstm_hiddens) * 2
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(total_hidden, 1))
        self.fc1 = nn.Linear(total_hidden, 64)
        self.fc  = nn.Linear(64, n_output)

        for lstm in [self.audio_lstm, self.face_lstm, self.hrv_lstm, self.eye_lstm]:
            for name, p in lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(p, 0.)
                elif 'weight' in name:
                    nn.init.xavier_normal_(p)

    def forward(self, x_audio, x_face, x_hrv, x_eye):
        # 期望各輸入 shape: (batch, seq_len, feature_dim)
        a, _ = self.audio_lstm(x_audio)
        f, _ = self.face_lstm(x_face)
        h, _ = self.hrv_lstm(x_hrv)
        e, _ = self.eye_lstm(x_eye)

        # 按 feature 維度 concat
        encoding = torch.cat([a, f, h, e], dim=2)  # (batch, seq_len, total_hidden*2)
        M = self.tanh1(encoding)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1)  # (batch, seq_len, 1)
        out = (encoding * alpha).sum(1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)
        return out


def evaluate(epoch, epochs, model, test_loader, highest_acc, current_ablation):
    
    # evaluate
    model.eval()  # set the model to evaluation mode
    test_loss = 0.0
    test_acc = 0.0
    test_acc_values = []  # 存储测试准确率的值
    
    total_samples = 0
    correct_samples = 0
    
    y_pred = []
    y_true = []
    
    with torch.no_grad():  # turn off gradient computation to save memory and computation time
        
        # subjects: [X_text, X_audio, X_face, X_hrv, X_eye_movement, Y]
        for i, (_, text, audio, face, hrv, eye, y) in enumerate(test_loader):
            
            ############################## In GPU ##############################
                
            # 前向传播
            text = text.to(device)
            audio = audio.to(device)
            hrv = hrv.to(device)
            eye_movement = eye.to(device)
            
            face = face.to(device)
            # face = face.view(face.size(0), facial_time_series, 256)
            outputs = model(
                            x_text=text.to(dtype=torch.float), 
                            x_audio=audio.to(dtype=torch.float), 
                            x_face=face.to(dtype=torch.float).to(device),
                            x_hrv=hrv.to(dtype=torch.float).to(device), 
                            x_eye=eye_movement.to(dtype=torch.float),
                            **current_ablation,
                            )
            # outputs = model(
            #     text.to(dtype=torch.float), 
            #     audio.to(dtype=torch.float), 
            #     face.to(dtype=torch.float).to(device),
            #     hrv.to(dtype=torch.float).to(device),
            #     eye_movement.to(dtype=torch.float),
            # )
                        # if y.to(device) == 2:
            #     print(outputs.to(device))
            #     print(y.to(device))
            
            loss = loss_fn(outputs, y.to(device))     # 计算两者的误差
            # print(loss)
            
            test_loss += loss.item()
            acc = accuracy(outputs.cpu(), y)
            test_acc += acc
            test_acc_values.append(acc)
            
            _, predicted_labels = torch.max(outputs.cpu(), 1)
            y_pred.extend(predicted_labels.tolist())
            y_true.extend(y.tolist())
                
            ############################## In GPU ##############################

        # for 畫圖
        test_loss_record.append(round(test_loss/(i+1), 4))
        test_accuracy_record.append(round(test_acc/(i+1), 2))
        current_acc = round(test_acc/(i+1), 2)
        print(f"Test Loss: {test_loss/(i+1):.4f}, Test Accuracy: {test_acc/(i+1):.4f}") # , Test std:{statistics.stdev(test_acc_values):.4f}")
        
        # 計算F1分數
        f1 = f1_score(y_true, y_pred, average='macro')
        # 計算精確度
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        # 計算召回率
        recall = recall_score(y_true, y_pred, average='macro')
        # Write model performances into train.txt
        if not os.path.exists(f"{experimentPath}/{timeString}"):
            os.mkdir(f"{experimentPath}/{timeString}")
        f = open(f'{experimentPath}/{timeString}/{timeString}_train.txt', 'a')
        cf_matrix = confusion_matrix(y_true, y_pred)
        f.write("----------\n")
        print(f"{epoch}/{epochs} | Test Loss: {test_loss/(i+1):.4f}", file=f)
        print(f'{cf_matrix}', file=f)
        print(f"{epoch}/{epochs} | Test Accuracy: {test_acc/(i+1):.2f}, F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}", file=f)
        f.write("----------\n\n")
        f.close()
    

    
        cf_matrix = confusion_matrix(y_true, y_pred)
            
        # print(f'max: {max_q_quntity}')
        print(cf_matrix) #打印出来看看

        annot_fontsize = 14
        label_fontsize = 14
        annot_kws = {'fontsize': annot_fontsize}
        
        # print(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cf_matrix, annot=True, annot_kws=annot_kws, ax=ax, cmap="YlGnBu")
        ax.set_title('confusion matrix', fontsize=label_fontsize)
        ax.set_xlabel('predict', fontsize=label_fontsize)
        ax.set_ylabel('true', fontsize=label_fontsize)
        fig.savefig(f"{experimentPath}/{timeString}/{timeStringOnly}_e{epoch}", dpi=300, bbox_inches='tight')
        plt.clf() # clear canvas
        plt.close()

        uniform_data = cf_matrix/cf_matrix.sum(axis=1)[:,None] # get percentage
        cm_df = pd.DataFrame(uniform_data)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.heatmap(cm_df, annot=True, annot_kws=annot_kws, ax=ax, cmap="YlGnBu")
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_title('confusion matrix', fontsize=label_fontsize)
        ax.set_xlabel('predict', fontsize=label_fontsize)
        ax.set_ylabel('true', fontsize=label_fontsize)
        fig.savefig(f"{experimentPath}/{timeString}/{timeStringOnly}_e{epoch}_uniformed", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

        if current_acc > highest_acc:
            torch.save(model, f'{experimentPath}/{timeString}/model.pkl')
            highest_acc = current_acc
    
    model.train()
    
    return highest_acc

def predict(model, dataset):
    
    # evaluate
    model.eval()  # set the model to evaluation mode
    test_acc_values = []  # 存储测试准确率的值
    test_acc = 0.0
    
    y_pred = []
    y_true = []
    
    with torch.no_grad():  # turn off gradient computation to save memory and computation time
        
        for i, (text, audio, face, hrv, eye, y) in enumerate(dataset):
            
            ############################## In GPU ##############################
                
            # 前向传播
            text = text.to(device)
            audio = audio.to(device)
            hrv = hrv.to(device)
            eye_movement = eye.to(device)
            
            face = face.to(device)
            # face = face.view(face.size(0), facial_time_series, 256)
            outputs = model(
                            x_text=text.to(dtype=torch.float),
                            x_audio=audio.to(dtype=torch.float), 
                            x_face=face.to(dtype=torch.float).to(device),
                            x_hrv=hrv.to(dtype=torch.float).to(device), 
                            x_eye=eye_movement.to(dtype=torch.float),
                            )
            
            acc = accuracy(outputs.cpu(), y)
            test_acc += acc
            test_acc_values.append(acc)

            _, predicted_labels = torch.max(outputs.cpu(), 1)
            y_pred.extend(predicted_labels.tolist())
            y_true.extend(y.tolist())
                
            ############################## In GPU ##############################

        # for 畫圖
        test_accuracy_record.append(round(test_acc/(i+1), 2))
        print(f"Test Accuracy: {test_acc/(i+1):.4f}")
        
        # 計算F1分數
        f1 = f1_score(y_true, y_pred, average='macro')
        # 計算精確度
        precision = precision_score(y_true, y_pred, average='macro')
        # 計算召回率
        recall = recall_score(y_true, y_pred, average='macro')
        # Write model performances into train.txt
        if not os.path.exists(f"{experimentPath}/{timeString}_predict"):
            os.mkdir(f"{experimentPath}/{timeString}_predict")
        f = open(f'{experimentPath}/{timeString}_predict/{timeString}_predict.txt', 'a')
        cf_matrix = confusion_matrix(y_true, y_pred)
        f.write("----------\n")
        print(f'{cf_matrix}', file=f)
        print(f"Predict Accuracy: {test_acc/(i+1):.2f}, F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}", file=f)
        f.write("----------\n\n")
        f.close()
    

        cf_matrix = confusion_matrix(y_true, y_pred)
            
        print(cf_matrix) #打印出来看看

        annot_fontsize = 14
        label_fontsize = 14
        annot_kws = {'fontsize': annot_fontsize}
        
        # print(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cf_matrix, annot=True, annot_kws=annot_kws, ax=ax, cmap="YlGnBu") #画热力图
        ax.set_title('confusion matrix', fontsize=label_fontsize) #标题
        ax.set_xlabel('predict', fontsize=label_fontsize) #x轴
        ax.set_ylabel('true', fontsize=label_fontsize) #y轴
        fig.savefig(f'{experimentPath}/{timeString}_predict/{timeString}_predict', dpi=300, bbox_inches='tight')
        plt.close()
    return


def modify_dataset_for_parent_sid(dataset):
    # 修改 dataset，使得 sid 變成去掉後綴的父 sid
    modified_dataset = []
    
    for data in dataset:
        sid = data[0]  # 假設 sid 是元組中的第一個元素
        if isinstance(sid, tuple):
            sid = sid[0]  # 取出元組中的第一個元素，假設它是字串
        
        parent_sid = '_'.join(sid.split('_')[:-1])  # 去掉後綴部分
        # 重新構造修改過的元組，將修改過的 sid 返回
        modified_dataset.append((parent_sid, *data[1:]))  # 保留其他部分不變
    
    return modified_dataset

def predict_sid(model, dataset):
    
    # evaluate
    model.eval()  # set the model to evaluation mode
    test_acc = 0.0
    
    y_pred = {}
    y_true = {}
    
    # Load sid-level label file
    # sid_label_df = pd.read_csv('/data/feature_data/output_with_averages_and_modes.csv')
    # sid_label_map = {
    #     sid: 0 if val in [2] else 1
    #     for sid, val in zip(sid_label_df['sid'], sid_label_df['most_frequent_value'])
    #     if val in [2, 3]
    # }

    # modified_dataset = modify_dataset_for_parent_sid(dataset)

    with torch.no_grad():  # turn off gradient computation to save memory and computation time
        
        for i, (sid, text, audio, face, hrv, eye, y) in enumerate(dataset):  # Assuming dataset yields sid as well
            
            text = text.to(device)
            audio = audio.to(device)
            hrv = hrv.to(device)
            eye_movement = eye.to(device)
            
            face = face.to(device)
            
            outputs = model(
                            x_text=text.to(dtype=torch.float),
                            x_audio=audio.to(dtype=torch.float), 
                            x_face=face.to(dtype=torch.float).to(device),
                            x_hrv=hrv.to(dtype=torch.float).to(device), 
                            x_eye=eye_movement.to(dtype=torch.float),
                            )
            
            # 多數決
            _, predicted_labels = torch.max(outputs.cpu(), 1)
            
            if sid not in y_pred:
                y_pred[sid] = []
            
            y_pred[sid].extend(predicted_labels.tolist())
            y_true[sid] = y[0].item()  # Assuming the same label for all samples of a sid

            # # 取平均
            # # outputs: logits (batch_size, 2)
            # probs = torch.softmax(outputs.cpu(), dim=1)  # shape: (B, 2)
            # class_1_probs = probs[:, 1]  # 只取 class=1 的機率

            # if sid not in y_pred:
            #     y_pred[sid] = []

            # y_pred[sid].extend(class_1_probs.tolist())  # 直接儲存機率


        final_predictions = []
        actual_labels = []

        for sid, predictions in y_pred.items():
            most_common_pred = Counter(predictions).most_common(1)[0][0]
            final_predictions.append(most_common_pred)
            actual_labels.append(y_true[sid])
            # if isinstance(sid, tuple):
            #     sid = sid[0]
            # # pure_sid = sid.split('_')[0]
            # actual_labels.append(sid_label_map[sid])

        # for sid, probabilities in y_pred.items():
        #     if isinstance(sid, tuple):
        #         sid = sid[0]
        #     pure_sid = sid.split('_')[0]

        #     if pure_sid not in sid_label_map:
        #         print(f"[!] SID {pure_sid} not found in sid_label_map, skipping.")
        #         continue

        #     # 🔄 用平均機率決定預測類別
        #     avg_prob = np.mean(probabilities)
        #     predicted_label = 1 if avg_prob >= 0.5 else 0

        #     final_predictions.append(predicted_label)
        #     actual_labels.append(sid_label_map[pure_sid])
        

        # Metrics calculation
        accuracy = accuracy_score(actual_labels, final_predictions)
        f1 = f1_score(actual_labels, final_predictions, average='macro')
        precision = precision_score(actual_labels, final_predictions, average='macro')
        recall = recall_score(actual_labels, final_predictions, average='macro')
        cf_matrix = confusion_matrix(actual_labels, final_predictions)

        print(f"Predict Accuracy: {accuracy:.2f}, F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

        # Save results
        if not os.path.exists(f"{experimentPath}/{timeString}_predict"):
            os.mkdir(f"{experimentPath}/{timeString}_predict")
        with open(f'{experimentPath}/{timeString}_predict/{timeString}_predict.txt', 'a') as f:
            f.write("----------\n")
            print(f'{cf_matrix}', file=f)
            print(f"Predict Accuracy: {accuracy:.2f}, F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}", file=f)
            f.write("----------\n\n")

        # Draw confusion matrix
        annot_fontsize = 14
        label_fontsize = 14
        annot_kws = {'fontsize': annot_fontsize}

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cf_matrix, annot=True, annot_kws=annot_kws, ax=ax, cmap="YlGnBu") #draw heatmap
        ax.set_title('confusion matrix', fontsize=label_fontsize) #title
        ax.set_xlabel('predict', fontsize=label_fontsize) #x-axis
        ax.set_ylabel('true', fontsize=label_fontsize) #y-axis
        fig.savefig(f'{experimentPath}/{timeString}_predict/{timeString}_predict', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

        uniform_data = cf_matrix/cf_matrix.sum(axis=1)[:,None] # get percentage
        cm_df = pd.DataFrame(uniform_data)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.heatmap(cm_df, annot=True, annot_kws=annot_kws, ax=ax, cmap="YlGnBu")
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_title('confusion matrix', fontsize=label_fontsize)
        ax.set_xlabel('predict', fontsize=label_fontsize)
        ax.set_ylabel('true', fontsize=label_fontsize)
        fig.savefig(f"{experimentPath}/{timeString}_predict/{timeString}_predict_uniformed", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

    return

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # x: (B, T, D)
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T, 2H)

            # mask padding
            mask = torch.arange(out.size(1), device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).expand_as(out).float()
            out = out * mask
            mean_out = out.sum(dim=1) / lengths.unsqueeze(1).to(x.device)
        else:
            # 如果 lengths 沒提供，就直接 mean-pool 所有時間步
            out, _ = self.lstm(x)  # (B, T, 2H)
            mean_out = out.mean(dim=1)  # (B, 2H)

        return self.fc(mean_out)  # (B, output_dim)

class GrowthMindsetModel(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        self.text_encoder = BiLSTMEncoder(771, 128, 256)
        self.audio_encoder = BiLSTMEncoder(193, 64, 128)
        self.face_encoder = BiLSTMEncoder(input_dim=256, hidden_dim=64, output_dim=128)

        self.hrv_encoder = ModalityEncoder(23, 64)
        self.eye_encoder = ModalityEncoder(13, 64)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 + 128 + 128 + 64 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # Apply sigmoid or softmax externally
        )

    def forward(self, x_text, x_audio, x_face, x_hrv, x_eye, lengths=None,
                use_text=True, use_audio=True, use_face=True, use_hrv=True, use_eye=True):
        batch_size = x_text.size(0)
        device = x_text.device

        text_vec = self.text_encoder(x_text, lengths) if use_text else torch.zeros(batch_size, 256, device=device)
        audio_vec = self.audio_encoder(x_audio, lengths) if use_audio else torch.zeros(x_audio.size(0), 128, device=x_audio.device)
        face_vec = self.face_encoder(x_face, lengths) if use_face else torch.zeros(x_face.size(0), 128, device=x_face.device)
        hrv_vec = self.hrv_encoder(x_hrv) if use_hrv else torch.zeros(x_hrv.size(0), 64, device=x_hrv.device)
        eye_vec = self.eye_encoder(x_eye) if use_eye else torch.zeros(x_eye.size(0), 64, device=x_eye.device)

        fusion = torch.cat([text_vec, audio_vec, face_vec, hrv_vec, eye_vec], dim=-1)
        return self.classifier(fusion)

import torch.nn as nn

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)  # x: (B, input_dim) → output: (B, output_dim)

if __name__ == '__main__':
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_name = ""
    
    # Formal: FiveFusionModel2_Att
    model = FiveFusionModel2_Att(
                    n_text=771,
                    n_audio=193,
                    n_face=256 * facial_time_series, # 輸入層
                    n_hrv=23,
                    n_eye=13,
                    lstm_hiddens=[771, 193, 256, 23, 13], # 輸出層
                    text_lstm_layers=1,
                    audio_lstm_layers=1,
                    face_lstm_layers=1,
                    hrv_lstm_layers=1,
                    eye_lstm_layers=1,
                    n_output=CATEGORY).to(device)
    model_name = "FiveFusionModel2_Att"

    # model = UniModalModel_Att(
    #             input_size=771,
    #             # input_size=256 * facial_time_series,
    #             lstm_hidden=771,        # 你也可以選較小 hidden，例如 256
    #             lstm_layers=1,
    #             n_output=CATEGORY
    #         ).to(device)
    # model_name = "TextOnlyModel"

    # model = NoTextModel(
    #     n_audio=193,
    #     n_face=256 * facial_time_series,
    #     n_hrv=23,
    #     n_eye=13,
    #     lstm_hiddens=[193, 256, 23, 13],
    #     audio_lstm_layers=1,
    #     face_lstm_layers=1,
    #     hrv_lstm_layers=1,
    #     eye_lstm_layers=1,
    #     n_output=CATEGORY).to(device)
    
    # linear_hiddens = [128, 64, 128, 32, 32]
    # lstm_hiddens = [128, 64]
    # model = FiveDenseFusionModel(
    #     n_text=771,
    #     n_audio=193,
    #     n_face=256 * facial_time_series,
    #     n_hrv=23,
    #     n_eye=13,
    #     linear_hiddens=linear_hiddens,
    #     lstm_hiddens=lstm_hiddens,
    #     n_output=2
    # ).to(device)

    # model = FivePureDenseFusionModel(
    #     n_text=771,
    #     n_audio=193,
    #     n_face=256 * facial_time_series,
    #     n_hrv=23,
    #     n_eye=13,
    #     linear_hiddens=linear_hiddens,
    #     n_output=2
    # ).to(device)
    # model_name = "FiveFusionModel2_Att"
    # model = GrowthMindsetModel(output_dim=2).to(device)

    # ------------------- Predict function -------------------
    if isPredict:
        model = torch.load(predictModel, weights_only=False).to(device)
        dataset = QuestionDatasetWithSID()
        test_loader = DataLoader(dataset)
        print(len(test_loader))
        print(model)
        predict_sid(model, test_loader)
        exit()
    
    timeString += f"_{model_name}_epochs={epochs}_" 
    
    if optimizerName == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.01)
    elif optimizerName == "AdamDefault":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    elif optimizerName == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=0.01)
    else:
        print("No optimeizer! Train quit.")
        exit
    
    highest_acc = 0.0

    # load data
    dataset = QuestionDatasetWithSID()
    # dataset = SequencePerSubjectDataset()
    dataset_size = len(dataset)

    # 獲取所有唯一的 sid
    unique_sids = list(set(dataset.X_sid))

    # 建立 sid -> label 映射（既然同一 sid 都是同一 label，只取第一個）
    sid_to_label = {}
    for i, sid in enumerate(dataset.X_sid):
        if sid not in sid_to_label:
            _, _, _, _, _, _, label = dataset[i]
            sid_to_label[sid] = label

    # 構建 stratified 資料
    sid_list = list(sid_to_label.keys())
    label_list = [sid_to_label[sid] for sid in sid_list]

    modalities = ['text', 'audio', 'face', 'hrv', 'eye']

    # Generate all 32 combinations (True/False)
    all_combinations = list(itertools.product([True, False], repeat=5))
    # Remove the combination where ALL modalities are False (leaves 31 valid configs)
    valid_combinations = [combo for combo in all_combinations if any(combo)]

    # Create the output file and write the header
    output_filename = '/data/train_mindset/5-Fold_Ablation_Results.txt'
    with open(output_filename, 'w') as f:
        f.write("5-Fold CV Comprehensive Ablation Study\n")
        f.write("="*65 + "\n\n")

    # =========================================================
    # START MASTER COMBINATION LOOP
    # =========================================================
    for combo in valid_combinations:
        
        # Map the boolean tuple back to the modality names
        current_ablation = dict(zip(modalities, combo))
        print(f"\n\n{'='*60}")
        print(f"🚀 RUNNING ABLATION CONFIG: {current_ablation}")
        print(f"{'='*60}\n")

        ablation_kwargs = {
            'use_text': current_ablation['text'],
            'use_audio': current_ablation['audio'],
            'use_face': current_ablation['face'],
            'use_hrv': current_ablation['hrv'],
            'use_eye': current_ablation['eye'],
        }

        # Implement Stratified 5-Fold Cross Validation
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=253)

        # Trackers for all metrics for this specific configuration
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1_scores = []

        # START K-FOLD LOOP
        for fold, (train_idx, test_idx) in enumerate(kf.split(sid_list, label_list)):
            print(f"\n--- Starting Fold {fold + 1}/{n_splits} ---")

            train_sids = [sid_list[i] for i in train_idx]
            test_sids = [sid_list[i] for i in test_idx]

            train_indices = [i for i, sid in enumerate(dataset.X_sid) if sid in train_sids]
            test_indices = [i for i, sid in enumerate(dataset.X_sid) if sid in test_sids]

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # CRITICAL: Re-initialize the model & optimizer for every fold!
            model = GrowthMindsetModel(output_dim=CATEGORY).to(device)
            
            if optimizerName == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.01)
            elif optimizerName == "AdamDefault":
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
            elif optimizerName == "RMSprop":
                optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=0.01)
            else:
                raise ValueError(f"Unsupported optimizerName: {optimizerName}")

            # Calculate balanced weights for the loss function
            current_train_labels = [label_list[i] for i in train_idx]
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(current_train_labels), y=current_train_labels)
            weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

            highest_acc = 0.0

            # INNER EPOCH LOOP
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                train_acc = 0.0
                batch_count = 0
                
                for i, (_, text, audio, face, hrv, eye, y) in enumerate(train_loader):
                    text = text.to(device)
                    audio = audio.to(device)
                    hrv = hrv.to(device)
                    eye_movement = eye.to(device)
                    face = face.to(device)
                    y = y.to(device)

                    optimizer.zero_grad()
                    
                    outputs = model(
                        x_text=text.to(dtype=torch.float), 
                        x_audio=audio.to(dtype=torch.float), 
                        x_face=face.to(dtype=torch.float),
                        x_hrv=hrv.to(dtype=torch.float), 
                        x_eye=eye_movement.to(dtype=torch.float),
                        **ablation_kwargs
                    )
                    
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_acc += accuracy(outputs.cpu(), y.cpu())  # Assuming you have an accuracy() function defined elsewhere
                    batch_count += 1

                # Optional: print epoch stats
                print(f"Fold {fold+1} | Epoch [{epoch+1}/{epochs}] | Loss: {train_loss/max(1, batch_count):.4f}")

            # ---------------------------------------------------------
            # POST-EPOCH EVALUATION: Calculate P, R, F1 on the Test Set
            # ---------------------------------------------------------
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for _, text, audio, face, hrv, eye, y in test_loader:
                    text = text.to(device)
                    audio = audio.to(device)
                    hrv = hrv.to(device)
                    eye_movement = eye.to(device)
                    face = face.to(device)
                    
                    outputs = model(
                        x_text=text.to(dtype=torch.float), 
                        x_audio=audio.to(dtype=torch.float), 
                        x_face=face.to(dtype=torch.float),
                        x_hrv=hrv.to(dtype=torch.float), 
                        x_eye=eye_movement.to(dtype=torch.float),
                        **ablation_kwargs
                    )
                    
                    # Get the predicted classes (argmax)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())

            # Calculate comprehensive metrics for this fold
            # average='macro' ensures all classes are treated equally regardless of imbalance
            fold_acc = accuracy_score(all_targets, all_preds)
            fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='macro', zero_division=0
            )

            fold_accuracies.append(fold_acc)
            fold_precisions.append(fold_prec)
            fold_recalls.append(fold_rec)
            fold_f1_scores.append(fold_f1)
            
            print(f"--> Fold {fold+1} Completed | Acc: {fold_acc:.4f} | F1: {fold_f1:.4f}")

        # =========================================================
        # END OF K-FOLD FOR THIS CONFIG: Save to File
        # =========================================================
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        mean_prec = np.mean(fold_precisions)
        std_prec = np.std(fold_precisions)
        mean_rec = np.mean(fold_recalls)
        std_rec = np.std(fold_recalls)
        mean_f1 = np.mean(fold_f1_scores)
        std_f1 = np.std(fold_f1_scores)

        # Use 'a' to append so data is saved immediately after every configuration finishes
        with open(output_filename, 'a') as f:
            f.write(f"Ablation Config: {current_ablation}\n")
            f.write(f"Fold Accuracies: {[round(num, 4) for num in fold_accuracies]}\n")
            f.write(f"Mean Accuracy:  {mean_acc:.4f} ± {std_acc:.4f}\n")
            f.write(f"Mean Precision: {mean_prec:.4f} ± {std_prec:.4f}\n")
            f.write(f"Mean Recall:    {mean_rec:.4f} ± {std_rec:.4f}\n")
            f.write(f"Mean F1-Score:  {mean_f1:.4f} ± {std_f1:.4f}\n")
            f.write("-" * 65 + "\n")
            
        print(f"✅ Config Saved. Mean Acc: {mean_acc:.4f} | Mean F1: {mean_f1:.4f}")

    print("\n🎉 ALL 32 ABLATION CONFIGURATIONS COMPLETE!")