import time
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import pickle
import random

from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from config import args
import numpy as np

import torch
import copy
import os
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from newTry.Trainer import GraphDataset, NoiseInjector
from newTry.model import Predictor
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import seaborn as sns
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


def multiclass_roc_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def multiclass_pr_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return average_precision_score(y_test, y_pred, average=average)


class Test():
    def __init__(self,seed,eventlog,num_prototypes):
        self.eventlog=eventlog
        self.device = torch.device('cuda:{}'.format(args.gpu))
        self.seed = seed
        self.num_prototypes=num_prototypes
        self.model =Predictor(self.eventlog,args.d_model,self.num_prototypes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)



    def eval(self, eval_model, data_loader):
        eval_model.eval()

        total_loss = 0.0
        total_samples = 0

        # 汇总全量数据用于计算高级指标
        all_labels = []
        all_preds = []

        all_attentions = []  # 存注意力权重
        all_label = []  # 存真实标签

        with torch.no_grad():
            for batch in data_loader:
                # 1. 动态兼容：无论是图模式还是LSTM模式，都将 Tensor 移至设备
                batch["event_matrix"] = batch["event_matrix"].to(self.device)
                batch["object_matrix"] = batch["object_matrix"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 2. 模型推理
                logits,_ = eval_model(batch)
                loss = self.criterion(logits, labels)

                batch_attn = eval_model.encoder.global_fusion.get_last_batch_attention()

                all_attentions.append(batch_attn.cpu().numpy())
                all_label.append(labels.cpu().numpy())

                # 3. 获取预测类别
                preds = logits.argmax(dim=1)

                # 4. 累加基础统计量
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # 5. 收集数据到 CPU (用于 sklearn 计算)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # 转换为 numpy 数组
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        final_attentions = np.concatenate(all_attentions, axis=0)
        # 形状: [Total_Samples]

        final_labels = np.concatenate(all_label, axis=0)


        avg_loss = total_loss / max(total_samples, 1)

        # 精确率、召回率、F1 (Macro 平均)
        # zero_division=0 防止由于模型未预测某些类别导致的报错
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        # 准确率
        acc = (y_true == y_pred).mean()

        # --- 调用你提供的自定义 AUC 函数 ---
        # 注意：这两个函数在内部执行了 LabelBinarizer().transform
        roc_auc = multiclass_roc_auc_score(y_true, y_pred, average='macro')
        pr_auc = multiclass_pr_auc_score(y_true, y_pred, average='macro')

        # --- 打印报告 ---
        print("\n" + "=" * 40)
        print(f"{'Evaluation Metrics Report':^40}")
        print("-" * 40)
        print(f"Loss      : {avg_loss:.4f}")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {precision:.4f} (Macro)")
        print(f"Recall    : {recall:.4f} (Macro)")
        print(f"F1 Score  : {f1:.4f} (Macro)")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"PR-AUC    : {pr_auc:.4f}")
        print("=" * 40 + "\n")

        # 返回核心指标，方便你在外部记录（如 Tensorboard 或 CSV）
        return avg_loss, acc, precision, recall,  f1, roc_auc, pr_auc

    #这个是用来训练并且评估的主函数
    def test(self):

        print(f"test {self.eventlog}  ******************************************************************")
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        parent_dir = current_dir.parent
        parent_dir_str = str(parent_dir)

        base_data_path = os.path.join(parent_dir_str, "data", self.eventlog)


        test_graph_path = os.path.join(base_data_path, "test_Graph.pt")
        test_label_path = os.path.join(base_data_path, "test_label.pt")


        test_graphs = torch.load(test_graph_path)
        test_labels = torch.load(test_label_path)



        test_loader = DataLoader(
            GraphDataset(test_graphs),
            batch_size=args.batch_size,
            shuffle=True,
            #collate_fn=lambda x: x,  # 不做任何 stack
            collate_fn=GraphDataset.graph_collate_fn,
            num_workers=2,  # <--- 手动加上这个参数，建议先设为 4 或 8
            pin_memory=True  # <--- 配合 3090 使用，建议也加上
        )



        model_path = (
                      'model/' + str(self.eventlog) + '/' +
                      self.eventlog + '_' + 'num_prototypes' + str(self.num_prototypes) +
                                                                                          '_seed' + str(self.seed) +
                      '_model.pkl'
                      )

        self.model = Predictor(self.eventlog, args.d_model, num_prototypes)

        check_model = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(check_model.state_dict())
        check_model = self.model


        # 35. 打印最佳模型的验证集指标
        avg_loss, acc,per,recall, f1, roc_auc, pr_auc = self.eval(check_model, test_loader)
        print('+' * 89)

        print(f'\tBest_Loss: {avg_loss:.5f}(test)\t|\tBest_Acc: {acc * 100:.2f}%(test)')
        print('+' * 89)

        return acc



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




if __name__ == "__main__":

    seed =42
    set_seed(seed)
    eventlog  = [
        "intermediate",
        "OTC",
        "p2p",
        "BPI2017"

    ]
    k_data_dict = {
        "OTC": 8,
        "BPI2017": 4,
        "intermediate": 16,
        "p2p": 64
    }
    for event in eventlog:
        num_prototypes = k_data_dict[event]
        TTest = Test(seed,event,num_prototypes)
        test_acc = TTest.test()


        print("-"*200)
        print(f"eventlog: {event}  accuracy:{round(test_acc,4)}")
