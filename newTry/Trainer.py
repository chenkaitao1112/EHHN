import time
from pathlib import Path
import seaborn as sns
import torch.nn as nn
import pickle
import random
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
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


from newTry.model import Predictor
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch

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




def extract_and_pad_lifecycle(batch_event_matrices, batch_primary_edges):
    """
    从每个样本的事件矩阵中提取主线事件，并填充为固定形状。

    Args:
        batch_event_matrices: List[Tensor], 每个元素形状为 [num_events, feat_dim]
        batch_primary_edges: List[List[tuple[int]]], 每个样本的生命周期边索引（修正后的格式）

    Returns:
        padded_features: Tensor [Batch, Max_Len, Feat_Dim]
        padding_mask: Tensor [Batch, Max_Len] (True 表示是 Padding 位)
    """
    # 1. 提取每个样本的主线向量
    lifecycle_sequences = []
    if len(batch_event_matrices) == 0:
        return torch.empty(0), torch.empty(0)
    feat_dim = batch_event_matrices[0].shape[-1]

    for i, g_primary_edges in enumerate(batch_primary_edges):
        # 取出第一个列表（即主线索引），并处理空值
        indices_raw = g_primary_edges[0] if (len(g_primary_edges) > 0 and len(g_primary_edges[0]) > 0) else []

        # 关键修复：处理 List[tuple[int]] 格式，展平为纯一维整数列表
        def flatten_to_int_list(lst):
            flat = []
            for item in lst:
                # 处理 tuple 元素（核心适配你发现的格式）
                if isinstance(item, tuple):
                    # 假设 tuple 里是单个索引，如 (0,) → 取 0；如果是 (0,1) 则取第一个元素（可根据你的实际逻辑调整）
                    flat.append(item[0])
                # 兼容原有的 list 格式
                elif isinstance(item, list):
                    flat.extend(flatten_to_int_list(item))
                # 处理纯整数
                elif isinstance(item, int):
                    flat.append(item)
            return flat

        # 展平为纯一维整数列表
        indices = flatten_to_int_list(indices_raw)

        if not indices:
            # 如果没有主线，放一个全 0 向量占位
            lifecycle_sequences.append(torch.zeros((1, feat_dim), device=batch_event_matrices[0].device))
        else:
            # 过滤无效索引（负数、超过张量行数的索引）
            valid_indices = [idx for idx in indices if 0 <= idx < batch_event_matrices[i].shape[0]]
            if not valid_indices:
                lifecycle_sequences.append(torch.zeros((1, feat_dim), device=batch_event_matrices[0].device))
            else:
                # 现在 indices 是纯一维整数列表，可正确索引二维张量
                seq = batch_event_matrices[i][valid_indices]
                lifecycle_sequences.append(seq)

    # 2. 确定当前 Batch 的最大长度
    max_len = max(seq.shape[0] for seq in lifecycle_sequences)
    batch_size = len(lifecycle_sequences)

    # 3. 初始化填充后的 Tensor 和 Mask
    padded_features = torch.zeros((batch_size, max_len, feat_dim),
                                  device=batch_event_matrices[0].device)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool,
                              device=batch_event_matrices[0].device)

    for i, seq in enumerate(lifecycle_sequences):
        length = seq.shape[0]
        padded_features[i, :length, :] = seq
        padding_mask[i, :length] = False  # 有效数据位设为 False

    return padded_features, padding_mask



class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    @staticmethod
    def graph_collate_fn(batch):
        # === 1. 初始化容器 ===
        batch_event_X = []
        batch_object_X = []
        all_merged_edges = []
        event_nums = []
        object_nums = []
        labels = []
        batch_aux_time_labels = []
        batch_primary_edges = []

        # --- 新增容器 ---
        batch_sep_event_hyperedges = []  # 分离版：事件
        batch_sep_object_hyperedges = []  # 分离版：对象
        batch_lifecycle_hyperedges = []  # 生命周期超边
        batch_main_object_indices = []  # 主视角对象索引

        # 测试变量
        batch_idx_events = []
        batch_idx_objects = []

        # === 2. 第一轮循环：收集基础数据 ===
        for i, g in enumerate(batch):
            # A. 收集矩阵与标签
            batch_event_X.append(g["event_matrix"])
            batch_object_X.append(g["object_matrix"])
            labels.append(g["label"])
            batch_aux_time_labels.append(g["aux_time_label"])

            # B. 收集主线边 (用于 padding)
            batch_primary_edges.append(g["local_primary_edges"])

            # C. 记录数量
            n_e = g["event_matrix"].shape[0]
            n_o = g["object_matrix"].shape[0]
            event_nums.append(n_e)
            object_nums.append(n_o)

            # D. Batch Index
            batch_idx_events.append(torch.full((n_e,), i, dtype=torch.long))
            batch_idx_objects.append(torch.full((n_o,), i, dtype=torch.long))

        # === 3. 生命周期 Padding (保持原有逻辑) ===
        if batch_event_X:
            lifecycle_tensor, lifecycle_mask = extract_and_pad_lifecycle(
                batch_event_X, batch_primary_edges
            )
        else:
            lifecycle_tensor = torch.empty(0)
            lifecycle_mask = torch.empty(0)

        # === 4. 计算全局偏移量 ===
        total_events_in_batch = sum(event_nums)
        total_objects_in_batch = sum(object_nums)

        # event_offsets[i] 表示第 i 个图的事件在 batch_event_X 中的起始行号
        event_offsets = [0] + torch.cumsum(torch.tensor(event_nums), dim=0).tolist()[:-1]
        object_offsets = [0] + torch.cumsum(torch.tensor(object_nums), dim=0).tolist()[:-1]

        # === 5. 第二轮循环：计算所有边的全局索引 (核心修改区) ===
        for i, g in enumerate(batch):
            n_e = event_nums[i]
            e_off = event_offsets[i]
            o_off = object_offsets[i]

            # ------------------------------------------------------------
            # A. 处理 Merged Edges (全图统一索引) - 保持你原有的逻辑
            # ------------------------------------------------------------
            local_edges = g["local_eo_edges"] + g["local_primary_edges"]
            for edge in local_edges:
                new_global_edge = []
                for v in edge:
                    if v < n_e:
                        # 事件：e_off + v
                        new_global_edge.append(e_off + v)
                    else:
                        # 对象：Total_Events + o_off + (v - n_e)
                        # 注意：v - n_e 是还原回相对索引
                        new_global_edge.append(total_events_in_batch + o_off + (v - n_e))
                all_merged_edges.append(tuple(new_global_edge))

            # ------------------------------------------------------------
            # B. 处理 Separated Edges (利用预计算好的索引，极快)
            # ------------------------------------------------------------
            # g["sep_eo_evt_indices"]: [local_e1, local_e2...]
            # g["sep_eo_obj_indices_list"]: [[local_o_a, local_o_b], ...]

            raw_sep_evts = g["sep_eo_evt_indices"]
            raw_sep_objs = g["sep_eo_obj_indices_list"]

            for local_e, local_objs in zip(raw_sep_evts, raw_sep_objs):
                # 事件全局索引 (指向 batch_event_X)
                batch_sep_event_hyperedges.append([e_off + local_e])

                # 对象全局索引 (指向 batch_object_X)
                # 注意：这里只加 o_off，因为是指向 Object Matrix
                global_objs = [o_off + lo for lo in local_objs]
                batch_sep_object_hyperedges.append(global_objs)

            # ------------------------------------------------------------
            # C. 处理 Main Object Index (主视角对象)
            # ------------------------------------------------------------
            po_local = g["primary_object_idx"]
            if po_local != -1:
                # 指向 batch_object_X 的行号
                batch_main_object_indices.append(o_off + po_local)
            else:
                batch_main_object_indices.append(-1)

            # ------------------------------------------------------------
            # D. 处理 Lifecycle Hyperedges (生命周期边)
            # ------------------------------------------------------------
            # 提取唯一节点并排序，转全局
            unique_nodes = set()
            for edge in g["local_primary_edges"]:  # 这里的边全是事件节点
                for node in edge:
                    unique_nodes.add(node)

            # 转全局：e_off + local_v
            global_lifecycle = [e_off + v for v in sorted(list(unique_nodes))]
            batch_lifecycle_hyperedges.append(global_lifecycle)

        # === 6. 返回结果 ===
        return {
            "event_matrix": torch.cat(batch_event_X, dim=0) if batch_event_X else torch.empty(0),
            "object_matrix": torch.cat(batch_object_X, dim=0) if batch_object_X else torch.empty(0),
            "merged_edges": all_merged_edges,

            "event_batch": torch.cat(batch_idx_events) if batch_idx_events else torch.empty(0, dtype=torch.long),
            "object_batch": torch.cat(batch_idx_objects) if batch_idx_objects else torch.empty(0, dtype=torch.long),

            "event_nums": event_nums,
            "event_offsets": event_offsets,
            "object_nums": object_nums,
            "object_offsets": object_offsets,

            "total_events": total_events_in_batch,
            "total_nodes": total_events_in_batch + total_objects_in_batch,

            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.empty(0, dtype=torch.long),
            "aux_time_labels": torch.tensor(batch_aux_time_labels,
                                            dtype=torch.float32) if batch_aux_time_labels else torch.empty(0),

            "lifecycle_features": lifecycle_tensor,
            "lifecycle_mask": lifecycle_mask,

            # --- 新增的返回值 ---
            "sep_event_hyperedges": batch_sep_event_hyperedges,
            "sep_object_hyperedges": batch_sep_object_hyperedges,
            "lifecycle_hyperedges": batch_lifecycle_hyperedges,
            "main_object_indices": torch.tensor(batch_main_object_indices, dtype=torch.long)
        }


class DHTrainer():
    def __init__(self,seed,eventlog,num_prototypes):
        self.eventlog=eventlog
        self.device = torch.device('cuda:{}'.format(args.gpu))
        self.warmup_steps = 10
        self.seed = seed

        self.model =Predictor(self.eventlog,args.d_model,num_prototypes)
        self.num_prototypes = num_prototypes



        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # self.criterion = FocalLoss(alpha=class_weights_tensor)
        self.criterion_reg = nn.SmoothL1Loss()  # 修订
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08,
                                           weight_decay=0)
        self.scheduler_cosine = CosineAnnealingLR(self.optimizer, args.epochs - self.warmup_steps)
        self.scheduler_warmup = LambdaLR(self.optimizer, lr_lambda=lambda epoch: epoch / self.warmup_steps)

    def train(self, train_loader,epoch):

        self.model.train()
        torch.cuda.reset_peak_memory_stats(device=self.device)
        torch.cuda.empty_cache()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        epoch_pure_compute_time = 0
        for batch in train_loader:
            start_time = time.perf_counter()
            # graphs, labels = zip(*batch)
            # labels = torch.tensor(labels, dtype=torch.long, device=self.device)
            batch["event_matrix"] = batch["event_matrix"].to(self.device)
            batch["object_matrix"] = batch["object_matrix"].to(self.  device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            logits,aux_loss = self.model(batch)  # ★ 模型直接吃 List[graph]
            main_loss = self.criterion(logits, labels)

            target_weight = 0.01
            warmup_epochs = 10

            # 计算当前权重 (从 0.0 -> 0.01)
            if epoch < warmup_epochs:
                current_weight = target_weight * (epoch / warmup_epochs)
            else:
                current_weight = target_weight

            loss = main_loss + current_weight * aux_loss #+ 0.001 * self.model.encoder.global_fusion.prototype_diversity_loss()


            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()


            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

            end_time = time.perf_counter()
            run_time = end_time - start_time
            epoch_pure_compute_time += run_time

            #print("run_time:", round(run_time, 2), "-" * 139)

        peak_memory_bytes = torch.cuda.max_memory_allocated(device=self.device)

        peak_memory_gb = peak_memory_bytes / (1024 ** 3)

        print(f"Epoch {epoch} Peak GPU Memory (on {self.device}): {peak_memory_gb:.4f} GB")


        print(f"Loss: {total_loss / total_samples:.4f}, Acc: {total_correct / total_samples:.4f}")
        print(f"时间{round(epoch_pure_compute_time, 2)}")

    def eval(self, eval_model,data_loader):
        eval_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                # graphs, labels = zip(*batch)
                # labels = torch.tensor(labels, dtype=torch.long, device=self.device)
                batch["event_matrix"] = batch["event_matrix"].to(self.device)
                batch["object_matrix"] = batch["object_matrix"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits,_ = eval_model(batch)
                loss = self.criterion(logits, labels)



                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        acc = total_correct / max(total_samples, 1)

        return avg_loss, acc



    def test(self, eval_model, data_loader):
        eval_model.eval()

        total_loss = 0.0
        total_samples = 0

        # 汇总全量数据用于计算高级指标
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in data_loader:
                # 1. 动态兼容：无论是图模式还是LSTM模式，都将 Tensor 移至设备
                batch["event_matrix"] = batch["event_matrix"].to(self.device)
                batch["object_matrix"] = batch["object_matrix"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 2. 模型推理
                logits,_ = eval_model(batch)
                loss = self.criterion(logits, labels)

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

        # --- 指标计算 ---
        # 平均损失
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
    def train_eval(self):

        print(f"start train {self.eventlog}  ******************************************************************")
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        parent_dir = current_dir.parent
        parent_dir_str = str(parent_dir)

        base_data_path = os.path.join(parent_dir_str, "data", self.eventlog)

        train_graph_path = os.path.join(base_data_path, "train_Graph.pt")
        train_label_path = os.path.join(base_data_path, "train_label.pt")

        val_graph_path = os.path.join(base_data_path, "val_Graph.pt")
        val_label_path = os.path.join(base_data_path, "val_label.pt")

        test_graph_path = os.path.join(base_data_path, "test_Graph.pt")
        test_label_path = os.path.join(base_data_path, "test_label.pt")

        train_graphs = torch.load(train_graph_path)
        train_labels = torch.load(train_label_path)

        val_graphs = torch.load(val_graph_path)
        val_labels = torch.load(val_label_path)

        test_graphs = torch.load(test_graph_path)
        test_labels = torch.load(test_label_path)

        print(len(train_graphs))


        best_val_acc = 0
        best_epoch = 0
        best_model = None
        patience = 20
        wait = 0
        min_epochs = 20




        train_loader = DataLoader(
            GraphDataset(train_graphs),
            batch_size=args.batch_size,
            shuffle=True,
            #collate_fn=lambda x: x,  # 不做任何 stack
            collate_fn=GraphDataset.graph_collate_fn,
            num_workers=2,  # <--- 手动加上这个参数，建议先设为 4 或 8
            pin_memory=True  # <--- 配合 3090 使用，建议也加上
        )

        valid_loader = DataLoader(
            GraphDataset(val_graphs),
            batch_size=args.batch_size,
            shuffle=True,
            #collate_fn=lambda x: x,  # 不做任何 stack
            collate_fn=GraphDataset.graph_collate_fn,
            num_workers=2,  # <--- 手动加上这个参数，建议先设为 4 或 8
            pin_memory=True  # <--- 配合 3090 使用，建议也加上
        )

        test_loader = DataLoader(
            GraphDataset(test_graphs),
            batch_size=args.batch_size,
            shuffle=True,
            #collate_fn=lambda x: x,  # 不做任何 stack
            collate_fn=GraphDataset.graph_collate_fn,
            num_workers=2,  # <--- 手动加上这个参数，建议先设为 4 或 8
            pin_memory=True  # <--- 配合 3090 使用，建议也加上
        )


        timeRun = []
        for epoch in range(args.epochs):
            start_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            self.train(train_loader,epoch)


            if torch.cuda.is_available():
                peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
                peak_memory_gb = peak_memory_bytes / (1024 ** 3)
                #print(f"\tPeak GPU Memory Usage: {peak_memory_gb:.2f} GB")


            valid_loss, valid_acc = self.eval(self.model,valid_loader)
            #print('-' * 89)
            print(f'\tEpoch: {epoch:d}\t|\tLoss: {valid_loss:.5f}(valid)\t|\tAcc: {valid_acc * 100:.2f}%(valid)')
            #print('-' * 89)


            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_epoch = epoch + 1
                best_model = copy.deepcopy(self.model)
                wait = 0
            else:
                wait += 1
                if epoch + 1 >= min_epochs and wait >= patience:
                    print(f"Early stopping at epoch {epoch + 1}, best epoch was {best_epoch}")
                    break

            if epoch < self.warmup_steps:
                self.scheduler_warmup.step()
            else:
                self.scheduler_cosine.step()


            end_time = time.perf_counter()
            run_time = end_time - start_time
            if epoch < 5:
                timeRun.append(run_time)
            print("run_time:",round(run_time, 2),"-"*139)


        total_run_time = sum(timeRun)
        num_epochs_recorded = len(timeRun)
        average_time = total_run_time / num_epochs_recorded
        #print(f"average run time a epoch: {average_time:.2f} seconds")

        path = os.path.join("model", self.eventlog)
        if not os.path.exists(path):
            os.makedirs(path)

        model_path = (
                'model/' + str(self.eventlog) + '/' +
                self.eventlog + '_' +'num_prototypes'+str(self.num_prototypes)+'_'
                'seed' + str(self.seed) +
                '_model.pkl'
        )

        # torch.save(best_model, model_path)
        # print(f"已保存模型到{model_path}")

        # 34. 用加载的最佳模型重新评估验证集（确认模型保存无误）
        valid_loss, valid_acc = self.eval(best_model, valid_loader)
        # 35. 打印最佳模型的验证集指标
        test_loss, test_acc, precision, recall, f1, roc_auc, pr_auc = self.test(best_model, test_loader)
        print('+' * 89)
        print(
            f'\tBest_Epoch: {best_epoch:d}')
        print(f'\tBest_Loss: {test_loss:.5f}(test)\t|\tBest_Acc: {test_acc * 100:.2f}%(test)\t|\tBest_Loss: {valid_loss:.5f}(valid)\t|\tBest_Acc: {valid_acc * 100:.2f}%(valid)')
        print('+' * 89)

        return test_acc, precision, recall, f1, roc_auc, pr_auc




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)





if __name__ == "__main__":



    seed = 12345
    seed_list =[42,12345,3423,523,6556]
    eventlog = ["intermediate"]
    k_data_dict = {
        "OTC": 8,
        "BPI2017": 4,
        "intermediate": 16,
        "p2p":64
    }

    for event in eventlog:
        metrics = {
            "acc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": [],
            "pr_auc": []
        }

        # 3. 遍历这 5 个不同的种子
        for i in range(5):
            print(f"======================={i}=================================")
            # 务必确保 set_seed 在这里调用，锁死当前这一轮的随机性
            set_seed(seed_list[i])
            k = k_data_dict[event]
            print(f"\n>>> Running: Dataset={event}, K={k}, Seed={seed} <<<")

            Trainer = DHTrainer(seed, event, k)
            # 假设 train_eval 返回的是 6 个浮点数
            test_acc, precision, recall, f1, roc_auc, pr_auc = Trainer.train_eval()

            metrics["acc"].append(test_acc)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)
            metrics["auc"].append(roc_auc)
            metrics["pr_auc"].append(pr_auc)
            print("-" * 60)

        # 打印当前数据集的最终统计结果
        print("\n" + "=" * 25 + f" Final Results for {event} " + "=" * 25)
        for key, values in metrics.items():
            # np.std 默认计算的是总体标准差(ddof=0)，论文中通常用样本标准差(ddof=1)
            # 考虑到样本量仅为5，建议加上 ddof=1 更加严谨
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)

            # 打印原始数据（留底用）
            print(f"{key}_raw: {[round(v, 4) for v in values]}")
            # 打印论文格式 (例如: 0.8532 ± 0.0012)
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")

        print("=" * 75 + "\n")
