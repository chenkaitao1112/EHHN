import os
from pathlib import Path
from typing import List, Tuple
import math
import dhg
import torch
import torch.nn as nn
import numpy as np
from dhg.nn import HGNNConv,UniGATConv
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from config import args
from newTry.encoder import encoder_conv


device = torch.device('cuda:{}'.format(args.gpu))


def merge_hypergraphs_simple(
        batch_X: List[torch.Tensor],
        batch_edge: List[List[List[int]]]
) -> Tuple[torch.Tensor, List[List[int]], List[int]]:
    """
    合并超图批次。

    参数:
        batch_X: 每个元素为 [N_i, D] 的 Tensor (节点特征)
        batch_edge: 每个元素为一个 list，包含该样本的所有超边 (每条超边是节点索引 list)

    返回:
        merged_X: 合并后的特征矩阵 [Total_N, D]
        merged_edges: 平移索引后的全局超边列表
        node_offsets: 每个样本在全局矩阵中的起始偏移量 (用于后续提取结果)
    """
    node_offsets = []
    merged_edges = []
    current_offset = 0

    # 1. 处理特征矩阵
    # 直接拼接所有样本的特征
    merged_X = torch.cat(batch_X, dim=0)

    # 2. 处理超边索引平移
    for i, (x, edges) in enumerate(zip(batch_X, batch_edge)):
        # 记录当前样本的起始位置
        node_offsets.append(current_offset)

        # 对当前样本的每一条超边进行索引平移
        for edge in edges:
            # global_edge = [idx + offset]
            # 使用列表推导式提高平移速度
            shifted_edge = [idx + current_offset for idx in edge]
            merged_edges.append(shifted_edge)

        # 累加当前样本的节点数，更新下一个样本的偏移量
        current_offset += x.shape[0]

    return merged_X, merged_edges, node_offsets


def extract_last_3_events_mean(h, simple_event_num, offsets):
    """
    从卷积后的特征矩阵 h 中提取每个样本最后三个事件的特征并做平均。

    参数:
        h: 卷积后的全局特征矩阵 [Total_Nodes, Hidden_Dim]
        simple_event_num: 列表，记录每个样本的事件数量
        offsets: 列表，记录每个样本在 h 中的起始偏移量

    返回:
        final_2d_vec: 聚合后的二维矩阵 [Batch_Size, Hidden_Dim]
    """
    vec_list = []

    for num, offset in zip(simple_event_num, offsets):
        # 1. 确定最后三个事件的索引
        # 使用 max(0, num-3) 是为了防止样本事件数少于 3 个时索引溢出
        start_idx = max(0, num - 3)
        end_idx = num  # 不包含

        # 计算在全局矩阵 h 中的索引范围
        # 因为事件在对象前面，所以直接在 offset 上加事件的相对索引即可
        indices = torch.arange(start_idx, end_idx, device=h.device) + offset

        # 2. 提取特征向量 [3, Hidden_Dim] 或 [n, Hidden_Dim] (如果 n < 3)
        last_events_features = h[indices]

        # 3. 进行平均聚合 [1, Hidden_Dim]
        # dim=0 表示对这三个事件向量取平均
        avg_feature = torch.mean(last_events_features, dim=0)

        vec_list.append(avg_feature)

    # 4. 拼成最终的二维矩阵 [Batch_Size, Hidden_Dim]
    final_2d_vec = torch.stack(vec_list, dim=0)

    return final_2d_vec

class Predictor(nn.Module):
    def __init__(self, eventlog, d_model,num_prototypes):
        super(Predictor, self).__init__()
        self.eventlog = eventlog
        self.d_model = d_model
        self.encoder =encoder_conv(self.eventlog, self.d_model,num_prototypes)
        self.device = device

        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        parent_dir = current_dir.parent
        parent_dir_str = str(parent_dir)
        base_data_path = os.path.join(parent_dir_str, "data", self.eventlog)
        num_path = os.path.join(base_data_path, "num.pt")
        self.num_activity = torch.load(num_path)

        self.feat_dim = d_model

        self.fc = nn.Linear(self.feat_dim, self.feat_dim).to(self.device)
        self.active = nn.Tanh().to(self.device)
        self.projection = nn.Linear(self.feat_dim, self.num_activity , bias=False).to(self.device)

        self.classifer = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feat_dim, self.num_activity , bias=False)
        ).to(self.device)

        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, graphs):
        global_feat,aux_loss = self.encoder(graphs)
        feat = self.active(self.fc(global_feat))  # [B, feat_dim]
        dec_logits = self.projection(feat)  # [B, activity_vocab_size]

        return dec_logits,aux_loss


