import os
from pathlib import Path

import dhg
import numpy as np
import torch
import torch.nn as nn
from config import args
from newTry.OCELhg import ObjectStateUpdater, LifecycleStateUpdater, OCELHyperGraph

from newTry.utils import extract_batch_results, HypergraphProcessor, \
 LocalGlobalFusion,  TimeAwareTransformerPredictor, \
 GlobalPrototypeFiLM

device = torch.device('cuda:{}'.format(args.gpu))


def extract_rows_by_tensor(matrix, index_tensor):
    is_numpy_input = isinstance(matrix, np.ndarray)
    if is_numpy_input:
        matrix_tensor = torch.from_numpy(matrix)
    else:
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("matrix 必须是 numpy 数组或 PyTorch 张量")
        matrix_tensor = matrix

    # 检查矩阵维度
    if matrix_tensor.dim() != 2:
        raise ValueError(f"matrix 必须是2维，当前维度: {matrix_tensor.dim()}")

    # 检查索引张量维度
    if index_tensor.dim() != 1:
        raise ValueError(f"index_tensor 必须是1维，当前维度: {index_tensor.dim()}")

    # 转换索引为整数类型（避免浮点索引）
    index_tensor = index_tensor.to(torch.long)

    # 检查索引是否非负
    if (index_tensor < 0).any():
        raise ValueError("index_tensor 中的索引不能为负数")

    # 检查索引是否越界
    max_valid_index = matrix_tensor.shape[0] - 1
    if (index_tensor > max_valid_index).any():
        raise ValueError(f"索引超出矩阵行数范围！矩阵共 {matrix_tensor.shape[0]} 行，最大有效索引: {max_valid_index}")

    # 核心逻辑：提取对应行
    extracted_tensor = matrix_tensor[index_tensor]

    # 还原为输入类型（numpy数组 / PyTorch张量）
    if is_numpy_input:
        extracted_rows = extracted_tensor.numpy()
    else:
        extracted_rows = extracted_tensor

    return extracted_rows


class GatedFusionModule(nn.Module):
    """
    门控融合模块：接收两个同维度矩阵，通过门控机制自适应融合

    输入：
        x1: 第一个矩阵，shape = [batch_size, ..., feature_dim]
        x2: 第二个矩阵，shape = [batch_size, ..., feature_dim]
    输出：
        fused: 融合后的矩阵，shape = [batch_size, ..., feature_dim]
    """

    def __init__(self, feature_dim):
        super(GatedFusionModule, self).__init__()
        # 门控权重层：学习融合权重，输出维度与特征维度一致
        self.gate = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),  # 拼接x1和x2后映射到特征维度
            nn.Sigmoid()  # 输出0-1之间的权重，控制x1的贡献度
        )

    def forward(self, x1, x2):
        # 检查两个输入的维度是否一致
        assert x1.shape == x2.shape, "两个输入矩阵的维度必须完全一致！"

        # 步骤1：拼接两个输入矩阵 (batch, ..., 2*feature_dim)
        concat = torch.cat([x1, x2], dim=-1)

        # 步骤2：计算门控权重 (batch, ..., feature_dim)，值在0-1之间
        gate_weight = self.gate(concat)

        # 步骤3：门控融合：x1*权重 + x2*(1-权重)
        # gate_weight控制x1的贡献，1-gate_weight控制x2的贡献
        fused = gate_weight * x1 + (1 - gate_weight) * x2

        return fused



class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 用于计算权重的门控网络
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Linear(d_model * 2, d_model)

    def forward(self, feat_a, feat_b):
        # 计算一个 0~1 之间的权重值
        # 如果当前样本更符合拓扑特征，g 会趋向于 1；如果符合演化特征，g 会趋向于 0
        g = self.gate(torch.cat([feat_a, feat_b], dim=-1))

        # 加权融合
        combined = g * feat_a + (1 - g) * feat_b
        return combined


class Conv_hg(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super(Conv_hg, self).__init__()
        self.d_model = d_model

        self.event_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        ).to(device)

        self.object_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        ).to(device)

        self.ObjectUpdater = ObjectStateUpdater(d_model, d_model, d_model).to(device)
        self.lifeUpdater = LifecycleStateUpdater(d_model, d_model).to(device)


        self.LayerNorm = nn.LayerNorm(d_model).to(device)
        self.LayerNorm2 = nn.LayerNorm(d_model).to(device)
        self.LayerNorm3 = nn.LayerNorm(d_model).to(device)

        self.hypergraph_processor=HypergraphProcessor(d_model,d_model,device,2)

    def forward(self, object_X, event_X,OCEL_hg,  hg, total_events):
        object_X = self.object_proj(object_X)
        event_X = self.event_proj(event_X)
        OCEL_hg.set_event_X(event_X)

        updater_object_X_t, t1 = self.ObjectUpdater(OCEL_hg,object_X)
        updater_object_X = self.LayerNorm(updater_object_X_t + object_X)

        updater_object_X2_t = self.lifeUpdater(OCEL_hg,updater_object_X)
        updater_object_X2 = self.LayerNorm2(updater_object_X2_t + updater_object_X)



        X =torch.cat([event_X, updater_object_X2], dim=0)

        h = self.hypergraph_processor(X, hg, total_events) #事件更新算子

        atten = self.hypergraph_processor.hgnn_layers[1].get_attention_weights()

        return h,atten



class encoder_conv(nn.Module):
    def __init__(self, eventlog, d_model,num_prototypes):
        super(encoder_conv, self).__init__()
        self.eventlog = eventlog
        self.d_model = d_model
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        parent_dir = current_dir.parent
        parent_dir_str = str(parent_dir)
        base_data_path = os.path.join(parent_dir_str, "data", self.eventlog)
        num_path = os.path.join(base_data_path, "dim.pt")
        event_dim, object_dim = torch.load(num_path)
        print(f"event_dim{event_dim}, object_dim{object_dim}")
        self.object_mlp = nn.Sequential(
            nn.Linear(object_dim, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        ).to(device)
        self.event_mlp = nn.Sequential(
            nn.Linear(event_dim, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        ).to(device)


        #self.seq_transform = TimeTransformerPredictor(event_dim, d_model).to(device)
        self.seq_transform = TimeAwareTransformerPredictor(event_dim, d_model).to(device)

        self.global_fusion = GlobalPrototypeFiLM(d_model=d_model, num_prototypes=num_prototypes).to(device)


        self.criterion_mse = nn.SmoothL1Loss()  # 推荐用 SmoothL1 甚至比 MSE 更稳


        self.fusion = GatedFusion(d_model).to(device)

        self.Conv_hg = Conv_hg(d_model)

    def forward(self, batch_data):
        e_feat = self.event_mlp(batch_data["event_matrix"])
        o_feat = self.object_mlp(batch_data["object_matrix"])
        event_time = batch_data["event_matrix"][:, -1:]
        total_events_in_batch = batch_data["total_events"]
        batch_data["lifecycle_features"] = batch_data["lifecycle_features"].to(device)
        batch_data["lifecycle_mask"] = batch_data["lifecycle_mask"].to(device)
        batch_data["aux_time_labels"] = batch_data["aux_time_labels"].to(device)

        edge_eo_event = batch_data["sep_event_hyperedges"]
        edge_eo_object = batch_data["sep_object_hyperedges"]
        lifecycle_hyperedges = batch_data["lifecycle_hyperedges"]
        main_object = batch_data["main_object_indices"].to(device)



        OCEL_hg = OCELHyperGraph(e_feat, o_feat, edge_eo_event, edge_eo_object, lifecycle_hyperedges, main_object,
                                 event_time)
        hg = dhg.Hypergraph(batch_data["total_nodes"], batch_data["merged_edges"]).to(device)
        h,atten =  self.Conv_hg(o_feat, e_feat,OCEL_hg,  hg, total_events_in_batch)
        event_update = h[ :total_events_in_batch ]
        obj_update = h[ total_events_in_batch :  ]
        event_vec = extract_batch_results(event_update, batch_data)
        obj_vec = extract_rows_by_tensor(obj_update, main_object)

        # temp = torch.cat([obj_vec,event_vec], dim=1)
        # final_2d_vec = self.fusion(temp)
        final_2d_vec = self.fusion(obj_vec,event_vec)

        pred_time_gap, Dynamic_vector = self.seq_transform(
            batch_data["lifecycle_features"],
            batch_data["lifecycle_mask"]
        )

        aux_loss = self.criterion_mse(pred_time_gap.squeeze(), batch_data["aux_time_labels"])
        h_final, att_weights = self.global_fusion(h_local=final_2d_vec, h_traj=Dynamic_vector)

        # visualize_batch_attention(batch_data,atten)

        return h_final, aux_loss


