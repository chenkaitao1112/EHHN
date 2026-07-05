import os
from pathlib import Path
from typing import List, Tuple
import math
import dhg
import torch
import torch.nn as nn
import numpy as np
from dhg import Hypergraph
from dhg.nn import HGNNConv,UniGATConv


from config import args


device = torch.device('cuda:{}'.format(args.gpu))



def extract_batch_results(h, batch_data):
    """
    [核心修复版] 提取每个样本的特征，严防空图导致的 NaN
    h: [Total_Nodes, d_model]
    """
    # 1. 获取元数据
    event_nums = batch_data["event_nums"]
    event_offsets = batch_data["event_offsets"]

    # 2. 截取 Event 部分特征 (防止越界)
    # 这里的 h 包含了 Event 和 Object，我们只看 Event
    max_idx = min(batch_data["total_events"], h.shape[0])
    event_h = h[:max_idx]

    results = []
    for i in range(len(event_nums)):
        n_e = event_nums[i]
        start = event_offsets[i]

        # ==========================================================
        # 【核心修复】检测到没有事件 (n_e=0) 时的处理
        # 这种情况发生在 t=1 时，prev 图是空的
        # ==========================================================
        if n_e == 0:
            # 必须返回一个全 0 向量 (维度与 h 一致)
            # 绝对不能让它去跑下面的 mean()，否则必出 NaN
            zero_vec = torch.zeros(h.shape[1], device=h.device)
            results.append(zero_vec)
            continue

        # ==========================================================
        # 正常提取逻辑
        # ==========================================================
        # 提取最后 3 个 (或者 n_e 个)
        num_to_take = min(n_e, 3)

        # 计算切片索引
        slice_start = start + n_e - num_to_take
        slice_end = start + n_e

        # 提取特征
        slice_vecs = event_h[slice_start: slice_end]

        # 【双重保险】万一取出来还是空的
        if slice_vecs.shape[0] == 0:
            results.append(torch.zeros(h.shape[1], device=h.device))
        else:
            # 取平均值
            results.append(slice_vecs.mean(dim=0))

    # 堆叠成 [Batch, d_model]
    return torch.stack(results)


class HeteroHGATConv(nn.Module):
    def __init__(self, in_dim, out_dim,  dropout=0.1):
        super().__init__()

        print("初始化异质编码器")
        # 1. 异构投影矩阵 (Heterogeneous Projection)
        # 即使 in_dim 相同，两类节点也使用独立的权重空间
        self.event_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

        self.object_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

        # 2. 核心注意力层 (基于 dhg 的 UniGATConv)
        # 使用多头注意力捕捉不同的高阶关系语义
        self.gat_conv = UniGATConv1(
            in_channels=out_dim,
            out_channels=out_dim
        )

        # 3. 残差连接投影 (如果输入维度和输出维度不一致时使用)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph, total_events: int) -> torch.Tensor:
        """
        X: [Total_Nodes, In_Dim]
        hg: 构建好的 dhg.Hypergraph 对象
        total_events: 由 collate_fn 传进来的事件节点总数
        """
        # --- 步骤 1: 异构投影 (Stage 3 in your design) ---
        # 逻辑切分：前 total_events 是事件，后面是对象
        X_e = X[:total_events]
        X_o = X[total_events:]

        # 映射到统一的公共特征空间: h' = σ(W·h + b)
        H_e = self.event_proj(X_e)
        H_o = self.object_proj(X_o)

        # 拼接回完整的特征矩阵
        X_hetero = torch.cat([H_e, H_o], dim=0)

        # --- 步骤 2: 注意力卷积 (Dual-Stage Attention) ---
        # UniGATConv 内部实现了从节点到超边，再到节点的注意力聚合
        X_attn = self.gat_conv(X_hetero, hg)

        out = X_attn

        return out

    def get_attention_weights(self):
        return self.gat_conv.get_last_attention()





class TimeAwareTransformerPredictor(nn.Module):
    def __init__(self, input_dim, d_model, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super(TimeAwareTransformerPredictor, self).__init__()
        self.d_model = d_model
        self.nhead = nhead

        # 1. 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)

        # 2. 位置编码 (保留，用于捕捉次序信息)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. [关键] 可学习的时间衰减参数
        # 这一项决定了模型对时间跨度的敏感程度
        # 初始值设小一点，防止一开始就过度惩罚远距离依赖
        self.time_decay_w = nn.Parameter(torch.tensor([0.1]))
        self.time_bias_b = nn.Parameter(torch.tensor([0.0]))

        # 4. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. 回归头
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def compute_time_bias_mask(self, features):
        """
        从特征中提取 Delta T 并构建 Attention Bias 矩阵
        """
        # 假设倒数第二列是相对前一个事件的时间差
        # features: [Batch, Seq, Feat]
        # delta_t: [Batch, Seq]
        delta_t = features[:, :, -2]

        # 1. 还原为相对起点的“绝对时间”
        # 因为我们需要计算任意 i 和 j 的距离，而不仅仅是相邻的
        # cumsum 后，absolute_time[k] 表示第 k 个事件距离第 0 个事件的时间
        absolute_time = torch.cumsum(delta_t, dim=1)

        # 2. 广播计算任意两点间的时间差矩阵
        # time_i: [Batch, Seq, 1]
        # time_j: [Batch, 1, Seq]
        time_i = absolute_time.unsqueeze(2)
        time_j = absolute_time.unsqueeze(1)

        # global_time_gap: [Batch, Seq, Seq]
        # 这里的 gap 是任意两个事件之间的时间距离
        global_time_gap = time_i - time_j

        # 3. 映射为 Bias (Time-aware Attention 核心公式)
        # 逻辑：距离越远 (|dt| 越大)，Bias 越负，Attention Score 越低
        # 为了防止数值过大，建议先取 log (如果时间差是秒级，数值会很大)
        # 加上 1e-6 防止 log(0)
        log_gap = torch.log1p(torch.abs(global_time_gap))

        time_bias = -torch.abs(self.time_decay_w) * log_gap + self.time_bias_b

        # 4. 维度适配 PyTorch 的 multi-head attention
        # 需要扩展为 [Batch * nhead, Seq, Seq]
        # repeat_interleave 会把 batch 维度复制 nhead 次
        time_bias = time_bias.repeat_interleave(self.nhead, dim=0)

        # 5. 处理 Padding 带来的干扰
        # (可选) 如果 delta_t 本身在 padding 位置是 0，cumsum 后也是 0，
        # 这其实没关系，因为后面还有 src_key_padding_mask 会再次遮挡 padding

        return time_bias

    def forward(self, lifecycle_features, lifecycle_mask):
        """
        Args:
            lifecycle_features: [Batch, Max_Len, Input_Dim]
                                (倒数第2列必须是 time delta)
            lifecycle_mask: [Batch, Max_Len] (True 代表 Padding)
        """
        # A. 基础映射
        x = self.input_projection(lifecycle_features)
        x = self.pos_encoder(x)

        # B. [核心] 就地计算时间 Bias Mask
        # 这个 mask 会作为加性项直接作用于 Softmax 前的 logits
        time_bias_mask = self.compute_time_bias_mask(lifecycle_features)

        # C. Transformer 编码
        # 注意：这里我们同时传入了 mask (自定义时间偏差) 和 src_key_padding_mask (原始 padding)
        # PyTorch 会自动叠加这两个 mask
        output = self.transformer_encoder(
            x,
            mask=time_bias_mask,
            src_key_padding_mask=lifecycle_mask
        )

        # D. 提取状态与预测
        last_state = self._get_last_valid_state(output, lifecycle_mask)
        predicted_gap = self.regressor(last_state)

        return predicted_gap, last_state

    def _get_last_valid_state(self, output, mask):
        # 保持不变
        lengths = (~mask).sum(dim=1) - 1
        last_indices = lengths.unsqueeze(1).unsqueeze(2).expand(-1, -1, output.shape[-1])
        last_states = torch.gather(output, 1, last_indices).squeeze(1)
        return last_states


class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalPrototypeFiLM(nn.Module):
    def __init__(self, d_model, num_prototypes=4):
        super().__init__()
        self.d_model = d_model
        self.num_prototypes = num_prototypes
        print(f"预设原型数{self.num_prototypes}")

        # 1. 全局原型记忆库 (Learnable Memory Bank)
        # 初始化为正态分布，代表 K 个潜在的流程模式中心
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))

        # 2. 查询投影层 (Query Projection)
        # 将 [Local_Graph || Temporal_Seq] 融合后映射为 Query 向量
        self.query_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh()  # Tanh 激活有助于将特征规范化到类似原型的分布区间
        )

        # 3. FiLM 生成器 (Feature-wise Linear Modulation)
        # 根据检索到的全局原型，生成缩放因子 gamma 和平移因子 beta
        self.gamma_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 初始化 gamma 为 1 (保持原信号)，beta 为 0 (无偏移)
        # 这是一个训练 trick，让模型从“不做调制”开始学起，防止一开始梯度爆炸
        with torch.no_grad():
            self.gamma_net[-1].weight.fill_(0)
            self.gamma_net[-1].bias.fill_(1)  # output starts at 1
            self.beta_net[-1].weight.fill_(0)
            self.beta_net[-1].bias.fill_(0)  # output starts at 0

    def forward(self, h_local, h_traj):
        """
        Args:
            h_local: [Batch, d_model] - 来自超图的局部特征
            h_traj:  [Batch, d_model] - 来自 Transformer 的动态特征
        Returns:
            h_final: [Batch, d_model] - 调制后的最终特征
            att_weights: [Batch, K]   - 原型检索的注意力权重 (用于可视化/可解释性)
        """
        # --- Stage 1: Retrieval (检索全局参考) ---

        # 构建 Query: "我是这样的环境(Local)和这样的惯性(Traj)，我属于哪一类？"
        query_input = torch.cat([h_local, h_traj], dim=-1)  # [B, 2*D]
        query = self.query_proj(query_input)  # [B, D]

        # Dot-Product Attention
        # Query * Key^T
        scores = torch.matmul(query, self.prototypes.T)  # [B, K]
        # Scaled (防止维度过大导致 Softmax 梯度消失)
        scores = scores / (self.d_model ** 0.5)

        # 计算权重 (Softmax)
        att_weights = F.softmax(scores, dim=-1)  # [B, K]

        # 检索全局信息 (Weighted Sum of Values)
        h_global_ctx = torch.matmul(att_weights, self.prototypes)  # [B, D]

        # --- Stage 2: FiLM Modulation (自适应调制) ---

        # 基础特征 (Base Feature)：我们将 Local 和 Traj 简单相加作为底座
        # 你也可以用 concat + linear，但相加通常足够且省参数
        h_base = h_local + h_traj

        # 根据全局上下文生成调制参数
        gamma = self.gamma_net(h_global_ctx)  # [B, D]
        beta = self.beta_net(h_global_ctx)  # [B, D]

        # 执行调制公式: Output = gamma * Input + beta
        h_final = gamma * h_base + beta

        return h_final, att_weights




class LocalGlobalFusion(nn.Module):
    """
    消融实验模块：w/o Evolutionary Flow
    只使用 [Local Graph] + [Global Prototype]
    """
    def __init__(self, d_model, num_prototypes=4):
        super().__init__()
        self.d_model = d_model
        self.num_prototypes = num_prototypes
        print(f"消融实验模式: [Local + Global] (No Trajectory), 原型数={num_prototypes}")

        # 1. 全局原型记忆库 (不变)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))

        # 2. 查询投影层 (Query Projection) --> 【关键修改点】
        # 原始版输入是 d_model * 2 (Local + Traj)
        # 这里输入只有 d_model (Local)
        self.query_proj = nn.Sequential(
            nn.Linear(d_model, d_model), # 输入维度减半
            nn.Tanh()
        )

        # 3. FiLM 生成器 (不变)
        # 即使没有演化流，我们依然用全局原型来调制局部特征
        self.gamma_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 初始化 trick (不变)
        with torch.no_grad():
            self.gamma_net[-1].weight.fill_(0)
            self.gamma_net[-1].bias.fill_(1)
            self.beta_net[-1].weight.fill_(0)
            self.beta_net[-1].bias.fill_(0)

    def forward(self, h_local, h_traj):
        """
        Args:
            h_local: [Batch, d_model] - 空间流 (保留)
            h_traj:  [Batch, d_model] - 演化流 (被无视，仅为了接口兼容)
        """
        # --- Stage 1: Retrieval (仅基于局部状态检索) ---

        # 【修改点】Query 只来自 h_local
        # 语义：根据当前的静态图结构，去回忆这种结构通常属于哪类模式
        query = self.query_proj(h_local)  # [B, D]

        # 原型匹配
        scores = torch.matmul(query, self.prototypes.T)  # [B, K]
        scores = scores / (self.d_model ** 0.5)
        att_weights = F.softmax(scores, dim=-1)  # [B, K]

        # 获取全局上下文
        h_global_ctx = torch.matmul(att_weights, self.prototypes)  # [B, D]

        # --- Stage 2: FiLM Modulation (调制局部特征) ---

        # 生成调制参数
        gamma = self.gamma_net(h_global_ctx)
        beta = self.beta_net(h_global_ctx)

        # 【修改点】只调制 h_local
        # Output = gamma * Local + beta
        h_final = gamma * h_local + beta

        return h_final, att_weights

class GlobalPrototypeFiLM(nn.Module):
    """
    改进版 Global Prototype FiLM 模块
    设计目标：
    1. 防止 prototype 塌缩（支持 diversity loss）
    2. 使用“半硬”原型检索（Top-k soft attention）
    3. 只调制 temporal / trajectory 分支，避免污染结构流
    4. FiLM 调制幅度受控，避免负迁移
    """

    def __init__(
        self,
        d_model,
        num_prototypes=8,
        topk=2,
        temperature=0.5,
        film_scale=0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.num_prototypes = num_prototypes
        self.topk = topk
        self.temperature = temperature
        self.film_scale = film_scale

        print(
            f"[GlobalPrototypeFiLM] K={num_prototypes}, "
            f"topk={topk}, temp={temperature}, scale={film_scale}"
        )

        # -------------------------------------------------
        # 1. Prototype Memory (L2-normalized usage)
        # -------------------------------------------------
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, d_model)
        )

        # -------------------------------------------------
        # 2. Query Projection (no Tanh, keep expressiveness)
        # -------------------------------------------------
        self.query_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # -------------------------------------------------
        # 3. FiLM generators (lightweight & bounded)
        # -------------------------------------------------
        self.gamma_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.beta_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Identity-biased initialization
        with torch.no_grad():
            self.gamma_net[-1].weight.zero_()
            self.gamma_net[-1].bias.zero_()   # gamma_raw starts at 0
            self.beta_net[-1].weight.zero_()
            self.beta_net[-1].bias.zero_()

        self.last_att_weights = None

    # -------------------------------------------------
    # Prototype diversity regularizer (call externally)
    # -------------------------------------------------
    def prototype_diversity_loss(self):
        """
        Encourage prototypes to be orthogonal / diverse
        """
        P = F.normalize(self.prototypes, dim=-1)  # [K, D]
        gram = torch.matmul(P, P.T)               # [K, K]
        I = torch.eye(self.num_prototypes, device=P.device)
        return ((gram - I) ** 2).mean()

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, h_local, h_traj):
        """
        Args:
            h_local: [B, D] structural / spatial representation
            h_traj:  [B, D] temporal / evolutionary representation

        Returns:
            h_out:       [B, D] fused representation
            att_weights: [B, K] prototype attention (for analysis)
        """

        # -------------------------------------------------
        # Stage 1: Prototype Retrieval
        # -------------------------------------------------
        query_input = torch.cat([h_local, h_traj], dim=-1)  # [B, 2D]
        query = self.query_proj(query_input)                # [B, D]
        query = F.normalize(query, dim=-1)

        prototypes_norm = F.normalize(self.prototypes, dim=-1)

        scores = torch.matmul(query, prototypes_norm.T)    # [B, K]
        scores = scores / self.temperature

        # Soft attention
        att_weights = F.softmax(scores, dim=-1)

        # Top-k masking (semi-hard assignment)
        if self.topk < self.num_prototypes:
            topk_vals, topk_idx = torch.topk(att_weights, self.topk, dim=-1)
            mask = torch.zeros_like(att_weights)
            mask.scatter_(1, topk_idx, 1.0)
            att_weights = att_weights * mask
            att_weights = att_weights / (att_weights.sum(dim=-1, keepdim=True) + 1e-8)

        self.last_att_weights = att_weights.detach()

        # Retrieve global context
        h_global = torch.matmul(att_weights, prototypes_norm)  # [B, D]

        # -------------------------------------------------
        # Stage 2: Controlled FiLM (ONLY on trajectory)
        # -------------------------------------------------
        gamma_raw = self.gamma_net(h_global)
        beta_raw = self.beta_net(h_global)

        # Bounded modulation
        gamma = 1.0 + self.film_scale * torch.tanh(gamma_raw)
        beta = self.film_scale * torch.tanh(beta_raw)

        # Only modulate h_traj
        h_traj_mod = gamma * h_traj + beta

        # Final fusion (structure is anchor)
        h_out = h_local + h_traj_mod

        return h_out, att_weights

    def get_prototype_matrix(self):
        """
        【外部调用函数】
        获取归一化后的原型矩阵 [K, D]。
        你可以直接用这个矩阵计算余弦相似度热力图。
        """
        return F.normalize(self.prototypes, dim=-1)

    def get_last_batch_attention(self):
        """
        【外部调用函数】
        获取上一次 forward 过程中产生的 Prototype Attention 权重。
        Returns:
            Tensor: [Batch_Size, Num_Prototypes]
        """
        if self.last_att_weights is None:
            raise RuntimeError("请先运行一次 forward() 再调用此函数获取权重！")
        return self.last_att_weights






class HypergraphProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, device,  num_layers=2 ):
        super().__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.hgnn_layers = nn.ModuleList()
        print("启用异构超图注意力卷积")
        # 第一层
        self.hgnn_layers.append(HeteroHGATConv(input_dim, hidden_dim).to(self.device))
        # 中间层
        for _ in range(num_layers - 2):
            self.hgnn_layers.append(HeteroHGATConv(hidden_dim, hidden_dim).to(self.device))
        # 最后一层
        if num_layers > 1:
            self.hgnn_layers.append(HeteroHGATConv(hidden_dim, hidden_dim).to(self.device))

        self.residual_proj = None
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)

        self.LayerNorm = nn.LayerNorm(hidden_dim).to(self.device)
        self.LayerNorm2 = nn.LayerNorm(hidden_dim).to(self.device)
        # 针对非残差层的特征归一化（中间层卷积后使用）


    def forward(self, x, hg, total_events):
        """
        Args:
            x: [N, input_dim] 节点特征
            hg: Hypergraph 对象
        Returns:
            h: [N, hidden_dim] 卷积后的节点特征
        """

        residual = x
        h = x
        h =h.to(device)
        hg = hg.to(device)
        for i, layer in enumerate(self.hgnn_layers):
            h_temp = h
            h_new = layer(h, hg, total_events)
            h = h_new + h_temp

            if i < len(self.hgnn_layers) - 1:
                h = self.LayerNorm(h)  # 中间层卷积后先归一化
                h = F.relu(h)  # 再激活，符合「Norm->Act」的现代网络设计

        h = self.LayerNorm2(h)
        return h




class UniGATConv1(nn.Module):
    r"""The UniGAT convolution layer proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Sparse Format:

    .. math::
        \left\{
            \begin{aligned}
                \alpha_{i e} &=\sigma\left(a^{T}\left[W h_{\{i\}} ; W h_{e}\right]\right) \\
                \tilde{\alpha}_{i e} &=\frac{\exp \left(\alpha_{i e}\right)}{\sum_{e^{\prime} \in \tilde{E}_{i}} \exp \left(\alpha_{i e^{\prime}}\right)} \\
                \tilde{x}_{i} &=\sum_{e \in \tilde{E}_{i}} \tilde{\alpha}_{i e} W h_{e}
            \end{aligned}
        \right. .

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. If ``dropout <= 0``, the layer will not drop values. Defaults to ``0.5``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to ``0.2``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            atten_neg_slope: float = 0.2,
            is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.atten_dropout = nn.Dropout(drop_rate)
        self.atten_act = nn.LeakyReLU(atten_neg_slope)
        self.act = nn.ELU(inplace=True)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.atten_e = nn.Linear(out_channels, 1, bias=False)
        self.atten_dst = nn.Linear(out_channels, 1, bias=False)

        self.last_attn_score = None

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        X = self.theta(X)
        Y = hg.v2e(X, aggr="mean")
        # ===============================================
        alpha_e = self.atten_e(Y)
        e_atten_score = alpha_e[hg.e2v_src]
        e_atten_score = self.atten_dropout(self.atten_act(e_atten_score).squeeze())
        # ================================================================================
        # We suggest to add a clamp on attention weight to avoid Nan error in training.
        # e_atten_score = torch.clamp(e_atten_score, min=0.001, max=5)
        temperature = 5.0  # 试着设为 2.0, 5.0, 甚至 10.0
        e_atten_score = e_atten_score * temperature
        # ================================================================================

        self.last_attn_score = e_atten_score.detach()

        X = hg.e2v(Y, aggr="softmax_then_sum", e2v_weight=e_atten_score)

        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
        return X

    def get_last_attention(self):
        """
        返回上一次 forward 传播时计算的注意力分数。
        Returns:
            Tensor: [Num_Edges] 的注意力权重
        """
        if self.last_attn_score is None:
            raise RuntimeError("Have not run forward pass yet!")
        return self.last_attn_score
