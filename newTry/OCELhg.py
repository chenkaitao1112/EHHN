import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add, scatter_softmax


class OCELHyperGraph():
    def __init__(self, event_X,object_X,edge_eo_event,edge_eo_object,lifecycle_hyperedges,main_object,event_time):
        self.event_X = event_X  #这里是事件特征矩阵
        self.object_X = object_X  #这里是对象特征矩阵，两个特征矩阵是独立索引的
        self.edge_eo_event = edge_eo_event  #这里是事件对象超边的事件节点索引 [[E_global1], [E_global2]，...]
        self.edge_eo_object = edge_eo_object  #这里是事件对象超边的对象节点索引 [[O_global_1, ...], ...]
        self.lifecycle_hyperedges = lifecycle_hyperedges   #这里是生命周期超边[[E1, E2, E3...], ...]
        self.main_object = main_object  #这里是主视角对象的索引 (Tensor形式方便索引)
        self.event_time = event_time #一个N×1的矩阵，存着每个事件和上一个事件的时间差，只有和主视角直接相连的事件有，其他的用-1填充

    def set_object_X(self, object_X):
        self.object_X = object_X

    def set_event_X(self, event_X):
        self.event = event_X


class ObjectStateUpdater(nn.Module):
    def __init__(self, obj_dim, event_dim, hidden_dim):
        """
        Args:
            obj_dim: 对象特征维度 (Object State Dimension)
            event_dim: 事件特征维度 (Event Attribute Dimension)
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        print("初始化一层事件冲击算子")
        # 1. 氛围提取器 (Object -> Atmosphere)
        # 将对象特征映射一下，准备聚合成氛围
        self.atmosphere_proj = nn.Linear(obj_dim, hidden_dim)

        # 2. 冲击计算器 (Event + Atmosphere -> Impulse)
        # 结合“事件本身的特征”和“它所处的对象氛围”，计算这个事件对他人的冲击力
        self.impulse_net = nn.Sequential(
            nn.Linear(event_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obj_dim)  # 输出维度与对象一致，方便更新
        )

        # 3. 状态更新门 (Old State + Impulse -> New State)
        # 使用 GRUCell 保证状态更新的平滑性，避免梯度爆炸
        self.update_gate = nn.GRUCell(obj_dim, obj_dim)

    def _process_edges_to_tensor(self, ocel_graph):
        """
        [内部工具函数]
        将 OCELHyperGraph 中的 list-of-lists 索引转换为 PyTorch 的 COO 张量。
        为了效率，建议这一步放在 DataLoader 里做，但为了符合你的类定义，我写在这里。
        """
        event_indices = []
        object_indices = []

        # 遍历每条超边（即每个事件）
        # 假设 edge_eo_object[i] 存的是第 i 个事件连接的所有对象索引
        for e_idx, connected_objects in enumerate(ocel_graph.edge_eo_object):
            # 构造源节点(对象)到目标节点(事件)的映射
            # 这里 e_idx 就是事件在 event_X 中的行号
            num_objs = len(connected_objects)
            event_indices.extend([e_idx] * num_objs)
            object_indices.extend(connected_objects)

        device = ocel_graph.event_X.device
        return (torch.tensor(event_indices, dtype=torch.long, device=device),
                torch.tensor(object_indices, dtype=torch.long, device=device))

    def forward(self, ocel_graph,object_X):
        """
        Args:
            ocel_graph: 实例化后的 OCELHyperGraph 对象
        Returns:
            updated_object_X: 更新后的对象特征矩阵
        """
        # 0. 数据解包与格式转换
        # 我们需要 [Source_Obj, Target_Event] 的索引对进行并行计算
        # event_idx: [N_edges], object_idx: [N_edges]
        event_idx, object_idx = self._process_edges_to_tensor(ocel_graph)

        x_event = ocel_graph.event_X
        x_object = object_X

        # =======================================================
        # Step 1: 聚合对象信息，得到“事件氛围” (Atmosphere)
        # Logic: Object -> Event
        # =======================================================

        # 先把对象特征投影一下
        h_obj_proj = self.atmosphere_proj(x_object)

        # 使用 scatter_mean 进行池化
        # 含义：对于每个 event_idx，把它连接的所有 object_idx 的特征取平均
        # 结果维度: [Num_Events, hidden_dim]
        c_atmosphere = scatter_mean(h_obj_proj[object_idx], event_idx, dim=0, dim_size=x_event.size(0))

        # =======================================================
        # Step 2: 结合事件特征，计算“事件冲击” (Event Impulse)
        # Logic: (Event | Atmosphere) -> Impulse
        # =======================================================

        # 拼接：事件特征 + 刚才算出来的氛围
        # 结果维度: [Num_Events, event_dim + hidden_dim]
        event_context = torch.cat([x_event, c_atmosphere], dim=1)

        # 计算冲击力 (Impulse)
        # 结果维度: [Num_Events, obj_dim]
        # 物理含义：这个事件发生后，它对参与者产生的一个更新向量
        h_impulse = self.impulse_net(event_context)

        # =======================================================
        # Step 3: 将冲击散发回对象，更新所有对象 (Global Update)
        # Logic: Impulse -> Objects
        # =======================================================

        # 将事件的冲击力，发送回所有参与该事件的对象
        # 使用 scatter_add，因为一个对象可能同时参与多个事件（或者在子图中有多条连边）
        # 结果维度: [Num_Objects, obj_dim]
        total_effect_on_object = scatter_add(h_impulse[event_idx], object_idx, dim=0, dim_size=x_object.size(0))

        # 使用 GRU 进行软更新：输入是累积的冲击力，隐状态是对象原本的状态
        new_object_X = self.update_gate(total_effect_on_object, x_object)

        return new_object_X, h_impulse


class LifecycleStateUpdater(nn.Module):
    def __init__(self, obj_dim, event_dim):
        """
        专门用于利用生命周期超边更新主视角对象的模块
        """
        super().__init__()
        print("初始化一层周期更新算子")

        # 1. 投影层
        self.lifecycle_proj = nn.Sequential(
            nn.Linear(event_dim, obj_dim),
            nn.ReLU()
        )

        # 2. 融合更新层
        # Input: Lifecycle Profile, Hidden: Current Object State (from Spatial Update)
        self.update_gate = nn.GRUCell(obj_dim, obj_dim)

    def _parse_lifecycle_edges(self, ocel_graph):
        """
        [辅助函数] 将 list 形式的 lifecycle_hyperedges 转为 Tensor 索引
        只提取 '主视角对象' 的历史
        """
        main_obj_indices = ocel_graph.main_object

        # 处理 boolean mask 情况
        if main_obj_indices.dtype == torch.bool:
            main_obj_indices = main_obj_indices.nonzero(as_tuple=False).view(-1)

        target_obj_list = []
        history_event_list = []

        main_objs_cpu = main_obj_indices.cpu().tolist()

        for obj_id in main_objs_cpu:
            # 边界检查：确保 obj_id 在 lifecycle 列表范围内
            if obj_id < len(ocel_graph.lifecycle_hyperedges):
                hist_events = ocel_graph.lifecycle_hyperedges[obj_id]
                if len(hist_events) > 0:
                    num_events = len(hist_events)
                    target_obj_list.extend([obj_id] * num_events)
                    history_event_list.extend(hist_events)

        device = ocel_graph.event_X.device

        return (torch.tensor(target_obj_list, dtype=torch.long, device=device),
                torch.tensor(history_event_list, dtype=torch.long, device=device))

    def forward(self, ocel_graph,object_X):
        """
        Args:
            ocel_graph: 图对象，其中 ocel_graph.object_X 已经是经过 Spatial Update 更新过的状态
        Returns:
            final_object_X: 再次更新后的对象特征矩阵
        """
        # 1. 直接从图中获取当前对象状态 (这就是上一步 Spatial Update 的结果)
        current_object_X = object_X

        # 2. 解析索引：只拿主对象的历史
        lc_obj_idx, lc_evt_idx = self._parse_lifecycle_edges(ocel_graph)

        # 如果没有历史数据，直接返回当前状态
        if lc_obj_idx.size(0) == 0:
            return current_object_X

        # 3. 提取历史事件特征并投影
        # [N_total_history, event_dim] -> [N_total_history, obj_dim]
        proj_event_feats = self.lifecycle_proj(ocel_graph.event_X[lc_evt_idx])

        # 4. 平均聚合 (Mean Pooling) -> 得到“生命周期画像”
        # scatter_mean 会自动处理维度，结果是 [Num_Objects, obj_dim]
        # 非主对象的行全是 0
        lifecycle_profile = scatter_mean(
            proj_event_feats,
            lc_obj_idx,
            dim=0,
            dim_size=current_object_X.size(0)
        )

        # 5. 融合更新 (GRU)
        # 此时 lifecycle_profile 只有主对象有值，辅助对象为 0
        # GRU(input=profile, h=current_state)
        # 注意：这里所有对象都会过一遍 GRU，辅助对象输入的是 0 向量，但这会改变其状态(GRU bias)
        # 所以后面必须 Mask 还原
        updated_values = self.update_gate(lifecycle_profile, current_object_X)

        # 6. Mask 还原：确保只有主对象被更新，辅助对象保持 Spatial Update 的结果
        mask = torch.zeros(current_object_X.size(0), 1, device=current_object_X.device)

        # 设置 Mask
        if ocel_graph.main_object.dtype == torch.bool:
            mask[ocel_graph.main_object] = 1.0
        else:
            mask[ocel_graph.main_object] = 1.0

        # 融合：主对象用新算的值，辅助对象保持原值
        final_object_X = mask * updated_values + (1 - mask) * current_object_X

        return final_object_X



import torch
import torch.nn as nn
import math
from torch_scatter import scatter_mean, scatter_add  # 假设你使用的是 torch_scatter


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        标准的正弦位置编码，用于给序列注入顺序信息
        """
        super().__init__()
        # 创建一个足够长的 PE 矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer，不参与梯度更新，但会随模型保存
        self.register_buffer('pe', pe)

    def forward(self, x, pos_indices):
        """
        x: [N, d_model] (这里不需要，只用维度做校验)
        pos_indices: [N] 每个事件在其生命周期中的相对位置索引 (0, 1, 2...)
        """
        # 根据索引查表
        return self.pe[pos_indices]

