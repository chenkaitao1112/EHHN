import json
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


def build_event_object_dict(df_relations):
    """
    Build event -> objects mapping.

    Input:
        df_relations: DataFrame with columns ["event_id", "object_id"]

    Output:
        event2objects: dict
            key: event_id
            value: list of related object_ids
    """
    event2objects = {}

    for _, row in df_relations.iterrows():
        eid = row["event_id"]
        oid = row["object_id"]

        if eid not in event2objects:
            event2objects[eid] = []
        event2objects[eid].append(oid)

    return event2objects


def get_primary_object_ids(df_objects, primary_object_type):
    """
    Get all object IDs of the given primary object type.

    Input:
        df_objects: DataFrame with columns ["object_id", "object_type", ...]
        primary_object_type: str

    Output:
        primary_object_ids: list of object_ids
    """
    df_sub = df_objects[df_objects["object_type"] == primary_object_type]
    primary_object_ids = df_sub["object_id"].tolist()

    return primary_object_ids

def build_relation_mappings(df_relations):
    """
    Build bi-directional mappings between events and objects.
    Optimized using pandas groupby for speed.

    Input:
        df_relations: DataFrame ["event_id", "object_id"]

    Output:
        event2objects: dict {event_id: [object_id, ...]}
        object2events: dict {object_id: [event_id, ...]} (Preserves order if df is sorted)
    """
    # 1. Event -> Objects
    # groupby 之后直接聚合为 list，再转 dict，比 iterrows 快得多
    event2objects = df_relations.groupby("event_id")["object_id"].apply(list).to_dict()

    # 2. Object -> Events (用于获取生命周期)
    object2events = df_relations.groupby("object_id")["event_id"].apply(list).to_dict()

    return event2objects, object2events



def build_primary_object_pe_structure(
        primary_object_ids,
        event2objects,
        object2events,
        object2type,
        stop_expansion_types,
        k_hop=1
):
    nodes_dict = {}
    event_object_edges = {}
    primary_lifecycle_edges = {}

    for po in primary_object_ids:
        event_set = set()
        object_set = set([po])

        if po not in object2events:
            continue
        lifecycle_events = object2events[po]



        for e in lifecycle_events:
            event_set.add(e)

        frontier_events = set(lifecycle_events)


        visited_objects = set([po])
        visited_events = set(lifecycle_events)

        for _ in range(k_hop):

            new_objects = set()
            for e in frontier_events:
                for o in event2objects.get(e, []):
                    if o not in visited_objects:
                        new_objects.add(o)

            object_set.update(new_objects)
            visited_objects.update(new_objects)


            new_events = set()
            for o in new_objects:

                o_type = object2type.get(o)
                if o_type in stop_expansion_types:

                    continue

                for e in object2events.get(o, []):
                    if e not in visited_events:
                        new_events.add(e)

            event_set.update(new_events)
            visited_events.update(new_events)


            if not new_events:
                break
            frontier_events = new_events

        nodes_dict[po] = {
            "primary_object": po,
            "events": event_set,
            "objects": object_set
        }

        eo_edges = []
        for e in event_set:
            edge = [e]
            for o in event2objects.get(e, []):
                if o in object_set:
                    edge.append(o)
            if len(edge) > 1:
                eo_edges.append(edge)
        event_object_edges[po] = eo_edges

        if len(lifecycle_events) > 0:
            primary_lifecycle_edges[po] = [lifecycle_events]
        else:
            primary_lifecycle_edges[po] = []


    return nodes_dict, event_object_edges, primary_lifecycle_edges


def build_primary_object_pe_structure2(
        primary_object_ids,
        event2objects,
        object2events,
        object2type,
        stop_expansion_types,
        k_hop=2
):
    nodes_dict = {}
    event_object_edges = {}
    primary_lifecycle_edges = {}

    for po in primary_object_ids:
        # --- 改动 1: 初始化 Hop 映射字典 ---
        # 用于存储每个节点（事件或对象）属于第几阶
        # 0 阶：主对象 + 主对象的生命周期事件
        node_hop_map = {po: 0}

        if po not in object2events:
            continue

        lifecycle_events = object2events[po]

        # 初始化集合
        event_set = set()
        object_set = set([po])

        # 初始化：主对象的一跳事件（记为 0 阶）
        for e in lifecycle_events:
            event_set.add(e)
            node_hop_map[e] = 0  # --- 标记 0 阶 ---

        frontier_events = set(lifecycle_events)

        visited_objects = set([po])
        visited_events = set(lifecycle_events)

        # 这里使用 explicit loop index (i)，方便计算深度
        for i in range(k_hop):
            # 计算当前的阶数
            # 每一轮循环包含两步：找对象(Hop N+1)，找事件(Hop N+2)
            current_obj_hop = 2 * i + 1
            current_event_hop = 2 * i + 2

            # --- 步骤 A: events -> objects (奇数阶) ---
            new_objects = set()
            for e in frontier_events:
                for o in event2objects.get(e, []):
                    if o not in visited_objects:
                        new_objects.add(o)
                        # --- 改动 2: 标记对象阶数 ---
                        node_hop_map[o] = current_obj_hop

            object_set.update(new_objects)
            visited_objects.update(new_objects)

            # --- 步骤 B: objects -> events (偶数阶) ---
            new_events = set()
            for o in new_objects:
                o_type = object2type.get(o)
                if o_type in stop_expansion_types:
                    continue

                for e in object2events.get(o, []):
                    if e not in visited_events:
                        new_events.add(e)
                        # --- 改动 3: 标记事件阶数 ---
                        node_hop_map[e] = current_event_hop

            event_set.update(new_events)
            visited_events.update(new_events)

            if not new_events:
                break
            frontier_events = new_events

        # -------------------------
        # 输出部分
        # -------------------------
        nodes_dict[po] = {
            "primary_object": po,
            "events": event_set,
            "objects": object_set,
            # --- 改动 4: 输出这个映射表 ---
            # 格式: {'event_id_1': 0, 'obj_id_A': 1, 'event_id_2': 2 ...}
            "node_hops": node_hop_map
        }

        # 下面的边构建逻辑保持原样，完全不动
        eo_edges = []
        for e in event_set:
            edge = [e]
            for o in event2objects.get(e, []):
                if o in object_set:
                    edge.append(o)
            if len(edge) > 1:
                eo_edges.append(edge)
        event_object_edges[po] = eo_edges

        if len(lifecycle_events) > 0:
            primary_lifecycle_edges[po] = [lifecycle_events]
        else:
            primary_lifecycle_edges[po] = []

    return nodes_dict, event_object_edges, primary_lifecycle_edges

def filter_pe_by_time(
        nodes_dict,
        event_object_edges,
        event_time_dict,
        primary_lifecycle_edges,  # [新增输入] 需要知道主对象的生命周期结束时间
        global_cutoff_timestamp=None  # [可选] 全局训练集截止时间
):
    """
    静态过滤：
    1. 全局截断：剔除晚于 global_cutoff_timestamp 的事件（用于划分训练集）。
    2. 生命周期优化：剔除晚于主对象'最后一个事件'的所有关联事件（因为它们对预测该对象的任何阶段都算未来信息）。
    """
    filtered_nodes_dict = {}
    filtered_event_object_edges = {}

    for po, node_info in nodes_dict.items():
        # 获取该主对象的生命周期列表
        lifecycle_lists = primary_lifecycle_edges.get(po, [])
        if not lifecycle_lists:
            continue

        # 找到该对象生命周期的“终点”时间
        # 假设 lifecycle_lists[0] 是按时间排序的事件ID列表
        last_event_id = lifecycle_lists[0][-1]
        lifecycle_end_time = event_time_dict[last_event_id]

        # 确定该对象的最终 Cutoff Time
        # 如果有全局截断时间（比如划分训练集），取两者的较小值
        if global_cutoff_timestamp is not None:
            cutoff_time = min(lifecycle_end_time, global_cutoff_timestamp)
        else:
            cutoff_time = lifecycle_end_time

        # -------------------------
        # 1. 过滤事件
        # -------------------------
        original_events = node_info["events"]
        kept_events = set()

        for e in original_events:
            # 只保留发生在 cutoff 之前的事件
            if event_time_dict[e] <= cutoff_time:
                kept_events.add(e)

        if not kept_events:
            continue

        # -------------------------
        # 2. 过滤边 (Event-Object Edges)
        # -------------------------
        kept_edges = []
        kept_objects = set([po])  # 主对象永远保留

        for edge in event_object_edges.get(po, []):
            e = edge[0]  # 第一个是 Event ID

            if e not in kept_events:
                continue

            # 保留这条边
            kept_edges.append(edge)
            # 记录涉及的对象
            for o in edge[1:]:
                kept_objects.add(o)

        # -------------------------
        # 3. 存储
        # -------------------------
        filtered_nodes_dict[po] = {
            "primary_object": po,
            "events": kept_events,
            "objects": kept_objects
        }
        filtered_event_object_edges[po] = kept_edges

    return filtered_nodes_dict, filtered_event_object_edges


def filter_pe_by_time2(
        nodes_dict,
        event_object_edges,
        event_time_dict,
        primary_lifecycle_edges,
        global_cutoff_timestamp=None
):
    """
    静态过滤：
    1. 全局截断：剔除晚于 global_cutoff_timestamp 的事件。
    2. 生命周期优化：剔除晚于主对象'最后一个事件'的所有关联事件。
    3. [新增] 同步过滤 node_hops 字典。
    """
    filtered_nodes_dict = {}
    filtered_event_object_edges = {}

    for po, node_info in nodes_dict.items():
        # 获取该主对象的生命周期列表
        lifecycle_lists = primary_lifecycle_edges.get(po, [])
        if not lifecycle_lists:
            continue

        # --- 获取原始的 hop 映射 ---
        # 如果上一步没生成，这里默认是空字典，防止报错
        original_node_hops = node_info.get("node_hops", {})

        # 找到该对象生命周期的“终点”时间
        last_event_id = lifecycle_lists[0][-1]
        lifecycle_end_time = event_time_dict[last_event_id]

        # 确定 Cutoff Time
        if global_cutoff_timestamp is not None:
            cutoff_time = min(lifecycle_end_time, global_cutoff_timestamp)
        else:
            cutoff_time = lifecycle_end_time

        # -------------------------
        # 1. 过滤事件
        # -------------------------
        original_events = node_info["events"]
        kept_events = set()

        for e in original_events:
            # 只保留发生在 cutoff 之前的事件
            if event_time_dict[e] <= cutoff_time:
                kept_events.add(e)

        if not kept_events:
            continue

        # -------------------------
        # 2. 过滤边 (Event-Object Edges)
        # -------------------------
        kept_edges = []
        kept_objects = set([po])  # 主对象永远保留

        for edge in event_object_edges.get(po, []):
            e = edge[0]  # 第一个是 Event ID

            if e not in kept_events:
                continue

            # 保留这条边
            kept_edges.append(edge)
            # 记录涉及的对象
            for o in edge[1:]:
                kept_objects.add(o)

        # -------------------------
        # 3. [新增] 过滤 Hop 标签字典
        # -------------------------
        kept_node_hops = {}

        # 逻辑很简单：只有被保留下来的事件和对象，才有资格保留它的 hop 标签
        # 我们可以把 kept_events 和 kept_objects 合并起来遍历
        all_kept_nodes = kept_events.union(kept_objects)

        for node in all_kept_nodes:
            # 只有当该节点在原始 map 中存在时才复制（安全起见）
            if node in original_node_hops:
                kept_node_hops[node] = original_node_hops[node]

        # -------------------------
        # 4. 存储
        # -------------------------
        filtered_nodes_dict[po] = {
            "primary_object": po,
            "events": kept_events,
            "objects": kept_objects,
            "node_hops": kept_node_hops  # <--- 记得把这个带上
        }
        filtered_event_object_edges[po] = kept_edges

    return filtered_nodes_dict, filtered_event_object_edges

def sort_primary_lifecycle_edges(primary_lifecycle_edges, event_time_dict):
    """
    Sort lifecycle events for each primary object by timestamp.

    Inputs:
        primary_lifecycle_edges:
            key: primary_object_id
            value: list of lists (each inner list is a lifecycle edge)

        event_time_dict:
            dict {event_id: timestamp}

    Output:
        sorted_primary_lifecycle_edges:
            same structure, but events sorted by time
    """
    sorted_primary_lifecycle_edges = {}

    for po, edge_lists in primary_lifecycle_edges.items():
        sorted_edges = []

        for edge in edge_lists:
            sorted_edge = sorted(edge, key=lambda e: event_time_dict[e])
            sorted_edges.append(sorted_edge)

        sorted_primary_lifecycle_edges[po] = sorted_edges

    return sorted_primary_lifecycle_edges


def build_prefix_subgraphs(
        nodes_dict,
        event_object_edges,
        sorted_lifecycle_edges,
        event_time_dict,
        window_size=3  # 示例k=3
):
    """
    修改版：从生命周期第一个事件开始划分，不足window_size也保留
    示例：6个事件[1,2,3,4,5,6]、k=3 → 生成[1]、[12]、[123]、[234]、[345]（标签对应2、3、4、5、6）

    Inputs:
        nodes_dict: 过滤后的节点字典 {po: {"events":..., "objects":...}}
        event_object_edges: 过滤后的事件-对象超边 {po: [[e1,o1],...]}
        sorted_lifecycle_edges: 排序后的生命周期超边 {po: [[e1,e2,...]]}
        event_time_dict: 事件-时间戳映射 {eid: timestamp}
        window_size: 目标窗口大小（k）

    Outputs:
        subgraph_nodes_list: 子图节点列表
        subgraph_eo_edges_list: 事件-对象超边列表
        subgraph_primary_edges_list: 生命周期前缀超边列表
        labels: 预测标签列表（每个前缀的下一个事件）
    """
    subgraph_nodes_list = []
    subgraph_eo_edges_list = []
    subgraph_primary_edges_list = []
    labels = []

    for po, lifecycle_lists in sorted_lifecycle_edges.items():
        if not lifecycle_lists:
            continue

        # 取主对象的生命周期事件列表（已排序）
        lifecycle = lifecycle_lists[0]
        num_events = len(lifecycle)

        # 若生命周期只有1个事件，无标签可预测，跳过
        if num_events < 2:
            continue

        # -------------------------
        # 核心修改：遍历所有可能的前缀长度（从1到num_events-1）
        # -------------------------
        for prefix_length in range(1, num_events):
            # 1. 确定当前前缀窗口（关键逻辑）
            if prefix_length <= window_size:
                # 前window_size-1个样本：前缀从第一个事件开始，长度=prefix_length
                prefix_events_list = lifecycle[0:prefix_length]
            else:
                # 超过window_size后：滑动窗口，长度固定为window_size
                start_idx = prefix_length - window_size
                prefix_events_list = lifecycle[start_idx:prefix_length]

            # 2. 预测目标：当前前缀的下一个事件
            target_event = lifecycle[prefix_length]

            # 3. 确定截止时间（前缀最后一个事件的时间，防未来信息泄露）
            last_event_in_window = prefix_events_list[-1]
            cutoff_timestamp = event_time_dict[last_event_in_window]

            # -------------------------------------------------------
            # 4. 构建子图节点（动态过滤：仅保留截止时间前的事件）
            # -------------------------------------------------------
            valid_events = set(prefix_events_list)
            valid_objects = set([po])  # 主对象始终保留

            # 从全局2-hop范围中过滤出“过去”的事件
            potential_context_events = nodes_dict[po]["events"]
            for e in potential_context_events:
                if event_time_dict[e] <= cutoff_timestamp:
                    valid_events.add(e)

            # -------------------------------------------------------
            # 5. 构建子图边（事件-对象超边）
            # -------------------------------------------------------
            current_eo_edges = []
            all_potential_edges = event_object_edges.get(po, [])

            for edge in all_potential_edges:
                eid = edge[0]
                if eid not in valid_events:
                    continue

                # 转换边并补充有效对象
                new_edge = [eid]
                for oid in edge[1:]:
                    new_edge.append(oid)
                    valid_objects.add(oid)

                if len(new_edge) > 1:
                    current_eo_edges.append(new_edge)

            # -------------------------------------------------------
            # 6. 构建生命周期前缀超边
            # -------------------------------------------------------
            current_primary_edge = [prefix_events_list]

            # -------------------------------------------------------
            # 7. 存储样本
            # -------------------------------------------------------
            subgraph_nodes_list.append({
                "primary_object":po,
                "events": list(valid_events),
                "objects": list(valid_objects)
            })
            subgraph_eo_edges_list.append(current_eo_edges)
            subgraph_primary_edges_list.append(current_primary_edge)
            labels.append(target_event)

    return (
        subgraph_nodes_list,
        subgraph_eo_edges_list,
        subgraph_primary_edges_list,
        labels
    )


def build_prefix_subgraphs2(
        nodes_dict,
        event_object_edges,
        sorted_lifecycle_edges,
        event_time_dict,
        window_size=3
):
    """
    修改版：
    1. 动态生成前缀子图。
    2. [新增] 同步生成子图对应的 node_hops 字典。
    """
    subgraph_nodes_list = []
    subgraph_eo_edges_list = []
    subgraph_primary_edges_list = []
    subgraph_node_hops_list = []  # [新增] 用于存储每个样本的 hop 字典
    labels = []

    for po, lifecycle_lists in sorted_lifecycle_edges.items():
        if not lifecycle_lists:
            continue

        # --- 获取该主对象的原始全局 Hop 字典 ---
        # 如果前面步骤没生成，这里给个空字典防止报错
        global_hop_map = nodes_dict[po].get("node_hops", {})

        lifecycle = lifecycle_lists[0]
        num_events = len(lifecycle)

        if num_events < 2:
            continue

        for prefix_length in range(1, num_events):
            # 1. 确定当前前缀窗口
            if prefix_length <= window_size:
                prefix_events_list = lifecycle[0:prefix_length]
            else:
                start_idx = prefix_length - window_size
                prefix_events_list = lifecycle[start_idx:prefix_length]

            # 2. 预测目标
            target_event = lifecycle[prefix_length]

            # 3. 确定截止时间
            last_event_in_window = prefix_events_list[-1]
            cutoff_timestamp = event_time_dict[last_event_in_window]

            # 4. 构建子图节点 (Events)
            valid_events = set(prefix_events_list)

            potential_context_events = nodes_dict[po]["events"]
            for e in potential_context_events:
                if event_time_dict[e] <= cutoff_timestamp:
                    valid_events.add(e)

            # 5. 构建子图边 & 收集对象
            valid_objects = set([po])  # 主对象始终保留
            current_eo_edges = []

            all_potential_edges = event_object_edges.get(po, [])

            for edge in all_potential_edges:
                eid = edge[0]
                if eid not in valid_events:
                    continue

                new_edge = [eid]
                for oid in edge[1:]:
                    new_edge.append(oid)
                    valid_objects.add(oid)  # 收集有效的对象

                if len(new_edge) > 1:
                    current_eo_edges.append(new_edge)

            # -------------------------------------------------------
            # 6. [新增核心逻辑] 构建当前子图的 Hop 字典
            # -------------------------------------------------------
            current_subgraph_hops = {}

            # 合并当前子图里所有的节点 (Event + Object)
            all_valid_nodes = valid_events.union(valid_objects)

            for node in all_valid_nodes:
                # 从全局字典里查该节点的 hop，如果查不到(理论不该发生)默认设为 0
                if node in global_hop_map:
                    current_subgraph_hops[node] = global_hop_map[node]
                else:
                    # 容错处理：主对象和前缀里的事件肯定是 0
                    current_subgraph_hops[node] = 0

            # 7. 构建生命周期前缀超边
            current_primary_edge = [prefix_events_list]

            # 8. 存储所有数据
            subgraph_nodes_list.append({
                "events": list(valid_events),
                "objects": list(valid_objects)
            })
            subgraph_eo_edges_list.append(current_eo_edges)
            subgraph_primary_edges_list.append(current_primary_edge)

            # [新增] 存入 hop 列表
            subgraph_node_hops_list.append(current_subgraph_hops)

            labels.append(target_event)

    return (
        subgraph_nodes_list,
        subgraph_eo_edges_list,
        subgraph_primary_edges_list,
        subgraph_node_hops_list,  # [新增返回值]
        labels
    )

def process_single_subgraph(
        nodes_dict,  # {"primary_object": "...", "events": [...], "objects": [...]}
        eo_edges,  # [[e1, o1, o2...], ...]
        primary_edges,  # [[e1, e2...]]
        target_event_id,
        X_evt, evt2idx,
        X_obj, obj2idx,
):
    """
    返回:
    1. 特征矩阵
    2. 统一索引边 (local_eo_edges) -> 用于全图 GNN
    3. 分离索引边 (sep_eo_evt_indices, sep_eo_obj_indices_list) -> 用于二分图/异构图
    4. 主对象索引
    """

    # 1. 获取原始 ID
    raw_event_ids = nodes_dict["events"]
    raw_object_ids = nodes_dict["objects"]
    po_id = nodes_dict.get("primary_object")

    num_events = len(raw_event_ids)

    # 1.5 计算主对象在局部对象矩阵中的索引 (相对索引)
    if po_id in raw_object_ids:
        po_local_idx = raw_object_ids.index(po_id)
    else:
        po_local_idx = 0

        # 2. 提取子图特征矩阵
    global_evt_indices = [evt2idx[eid] for eid in raw_event_ids]
    global_obj_indices = [obj2idx[oid] for oid in raw_object_ids]

    if isinstance(X_evt, torch.Tensor):
        sub_X_evt = X_evt[global_evt_indices]
    else:
        sub_X_evt = torch.FloatTensor(X_evt[global_evt_indices])

    if isinstance(X_obj, torch.Tensor):
        sub_X_obj = X_obj[global_obj_indices]
    else:
        sub_X_obj = torch.FloatTensor(X_obj[global_obj_indices])

    # -------------------------------------------------------
    # 3. 构建索引映射 (准备两套)
    # -------------------------------------------------------

    # A. 相对索引映射 (Relative Index Map)
    # 直接对应 sub_X_evt 和 sub_X_obj 的行号，不加偏移
    evt_rel_map = {eid: i for i, eid in enumerate(raw_event_ids)}
    obj_rel_map = {oid: i for i, oid in enumerate(raw_object_ids)}

    # B. 统一索引映射 (Unified Index Map)
    # 用于 local_eo_edges，对象索引需要加上事件数量
    # (实际上可以复用 evt_rel_map，但 obj 需要新算)
    unified_obj_map = {oid: i + num_events for i, oid in enumerate(raw_object_ids)}

    # -------------------------------------------------------
    # 4. 转换边列表
    # -------------------------------------------------------

    # === 4.1 事件-对象超边 (两套逻辑) ===
    local_eo_edges = []  # 逻辑1：统一索引
    sep_eo_evt_indices = []  # 逻辑2：分离索引 - 事件部分
    sep_eo_obj_indices_list = []  # 逻辑2：分离索引 - 对象部分 (List of Lists)

    for edge in eo_edges:
        eid = edge[0]

        # 必须确保事件在当前子图中
        if eid not in evt_rel_map:
            continue

        # --- 逻辑 1: 构建统一索引边 (Flat) ---
        # 结构: (evt_idx, obj_idx_offset_1, obj_idx_offset_2...)
        unified_edge = [evt_rel_map[eid]]  # 事件索引一样

        # --- 逻辑 2: 准备分离数据 ---
        current_rel_obj_idxs = []

        for oid in edge[1:]:
            if oid in obj_rel_map:
                # 统一索引 (加偏移)
                unified_edge.append(unified_obj_map[oid])
                # 相对索引 (不加偏移)
                current_rel_obj_idxs.append(obj_rel_map[oid])

        # 只有当这条边至少连接了一个有效对象时，才保存
        if len(unified_edge) > 1:
            # 保存统一边
            local_eo_edges.append(tuple(unified_edge))

            # 保存分离边 (一一对应)
            # 列表第 i 项事件 对应 列表第 i 项对象集合
            sep_eo_evt_indices.append(evt_rel_map[eid])
            sep_eo_obj_indices_list.append(current_rel_obj_idxs)

    # === 4.2 生命周期超边 (仅事件，无需分离逻辑) ===
    local_primary_edges = []
    for edge in primary_edges:
        mapped_edge = [evt_rel_map[n] for n in edge if n in evt_rel_map]
        if len(mapped_edge) > 0:
            local_primary_edges.append(tuple(mapped_edge))

    # -------------------------------------------------------
    # 5. 返回结果 (新增两个返回值)
    # -------------------------------------------------------
    return (
        sub_X_evt,
        sub_X_obj,
        local_eo_edges,
        local_primary_edges,
        po_local_idx,
        sep_eo_evt_indices,  # <--- 新增
        sep_eo_obj_indices_list  # <--- 新增
    )


def process_all_subgraphs(
        subgraph_nodes_list: List[Dict[str, List[str]]],
        subgraph_eo_edges_list: List[List[List[str]]],
        subgraph_primary_edges_list: List[List[List[str]]],
        labels: List[str],
        X_evt: torch.Tensor,
        evt2idx: Dict[str, int],
        X_obj: torch.Tensor,
        obj2idx: Dict[str, int],
        df_events: pd.DataFrame,
        act2idx: Dict[str, int],
        event_time_dict: Dict[str, Any],
        skip_invalid: bool = True
) -> Tuple[List[Dict[str, Any]], List[int], List[int], Dict[str, str]]:
    """
    遍历并处理所有子图，整合节点特征、边信息及时间特征。
    """
    # 1. 构建全局事件ID到活动类型的映射
    df_events_unique = df_events.drop_duplicates(subset=["event_id"], keep="first")
    event2activity = dict(zip(df_events_unique["event_id"], df_events_unique["activity"]))

    processed_subgraphs = []
    invalid_indices = []
    valid_activity_labels = []
    valid_time_X_list = []  # 仅存储有效样本的时间矩阵
    valid_aux_time_labels = []

    # 校验输入长度
    input_lengths = [len(subgraph_nodes_list), len(subgraph_eo_edges_list),
                     len(subgraph_primary_edges_list), len(labels)]
    if len(set(input_lengths)) != 1:
        raise ValueError(f"输入列表长度不一致！{input_lengths}")

    for idx in range(len(subgraph_nodes_list)):
        # 提取单个子图原始数据
        single_nodes = subgraph_nodes_list[idx]
        single_eo_edges = subgraph_eo_edges_list[idx]
        single_primary_edges = subgraph_primary_edges_list[idx]
        single_target_event = labels[idx]

        # A. 基础处理：节点索引映射与子矩阵提取
        sub_X_evt, sub_X_obj, local_eo_edges, local_primary_edges,po_local_idx,sep_eo_evt_indices, sep_eo_obj_indices_list   = process_single_subgraph(
            nodes_dict=single_nodes,
            eo_edges=single_eo_edges,
            primary_edges=single_primary_edges,
            target_event_id=single_target_event,
            X_evt=X_evt,
            evt2idx=evt2idx,
            X_obj=X_obj,
            obj2idx=obj2idx
        )

        # B. 确定标签与活动类型
        activity = event2activity.get(single_target_event, "未知")
        label = act2idx.get(activity, -1)


        # C. 有效性校验 (核心：如果无效，直接 continue，不进入后续列表)
        is_invalid = False
        if skip_invalid:
            if (label == -1
                    or sub_X_evt.shape[0] == 0
                    or sub_X_obj.shape[0] == 0
                    or (len(local_eo_edges) == 0 and len(local_primary_edges) == 0)):
                is_invalid = True

        if is_invalid:
            invalid_indices.append(idx)
            continue

        aux_time_gap = calculate_aux_time_label(
            prefix_event_ids=single_nodes["events"],
            target_event_id=single_target_event,
            event_time_dict=event_time_dict,
            log_transform=True
        )
        valid_aux_time_labels.append(aux_time_gap)

        # D. 计算时间特征 (仅针对有效样本)
        # 确保传入的 event_ids 顺序与 sub_X_evt 的行顺序一致
        X_time = build_ocpm_time_matrix_aligned(
            event_ids=single_nodes["events"],
            single_primary_edges=single_primary_edges,
            event_time_dict=event_time_dict
        )
        valid_time_X_list.append(X_time)

        # E. 暂存有效样本
        processed_subgraph = {
            # --- 矩阵 ---
            "event_matrix": sub_X_evt,
            "object_matrix": sub_X_obj,

            # --- 统一逻辑 (Merged) ---
            "local_eo_edges": local_eo_edges,           # 包含 offset
            "local_primary_edges": local_primary_edges,

            # --- 分离逻辑 (Separated) ---
            "sep_eo_evt_indices": sep_eo_evt_indices,           # [e_idx1, e_idx2...] (相对索引)
            "sep_eo_obj_indices_list": sep_eo_obj_indices_list, # [[o_idxA, o_idxB], [o_idxC]...] (相对索引)

            # --- 其他信息 ---
            "primary_object_idx": po_local_idx,
            "target_event_id": single_target_event,
            "label": label,
            "activity": activity,
            "aux_time_label": aux_time_gap,
        }
        #print(processed_subgraph["local_primary_edges"],"++++++++++++++++++++++++++")
        processed_subgraphs.append(processed_subgraph)
        valid_activity_labels.append(label)

    # 2. 全局归一化并拼接特征
    if len(processed_subgraphs) > 0:
        # 假设 normalize_relative_time_features 已定义
        # 它会对整个 batch 的时间特征进行 Z-Score 或 MinMax 归一化
        time_X_list_updated, _ = normalize_relative_time_features(valid_time_X_list)

        for i, subgraph in enumerate(processed_subgraphs):
            X_evt = subgraph["event_matrix"]
            X_time_np = time_X_list_updated[i]

            # 转为 Tensor 并移动到相同设备
            X_time_tensor = torch.from_numpy(X_time_np).to(X_evt.device).float()

            # 在特征维度 (dim=1) 进行拼接: [num_events, feat_dim + 2]
            subgraph["event_matrix"] = torch.cat([X_evt, X_time_tensor], dim=1)

    print(f"处理完成：总计 {len(labels)}，有效 {len(processed_subgraphs)}，跳过 {len(invalid_indices)}")
    return processed_subgraphs, valid_activity_labels, invalid_indices, event2activity


def generate_multi_scale_masks(
        node_ids: List[str],
        hop_map: Dict[str, int],
        device: torch.device,
        max_k: int = 4
) -> Dict[int, torch.Tensor]:
    """
    为一组节点生成不同阶数下的掩码。
    返回格式: {1: Tensor(N,1), 2: Tensor(N,1), ..., 4: Tensor(N,1)}
    """
    # 1. 提取这组节点对应的 hop 值列表 (保持顺序)
    # 如果 map 里找不到 (理论不该发生)，默认给 0 (保留)
    hops = [hop_map.get(nid, 0) for nid in node_ids]

    # 转为 Tensor
    hops_tensor = torch.tensor(hops, device=device, dtype=torch.long)

    mask_dict = {}
    for k in range(1, max_k + 1):
        # 核心逻辑：小于等于 k 的节点设为 1.0，否则 0.0
        # unsqueeze(1) 是为了变成 [N, 1] 形状，方便后续直接与特征矩阵 [N, D] 广播相乘
        mask = (hops_tensor <= k).float().unsqueeze(1)
        mask_dict[k] = mask

    return mask_dict


def process_all_subgraphs2(
        subgraph_nodes_list: List[Dict[str, List[str]]],
        subgraph_eo_edges_list: List[List[List[str]]],
        subgraph_primary_edges_list: List[List[List[str]]],
        subgraph_node_hops_list: List[Dict[str, int]],  # [新增输入] Hop 字典列表
        labels: List[str],
        X_evt: torch.Tensor,
        evt2idx: Dict[str, int],
        X_obj: torch.Tensor,
        obj2idx: Dict[str, int],
        df_events: pd.DataFrame,
        act2idx: Dict[str, int],
        event_time_dict: Dict[str, Any],
        skip_invalid: bool = True,
        max_hop: int = 4  # [新增] 最大阶数
) -> Tuple[List[Dict[str, Any]], List[int], List[int], Dict[str, str]]:
    """
    遍历并处理所有子图，整合特征，并预计算多阶掩码。
    """
    # 1. 构建全局事件ID到活动类型的映射
    df_events_unique = df_events.drop_duplicates(subset=["event_id"], keep="first")
    event2activity = dict(zip(df_events_unique["event_id"], df_events_unique["activity"]))

    processed_subgraphs = []
    invalid_indices = []
    valid_activity_labels = []
    valid_time_X_list = []
    valid_aux_time_labels = []

    # 校验输入长度
    input_lengths = [len(subgraph_nodes_list), len(subgraph_eo_edges_list),
                     len(subgraph_primary_edges_list), len(labels),
                     len(subgraph_node_hops_list)]  # 校验 hop list 长度
    if len(set(input_lengths)) != 1:
        raise ValueError(f"输入列表长度不一致！{input_lengths}")

    for idx in range(len(subgraph_nodes_list)):
        # 提取单个子图数据
        single_nodes = subgraph_nodes_list[idx]
        single_eo_edges = subgraph_eo_edges_list[idx]
        single_primary_edges = subgraph_primary_edges_list[idx]
        single_hops = subgraph_node_hops_list[idx]  # [获取当前样本的 Hop 字典]
        single_target_event = labels[idx]

        # A. 基础处理：提取特征矩阵
        # 注意：process_single_subgraph 必须保证返回的 sub_X_evt 行顺序与 single_nodes["events"] 一致
        sub_X_evt, sub_X_obj, local_eo_edges, local_primary_edges = process_single_subgraph(
            nodes_dict=single_nodes,
            eo_edges=single_eo_edges,
            primary_edges=single_primary_edges,
            target_event_id=single_target_event,
            X_evt=X_evt,
            evt2idx=evt2idx,
            X_obj=X_obj,
            obj2idx=obj2idx
        )

        # B. 确定标签
        activity = event2activity.get(single_target_event, "未知")
        label = act2idx.get(activity, -1)

        # C. 有效性校验
        is_invalid = False
        if skip_invalid:
            if (label == -1
                    or sub_X_evt.shape[0] == 0
                    or sub_X_obj.shape[0] == 0
                    or (len(local_eo_edges) == 0 and len(local_primary_edges) == 0)):
                is_invalid = True

        if is_invalid:
            invalid_indices.append(idx)
            continue

        # -------------------------------------------------------------
        # [新增核心逻辑] D. 预计算多阶掩码字典 (Pre-computed Multi-scale Masks)
        # -------------------------------------------------------------
        # 假设 sub_X_evt 和 sub_X_obj 已经在正确的 device 上 (通常是 CPU 或 GPU)
        current_device = sub_X_evt.device

        # 1. 生成事件掩码字典: {1: Mask1, 2: Mask2...}
        evt_mask_dict = generate_multi_scale_masks(
            node_ids=single_nodes["events"],  # 关键：顺序必须和 Matrix 行一致
            hop_map=single_hops,
            device=current_device,
            max_k=max_hop
        )

        # 2. 生成对象掩码字典
        obj_mask_dict = generate_multi_scale_masks(
            node_ids=single_nodes["objects"],
            hop_map=single_hops,
            device=current_device,
            max_k=max_hop
        )
        # -------------------------------------------------------------

        # E. 计算时间特征
        aux_time_gap = calculate_aux_time_label(
            prefix_event_ids=single_nodes["events"],
            target_event_id=single_target_event,
            event_time_dict=event_time_dict,
            log_transform=True
        )
        valid_aux_time_labels.append(aux_time_gap)

        X_time = build_ocpm_time_matrix_aligned(
            event_ids=single_nodes["events"],
            single_primary_edges=single_primary_edges,
            event_time_dict=event_time_dict
        )
        valid_time_X_list.append(X_time)

        # F. 暂存有效样本
        processed_subgraph = {
            "event_matrix": sub_X_evt,
            "object_matrix": sub_X_obj,
            "local_eo_edges": local_eo_edges,
            "local_primary_edges": local_primary_edges,
            "target_event_id": single_target_event,
            "label": label,
            "activity": activity,
            "aux_time_label": aux_time_gap,
            # [保存掩码字典]
            "evt_masks": evt_mask_dict,
            "obj_masks": obj_mask_dict
        }
        processed_subgraphs.append(processed_subgraph)
        valid_activity_labels.append(label)

    # 3. 全局归一化并拼接特征 (保持原逻辑)
    if len(processed_subgraphs) > 0:
        time_X_list_updated, _ = normalize_relative_time_features(valid_time_X_list)

        for i, subgraph in enumerate(processed_subgraphs):
            X_evt = subgraph["event_matrix"]
            X_time_np = time_X_list_updated[i]
            X_time_tensor = torch.from_numpy(X_time_np).to(X_evt.device).float()

            # 拼接时间特征
            subgraph["event_matrix"] = torch.cat([X_evt, X_time_tensor], dim=1)

    print(f"处理完成：总计 {len(labels)}，有效 {len(processed_subgraphs)}，跳过 {len(invalid_indices)}")
    return processed_subgraphs, valid_activity_labels, invalid_indices, event2activity



def map_event_labels_to_activities(labels, df_events):
    """
    将 build_prefix_subgraphs 输出的事件ID标签列表，转换为对应的活动类型列表

    Inputs:
        labels: list[str] → build_prefix_subgraphs 输出的labels（事件实例ID列表，如["E2","E3","E4"]）
        df_events: pd.DataFrame → 原始/预处理后的事件表（必须包含 "event_id" 和 "activity" 列）

    Outputs:
        activity_labels: list[str] → 对应的活动类型列表（如["支付","发货","支付"]）
        event2activity: dict → 事件ID→活动类型的映射字典（方便溯源，如{"E2":"支付","E3":"发货"}）
    """
    # 1. 构建事件ID→活动类型的全局映射字典（去重，确保每个事件ID只对应一个活动）
    event2activity = dict(zip(df_events["event_id"], df_events["activity"]))

    # 2. 遍历labels，批量转换为活动类型
    activity_labels = []
    for event_id in labels:
        # 兜底：若事件ID无对应活动，标记为"未知"（也可改为None/空字符串，根据需求调整）
        activity = event2activity.get(event_id, "未知")
        activity_labels.append(activity)

    return activity_labels, event2activity


def build_ocpm_time_matrix_aligned(event_ids, single_primary_edges, event_time_dict,
                                   log_transform=True, branch_padding=-1.0):
    """
    构建对齐的时间矩阵。
    特征1: dt_start (相对于生命周期起始点的时间差)
    特征2: dt_prev (相对于主线前序节点的时间差，支线节点用填充值)
    """
    n = len(event_ids)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # 1. 提取生命周期主线并按时间排序
    lifecycle_list = single_primary_edges[0] if len(single_primary_edges) > 0 else []
    sorted_lifecycle = sorted(lifecycle_list, key=lambda x: event_time_dict.get(x, 0))
    lifecycle_set = set(lifecycle_list)

    # 2. 确定时间基准 t0
    if not sorted_lifecycle:
        # 若无生命周期边，dt_start 只能退而求其次用当前子图最早时间
        t0 = min([event_time_dict.get(eid, 0) for eid in event_ids])
    else:
        t0 = event_time_dict.get(sorted_lifecycle[0], 0)

    # 3. 预计算主线的前序时间映射
    prev_time_map = {}
    for i, curr_e in enumerate(sorted_lifecycle):
        # 第一个节点的前序设为自身，即 dt_prev = 0
        prev_t = event_time_dict[sorted_lifecycle[i - 1]] if i > 0 else event_time_dict[curr_e]
        prev_time_map[curr_e] = prev_t

    # 4. 按 event_ids 顺序构造
    res_matrix = []
    for eid in event_ids:
        t_curr = event_time_dict.get(eid, 0)

        # 特征 1: dt_start
        d_start = max((t_curr - t0).total_seconds(), 0.0)

        # 特征 2: dt_prev
        if eid in lifecycle_set:
            d_prev = max((t_curr - prev_time_map[eid]).total_seconds(), 0.0)
            if log_transform:
                d_prev = np.log1p(d_prev)
        else:
            d_prev = branch_padding  # 支线填充

        if log_transform:
            d_start = np.log1p(d_start)

        res_matrix.append([d_prev, d_start])

    return np.array(res_matrix, dtype=np.float32)


def calculate_aux_time_label(prefix_event_ids, target_event_id, event_time_dict, log_transform=True):
    """
    计算辅助标签：前缀子图最后一个事件到目标事件的时间差（秒）

    Args:
        prefix_event_ids: 当前子图包含的事件ID列表
        target_event_id: 预测的目标事件ID
        event_time_dict: 全局时间字典 {event_id: timestamp}
        log_transform: 是否进行对数变换

    Returns:
        float: 变换后的时间间隔
    """
    if not prefix_event_ids:
        return 0.0

    # 1. 获取目标事件时间
    t_target = event_time_dict.get(target_event_id)
    if t_target is None:
        return 0.0

    # 2. 找到前缀中时间最晚的事件（即当前状态的“终点”）
    # 注意：这里取所有前缀事件的最大值，保证计算的是“距离下一件事还有多久”
    prefix_times = [event_time_dict.get(eid) for eid in prefix_event_ids if eid in event_time_dict]
    if not prefix_times:
        return 0.0

    t_last = max(prefix_times)

    # 3. 计算秒数差并截断（保证不为负）
    delta_seconds = (t_target - t_last).total_seconds()
    delta_seconds = max(delta_seconds, 0.0)

    # 4. 对数变换
    if log_transform:
        # np.log1p(x) = log(1 + x)
        return float(np.log1p(delta_seconds))

    return float(delta_seconds)


def build_next_lifecycle_time_diff(
        event_ids: List[str],
        single_primary_edges: List[List[str]],
        event_time_dict: Dict[str, Any],
        log_transform: bool = True,
        branch_padding: float = -1.0,
        last_event_padding: float = 0.0
) -> np.ndarray:
    """
    计算每个事件到下一个主线事件的时间差（辅助任务标签）
    Args:
        event_ids: 事件ID列表（与sub_X_evt行顺序一致）
        single_primary_edges: 生命周期超边 [[e1,e2,e3...]]
        event_time_dict: {event_id: timestamp}
        log_transform: 是否对数变换
        branch_padding: 支线事件的填充值
        last_event_padding: 最后一个主线事件的填充值（无下一个事件）
    Returns:
        next_dt: 形状[n,]的数组，每个元素是对应事件到下一个主线事件的时间差
    """
    n = len(event_ids)
    if n == 0:
        return np.zeros((0,))

    # 1. 提取并排序主线事件（和原函数逻辑一致）
    lifecycle_list = single_primary_edges[0] if len(single_primary_edges) > 0 else []
    sorted_lifecycle = sorted(lifecycle_list, key=lambda x: event_time_dict.get(x, 0))
    lifecycle_set = set(lifecycle_list)
    if not sorted_lifecycle:
        return np.full((n,), branch_padding)

    # 2. 构建主线事件的时间映射和下一个事件映射
    lifecycle_time = {eid: event_time_dict.get(eid) for eid in sorted_lifecycle}
    next_lifecycle_map = {}
    for i in range(len(sorted_lifecycle)):
        curr_e = sorted_lifecycle[i]
        if i < len(sorted_lifecycle) - 1:
            # 非最后一个主线事件：映射到下一个主线事件
            next_e = sorted_lifecycle[i + 1]
            next_lifecycle_map[curr_e] = next_e
        else:
            # 最后一个主线事件：无下一个，标记为None
            next_lifecycle_map[curr_e] = None

    # 3. 遍历每个事件计算下一个主线事件的时间差
    next_dt = []
    for eid in event_ids:
        t_curr = event_time_dict.get(eid)
        if t_curr is None:
            next_dt.append(branch_padding)
            continue

        # 找到当前事件之后的第一个主线事件
        if eid in lifecycle_set:
            # 主线事件：直接取配置的下一个主线事件
            next_e = next_lifecycle_map.get(eid)
            if next_e is None:
                # 最后一个主线事件，无下一个
                delta_next = last_event_padding
            else:
                t_next = lifecycle_time.get(next_e)
                delta_next = (t_next - t_curr).total_seconds()
                delta_next = max(delta_next, 0.0)  # 避免负数
                if log_transform:
                    delta_next = np.log1p(delta_next)
            next_dt.append(delta_next)
        else:
            # 支线事件：找当前事件之后最近的主线事件
            # 筛选出时间晚于当前事件的主线事件
            later_lifecycle = [e for e in sorted_lifecycle if lifecycle_time[e] > t_curr]
            if not later_lifecycle:
                # 无后续主线事件
                next_dt.append(branch_padding)
            else:
                # 取最近的下一个主线事件
                next_e = later_lifecycle[0]
                t_next = lifecycle_time[next_e]
                delta_next = (t_next - t_curr).total_seconds()
                delta_next = max(delta_next, 0.0)
                if log_transform:
                    delta_next = np.log1p(delta_next)
                next_dt.append(delta_next)

    return np.array(next_dt, dtype=np.float32)



def normalize_relative_time_features(time_X_list):
    """
    对存储在列表中的多个相对时间矩阵进行全局 Z-Score 归一化。

    Args:
        time_X_list: List[np.ndarray]，每个矩阵形状为 (N_i, 2)
                     第一列: log(dt_prev + 1), 第二列: log(dt_start + 1)

    Returns:
        normalized_list: 处理后的矩阵列表，数值量级缩放至 0 附近
        stats: 包含均值和标准差的字典（用于后续推理或保存模型）
    """
    if not time_X_list:
        return [], {}

    # 1. 将所有矩阵纵向拼接，形成一个巨大的“全局观测池”
    # 这样我们可以一次性算出两列特征的全局均值和标准差
    all_data = np.vstack(time_X_list)  # 形状为 (总事件数, 2)

    # 2. 计算全局统计量
    # axis=0 表示按列计算
    means = np.mean(all_data, axis=0)
    stds = np.std(all_data, axis=0)

    # 防止除以 0（针对某些极其特殊或单一的数据集）
    stds = np.where(stds == 0, 1e-8, stds)

    print(f"--- 全局时间统计完成 ---")
    print(f"Step_Interval (Log): Mean={means[0]:.4f}, Std={stds[0]:.4f}")
    print(f"Total_Duration (Log): Mean={means[1]:.4f}, Std={stds[1]:.4f}")

    # 3. 遍历列表，对每个子矩阵执行 Z-Score 归一化
    # 公式: z = (x - mu) / std
    normalized_list = []
    for mat in time_X_list:
        norm_mat = (mat - means) / stds
        normalized_list.append(norm_mat.astype(np.float32))

    stats = {
        "means": means,
        "stds": stds
    }

    return normalized_list, stats