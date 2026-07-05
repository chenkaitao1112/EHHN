import os
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from typing import Dict, Tuple, Set
from newTry.construct_PE import build_relation_mappings, get_primary_object_ids, build_primary_object_pe_structure, \
    sort_primary_lifecycle_edges, filter_pe_by_time, build_prefix_subgraphs, process_single_subgraph, \
    map_event_labels_to_activities, process_all_subgraphs, build_prefix_subgraphs2, process_all_subgraphs2, \
    build_primary_object_pe_structure2, filter_pe_by_time2
from newTry.preprocess import load_jsonocel_to_dfs, encode_object_table, add_absolute_time_features, encode_event_table, \
    process_omap, transform_objects_to_features, build_object2type_mapping


def split_primary_objects_train_val_test(
    po_ids,
    test_ratio=0.2,
    val_ratio=0.1,   # 注意：这是“相对于 train 的比例”
    seed=42
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    按 primary object id 划分 Train / Val / Test

    Args:
        po_ids: iterable of primary object ids
        test_ratio: 测试集比例（相对于总量）
        val_ratio: 验证集比例（相对于 train 部分）
        seed: 随机种子

    Returns:
        train_pos, val_pos, test_pos (三者互不重叠)
    """
    rng = np.random.default_rng(seed)
    po_ids = list(po_ids)
    rng.shuffle(po_ids)

    # 1. Test
    n_total = len(po_ids)
    n_test = int(n_total * test_ratio)
    test_pos = set(po_ids[:n_test])

    # 2. Train + Val
    remaining = po_ids[n_test:]

    n_val = int(len(remaining) * val_ratio)
    val_pos = set(remaining[:n_val])
    train_pos = set(remaining[n_val:])

    return train_pos, val_pos, test_pos

def filter_dicts_by_keys(keys, *dicts):
    """
    根据 key 集合，同时过滤多个 dict
    """
    return [
        {k: d[k] for k in keys if k in d}
        for d in dicts
    ]

def build_and_save_split(
    split_name: str,
    nodes_dict,
    eo_edges_dict,
    lifecycle_edges_dict,
    event_time_dict,
    X_evt, evt2idx,
    X_obj, obj2idx,
    df_events,
    base_save_path,
    act2idx,
    window_size=3
):
    """
    从 PO-level dict → prefix subgraphs → processed subgraphs → 保存
    """

    # 1. Prefix 子图
    subgraph_nodes_list, subgraph_eo_edges_list, \
    subgraph_primary_edges_list, labels = build_prefix_subgraphs(
        nodes_dict,
        eo_edges_dict,
        lifecycle_edges_dict,
        event_time_dict,
        window_size=window_size
    )



    # 2. 转换为模型输入
    processed_subgraphs, valid_activity_labels, \
    invalid_indices, event2activity = process_all_subgraphs(
        subgraph_nodes_list,
        subgraph_eo_edges_list,
        subgraph_primary_edges_list,
        labels,
        X_evt, evt2idx,
        X_obj, obj2idx,
        df_events,
        act2idx,
        event_time_dict,
    )



    # 3. 保存
    graph_path = os.path.join(
        base_save_path,
        f"{split_name}_Graph.pt"
    )
    label_path = os.path.join(
        base_save_path,
        f"{split_name}_label.pt"
    )

    torch.save(processed_subgraphs, graph_path)
    torch.save(valid_activity_labels, label_path)

    print(
        f"[{split_name}] 保存完成 | "
        f"子图数: {len(processed_subgraphs)} | "
        f"无效样本: {len(invalid_indices)}"
    )

    return {
        "graphs": processed_subgraphs,
        "labels": valid_activity_labels,
        "invalid_indices": invalid_indices,
        "event2activity": event2activity
    }



def build_and_save_split2(
    split_name: str,
    nodes_dict,
    eo_edges_dict,
    lifecycle_edges_dict,
    event_time_dict,
    X_evt, evt2idx,
    X_obj, obj2idx,
    df_events,
    base_save_path,
    act2idx,
    window_size=3
):
    """
    从 PO-level dict → prefix subgraphs → processed subgraphs → 保存
    """

    # 1. Prefix 子图
    subgraph_nodes_list, subgraph_eo_edges_list, \
    subgraph_primary_edges_list,subgraph_node_hops_list, labels = build_prefix_subgraphs2(
        nodes_dict,
        eo_edges_dict,
        lifecycle_edges_dict,
        event_time_dict,
        window_size=window_size
    )


    # 2. 转换为模型输入
    processed_subgraphs, valid_activity_labels, \
    invalid_indices, event2activity = process_all_subgraphs2(
        subgraph_nodes_list,
        subgraph_eo_edges_list,
        subgraph_primary_edges_list,
        subgraph_node_hops_list,
        labels,
        X_evt, evt2idx,
        X_obj, obj2idx,
        df_events,
        act2idx,
        event_time_dict,
    )


    # 3. 保存
    graph_path = os.path.join(
        base_save_path,
        f"{split_name}_Graph_4hops.pt"
    )
    label_path = os.path.join(
        base_save_path,
        f"{split_name}_label_4hops.pt"
    )

    torch.save(processed_subgraphs, graph_path)
    torch.save(valid_activity_labels, label_path)

    print(
        f"[{split_name}] 保存完成 | "
        f"子图数: {len(processed_subgraphs)} | "
        f"无效样本: {len(invalid_indices)}"
    )

    return {
        "graphs": processed_subgraphs,
        "labels": valid_activity_labels,
        "invalid_indices": invalid_indices,
        "event2activity": event2activity
    }


def build_activity2idx(df_events, activity_col="activity"):
    """
    从 df_events 中构建 activity -> integer index 的映射

    Args:
        df_events: pandas DataFrame，包含所有事件（train+val+test 的全集）
        activity_col: activity 列名

    Returns:
        act2idx: dict[str, int]
        idx2act: dict[int, str]
    """
    # 1. 取唯一 activity
    activities = df_events[activity_col].unique().tolist()

    # 2. 排序，保证可复现 & 稳定对齐
    activities = sorted(activities)

    # 3. 编号
    act2idx = {act: idx for idx, act in enumerate(activities)}
    idx2act = {idx: act for act, idx in act2idx.items()}

    return act2idx, idx2act


def process_object_id(row):
    """
    逻辑：
    - 如果是 products (商品) 或 customers (客户)，保留其原始 ID（包含具体信息）。
    - 如果是 orders/items/packages (流水号)，统一标记为 'Other'。
    """
    if row['object_type'] in keep_types:
        return row['object_id']
    else:
        return "Other"

def print_top3_subgraphs(
    nodes_dict,
    event_object_edges,
    primary_lifecycle_edges
):
    """
    仅打印前3个主对象的子图核心信息（纯打印，无任何内部调用）
    :param nodes_dict: 外部生成的节点字典
    :param event_object_edges: 外部生成的事件-对象边字典
    :param primary_lifecycle_edges: 外部生成的生命周期边字典
    """
    # 取所有主对象ID（按传入顺序），仅保留前3个
    primary_object_ids = list(nodes_dict.keys())[:3]

    if not primary_object_ids:
        print("无有效主对象数据！")
        return

    # 逐个打印前3个
    for i, po in enumerate(primary_object_ids, 1):
        print(f"\n===== 第{i}个子图 - 主对象：{po} =====")
        # 1. 事件节点（排序后打印，更易读）
        event_nodes = sorted(nodes_dict[po]["events"])
        print(f"事件节点：{event_nodes}")
        # 2. 对象节点（排序后打印）
        object_nodes = sorted(nodes_dict[po]["objects"])
        print(f"对象节点：{object_nodes}")
        # 3. 事件-对象边
        eo_edges = event_object_edges[po]
        print(f"事件-对象边：{eo_edges}")
        # 4. 生命周期边
        lifecycle_edges = primary_lifecycle_edges[po]
        print(f"生命周期边：{lifecycle_edges}")


def create_type_lookup(df, id_col, type_col):
    """
    创建一个列表，index i 对应 feature_matrix 第 i 行对象的类型名称
    """
    # 必须和 transform_objects_to_features 里的逻辑保持一致的排序
    df_sorted = df.set_index(id_col, drop=False)

    # 提取类型列转为列表
    # 结果通过: ['Item', 'Item', 'Order', 'Package', ...]
    # 这样 object_types[0] 就是第0行对象的类型
    object_types = df_sorted[type_col].tolist()

    return object_types
if __name__ == "__main__":
    list_eventlog = [
        'OTC'
    ]
    # current_file = Path(__file__).resolve()
    # current_dir = current_file.parent
    # parent_dir = current_dir.parent
    # parent_dir_str = str(parent_dir)
    #获得文件路径
    current_file = Path(os.getcwd()).resolve()
    current_dir = current_file.parent
    parent_dir_str = str(current_dir)
    for eventlog in list_eventlog:
        eventlog_path = parent_dir_str + "/data/" + eventlog + "/source/" + eventlog + ".jsonocel"
        df_events, df_objects, df_relations = load_jsonocel_to_dfs(eventlog_path)

        objedt2type = build_object2type_mapping(df_objects)

        act2idx, idx2act = build_activity2idx(df_events)


        #解析为三个表格，对象表事件表关系表
        #在对象表中的object_id里面保留几个类型对象的id，他们是有效信息，别的是无效的比如订单号
        keep_types = ['products', 'customers']
        df_objects['refined_id'] = df_objects.apply(process_object_id, axis=1)
        #类别特征
        categorical_cols = [
            "object_type",
            "refined_id"
        ]
        #数值特征
        numeric_cols = []
        #转为对象特征矩阵
        X_obj, obj2idx, feature_names_obj = transform_objects_to_features(df_objects, categorical_cols, numeric_cols)

        #增加绝对时间特征
        df_events_ab, event_time_dict = add_absolute_time_features(df_events)
        #omap这列存了这个事件关联的对象，其实和关系表重复了，所以删掉
        df_events_updated = process_omap(df_events_ab)

        #类别属性和数值属性
        categorical_evt = [
            "activity",
        ]
        numeric_evt = [
            "vmap_weight",
            "vmap_price",
            "weekday_sin",
            "weekday_cos",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "second_sin"
        ]
        #得到事件特征矩阵
        X_evt, evt2idx, feature_names_evt = transform_objects_to_features(df_events_updated, categorical_evt,
                                                                          numeric_evt, id_col="event_id")

        #事件和对象以及对象和事件的双向关联字典
        event2objects, object2events = build_relation_mappings(df_relations)
        #设置主视角对象
        primary_object_type = "items"
        primary_object_ids = get_primary_object_ids(df_objects,primary_object_type)
        #这里划分了主视角对象的子图
        filter_list = ["products","customers"]
        nodes_dict, event_object_edges, primary_lifecycle_edges = build_primary_object_pe_structure(primary_object_ids,event2objects, object2events,objedt2type,filter_list)
        #给生命周期超边排一下序
        sorted_primary_lifecycle_edges = sort_primary_lifecycle_edges(primary_lifecycle_edges, event_time_dict)
        #静态过滤
        filtered_nodes_dict, filtered_event_object_edges = filter_pe_by_time(nodes_dict, event_object_edges,event_time_dict,primary_lifecycle_edges)

        #print_top3_subgraphs(filtered_nodes_dict, filtered_event_object_edges, sorted_primary_lifecycle_edges)




        #得到主视角对象列表，用来划分训练集测试集验证集
        all_pos = filtered_nodes_dict.keys()
        train_pos, val_pos, test_pos = \
            split_primary_objects_train_val_test(
                all_pos,
                test_ratio=0.2,
                val_ratio=0.1,
                seed=42
            )
        #划分训练集测试集验证集
        (train_nodes_dict,
         train_event_object_edges,
         train_primary_lifecycle_edges) = filter_dicts_by_keys(
            train_pos,
            filtered_nodes_dict,
            filtered_event_object_edges,
            sorted_primary_lifecycle_edges
        )

        (val_nodes_dict,
         val_event_object_edges,
         val_primary_lifecycle_edges) = filter_dicts_by_keys(
            val_pos,
            filtered_nodes_dict,
            filtered_event_object_edges,
            sorted_primary_lifecycle_edges
        )

        (test_nodes_dict,
         test_event_object_edges,
         test_primary_lifecycle_edges) = filter_dicts_by_keys(
            test_pos,
            filtered_nodes_dict,
            filtered_event_object_edges,
            sorted_primary_lifecycle_edges
        )
        print("开始构建子图")
        #接下来是构建子图超图了
        base_save_path = os.path.join(parent_dir_str, "data", eventlog)
        os.makedirs(base_save_path, exist_ok=True)
        K =5
        train_data = build_and_save_split(
            "train",
            train_nodes_dict,
            train_event_object_edges,
            train_primary_lifecycle_edges,
            event_time_dict,
            X_evt, evt2idx,
            X_obj, obj2idx,
            df_events,
            base_save_path,
            act2idx,
            window_size=K
        )

        val_data = build_and_save_split(
            "val",
            val_nodes_dict,
            val_event_object_edges,
            val_primary_lifecycle_edges,
            event_time_dict,
            X_evt, evt2idx,
            X_obj, obj2idx,
            df_events,
            base_save_path,
            act2idx,
            window_size=K
        )

        test_data = build_and_save_split(
            "test",
            test_nodes_dict,
            test_event_object_edges,
            test_primary_lifecycle_edges,
            event_time_dict,
            X_evt, evt2idx,
            X_obj, obj2idx,
            df_events,
            base_save_path,
            act2idx,
            window_size=K
        )

        len_activity = len(act2idx)
        num_path = os.path.join(
            base_save_path,
            f"num.pt"
        )
        torch.save(len_activity, num_path)


        base_save_path = os.path.join(parent_dir_str, "data", eventlog)
        event_dim = X_evt.shape[1] + 2
        object_dim = X_obj.shape[1]
        dim = (event_dim, object_dim)
        dim_path = os.path.join(
            base_save_path,
            f"dim.pt"
        )
        print(dim_path)
        torch.save(dim, dim_path)


