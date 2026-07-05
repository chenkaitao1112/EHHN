import os
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from typing import Dict, Tuple, Set

from newTry.ana import transform_ocel_flat_to_tables, remove_column_prefix, delete_columns
from newTry.construct_PE import build_relation_mappings, get_primary_object_ids, build_primary_object_pe_structure, \
    sort_primary_lifecycle_edges, filter_pe_by_time, build_prefix_subgraphs, process_single_subgraph, \
    map_event_labels_to_activities, process_all_subgraphs, build_primary_object_pe_structure2, filter_pe_by_time2
from newTry.preprocess import load_jsonocel_to_dfs, encode_object_table, add_absolute_time_features, encode_event_table, \
    process_omap, transform_objects_to_features, build_object2type_mapping
from newTry.pipeline_OTC import build_and_save_split, build_and_save_split2


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


def add_duration_and_cleanup(df, start_col='start_timestamp', end_col='timestamp'):
    """
    计算结束时间与开始时间的差值，添加为新列 'duration'，并删除原始开始时间列。

    参数:
    df: pandas.DataFrame
    start_col: 开始时间列名
    end_col: 结束时间列名

    返回:
    处理后的 DataFrame
    """
    # 1. 确保时间列是 datetime 类型
    # errors='coerce' 会将无法解析的格式设为 NaT
    # 如果你的数据只有 分:秒，Pandas 会默认补全日期
    temp_start = pd.to_datetime(df[start_col], errors='coerce')
    temp_end = pd.to_datetime(df[end_col], errors='coerce')

    # 2. 计算差值（得到的是 Timedelta 对象）
    # 我们通常将其转换为秒(seconds)或分钟(minutes)以便分析
    df['duration'] = (temp_end - temp_start).dt.total_seconds()

    # 3. 删除开始时间列
    new_df = df.drop(columns=[start_col])

    return new_df


def inspect_duration_values(df, column_name='duration'):
    """
    检查 duration 列的数值分布情况
    """
    if column_name not in df.columns:
        print(f"错误: 找不到列 '{column_name}'")
        return

    print(f"--- '{column_name}' 列数据概况 ---")

    # 1. 基本统计信息
    print(df[column_name].describe())
    print("-" * 30)

    # 2. 检查 0 和 非 0 的比例
    zero_count = (df[column_name] == 0).sum()
    non_zero_count = (df[column_name] != 0).sum()
    print(f"数值为 0 的行数: {zero_count} ({zero_count / len(df):.2%})")
    print(f"数值非 0 的行数: {non_zero_count} ({non_zero_count / len(df):.2%})")
    print("-" * 30)

    # 3. 列出出现频率最高的几个值
    print("出现频率最高的数值 (Top 10):")
    print(df[column_name].value_counts().head(10))
    print("-" * 30)

    # 4. 如果你想看那些“微小但非0”的值
    small_values = df[(df[column_name] > 0) & (df[column_name] < 1)][column_name]
    if not small_values.empty:
        print(f"检测到 {len(small_values)} 行处于 0 到 1 秒之间的微小值")
        print(f"平均微小值: {small_values.mean():.4f} 秒")
    else:
        print("未检测到 0 到 1 秒之间的微小差异值。")


def flatten_relations(df_relations, target_col="object_id"):
    """
    将包含列表的关系表转换为“一关系一行”的适配形式。

    参数:
    df_relations: 原始 DataFrame，其中 target_col 列包含 list
    target_col: 需要展开的列名，默认是 "object_id"

    返回:
    扁平化后的 DataFrame (适配 build_relation_mappings 函数)
    """
    # 1. 检查列是否存在
    if target_col not in df_relations.columns:
        return df_relations

    # 2. 使用 explode 将列表展开为多行
    # 展开前: Event_1 | ['App_1', 'Offer_1']
    # 展开后:
    #   Event_1 | 'App_1'
    #   Event_1 | 'Offer_1'
    df_flat = df_relations.explode(target_col)

    # 3. 去除可能存在的空值（如果某个事件没有关联对象，explode 会生成 NaN）
    df_flat = df_flat.dropna(subset=[target_col])

    # 4. (可选) 如果对象 ID 还是带有方括号的字符串，这里可以加一步清理
    # 但根据你之前的解析逻辑，这里 object_id 应该已经是纯字符串了

    return df_flat

if __name__ == "__main__":
    list_eventlog = [
        'BPI2017'
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
        eventlog_path = parent_dir_str + "/data/" + eventlog + "/source/" + eventlog + ".csv"
        df_events, df_objects, df_relations = transform_ocel_flat_to_tables(eventlog_path)
        df_events = remove_column_prefix(df_events)
        delete_col = ["None", "Unnamed: 0","CaseID","EventID"]
        df_events = delete_columns(df_events, delete_col)
        df_relations = df_relations.rename(columns={'object_ids': 'object_id'})
        df_relations = flatten_relations(df_relations)



        objedt2type = build_object2type_mapping(df_objects)
        act2idx, idx2act = build_activity2idx(df_events)
        print(idx2act)
        exit()
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

        df_events = add_duration_and_cleanup(df_events)
        df_events_ab, event_time_dict = add_absolute_time_features(df_events)

        # pd.set_option('display.max_columns', None)
        # print(df_events_ab)
        # exit()

        #类别属性和数值属性
        categorical_evt = [
            "activity",
            "LoanGoal",
            "ApplicationType",
            "Action",
            "Accepted",
            "org:resource",
            "EventOrigin",
            "Selected"
        ]
        numeric_evt = [
            "RequestedAmount",
            "FirstWithdrawalAmount",
            "NumberOfTerms",
            "MonthlyCost",
            "CreditScore",
            "OfferedAmount",
            "duration",
            "weekday_sin",
            "weekday_cos",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "second_sin"
        ]
        #得到事件特征矩阵
        X_evt, evt2idx, feature_names_evt = transform_objects_to_features(df_events_ab, categorical_evt,
                                                                          numeric_evt, id_col="event_id")

        #事件和对象以及对象和事件的双向关联字典
        event2objects, object2events = build_relation_mappings(df_relations)
        #设置主视角对象
        primary_object_type = "application"
        primary_object_ids = get_primary_object_ids(df_objects,primary_object_type)
        #这里划分了主视角对象的子图
        filter_list = []
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
        event_dim = X_evt.shape[1]+2
        object_dim = X_obj.shape[1]
        dim = (event_dim, object_dim)
        dim_path = os.path.join(
            base_save_path,
            f"dim.pt"
        )
        torch.save(dim, dim_path)


#这里需要备注一下，OTC数据集处理完，对象特征矩阵43维，事件特征矩阵20维