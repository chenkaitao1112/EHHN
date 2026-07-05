import os
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from typing import Dict, Tuple, Set
from newTry.construct_PE import build_relation_mappings, get_primary_object_ids, build_primary_object_pe_structure, \
    sort_primary_lifecycle_edges, filter_pe_by_time, build_prefix_subgraphs, process_single_subgraph, \
    map_event_labels_to_activities, process_all_subgraphs, build_primary_object_pe_structure2, filter_pe_by_time2
from newTry.pipeline_OTC import process_object_id, build_and_save_split2
from newTry.preprocess import load_jsonocel_to_dfs, encode_object_table, add_absolute_time_features, encode_event_table, \
    process_omap, transform_objects_to_features, build_object2type_mapping, add_absolute_time_features_v2
from newTry.pipeline_OTC import build_activity2idx, split_primary_objects_train_val_test, filter_dicts_by_keys, \
    build_and_save_split




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
def process_vmap_start_timestamp(df):
    # 删除原omap列
    df = df.drop(columns=['vmap_start_timestamp'])
    return df


def add_event_duration_feature(df_events):
    """
    计算活动处理时长，并将其作为新列加入
    """
    # 1. 确保时间戳是 datetime 格式
    # errors='coerce' 可以把无法解析的格式转为 NaT，防止程序崩溃
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], errors='coerce')
    df_events['vmap_start_timestamp'] = pd.to_datetime(df_events['vmap_start_timestamp'], errors='coerce')

    # 2. 计算处理时长 (Processing Time)
    # 单位：秒 (Seconds)
    df_events['proc_duration_sec'] = (df_events['timestamp'] - df_events['vmap_start_timestamp']).dt.total_seconds()

    # 3. 处理可能的负数或空值 (比如数据录入错误导致开始晚于结束)
    # 将负数时长重置为 0
    df_events.loc[df_events['proc_duration_sec'] < 0, 'proc_duration_sec'] = 0
    # 空值填充为 0（或者中位数，视业务而定）
    df_events['proc_duration_sec'] = df_events['proc_duration_sec'].fillna(0)

    print(f"✅ 特征添加成功：'proc_duration_sec' (处理时长/秒)")
    print(f"📊 平均处理时长: {df_events['proc_duration_sec'].mean():.2f} 秒")

    return df_events



if __name__ == "__main__":
    list_eventlog = [
        'intermediate'
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
        keep_types = []
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
        df_events_ab, event_time_dict = add_absolute_time_features_v2(df_events)
        #omap这列存了这个事件关联的对象，其实和关系表重复了，所以删掉
        df_events_updated = process_omap(df_events_ab)
        # pd.set_option('display.max_columns', None)
        # print(df_events_updated)
        # exit()

        #类别属性和数值属性
        categorical_evt = [
            "activity",
        ]
        numeric_evt = [
            "weekday_sin",
            "weekday_cos",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "second_sin",
            "second_cos",
            "omap_length"
        ]
        #得到事件特征矩阵
        X_evt, evt2idx, feature_names_evt = transform_objects_to_features(df_events_updated, categorical_evt,
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
        event_dim = X_evt.shape[1] + 2
        object_dim = X_obj.shape[1]
        dim = (event_dim, object_dim)
        dim_path = os.path.join(
            base_save_path,
            f"dim.pt"
        )
        torch.save(dim, dim_path)


#标注一下，事件表29维，对象表3