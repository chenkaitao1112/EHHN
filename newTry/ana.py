import os
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from typing import Dict, Tuple, Set
from newTry.construct_PE import build_relation_mappings, get_primary_object_ids, build_primary_object_pe_structure, \
    sort_primary_lifecycle_edges, filter_pe_by_time, build_prefix_subgraphs, process_single_subgraph, \
    map_event_labels_to_activities, process_all_subgraphs
from newTry.preprocess import load_jsonocel_to_dfs, encode_object_table, add_absolute_time_features, encode_event_table, \
    process_omap, transform_objects_to_features
import json
import pandas as pd
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional


def get_object_types(object_ids: List[Any], df: pd.DataFrame) -> Dict[Any, Optional[str]]:

    if not isinstance(object_ids, list):
        raise TypeError("object_ids必须是列表类型")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入的df必须是pandas.DataFrame类型")


    required_cols = ['object_id', 'object_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame缺少必要列: {', '.join(missing_cols)}")


    unique_object_ids = list(set(object_ids))


    df = df.copy()
    df['object_id'] = df['object_id'].astype(str)
    query_ids = [str(id_) for id_ in unique_object_ids]


    id_type_map = dict(zip(df['object_id'], df['object_type']))


    result = {}
    not_found_ids = []

    for original_id in object_ids:
        str_id = str(original_id)
        if str_id in id_type_map:
            result[original_id] = id_type_map[str_id]
        else:
            result[original_id] = None
            if original_id not in not_found_ids:
                not_found_ids.append(original_id)



    for obj_id, obj_type in result.items():
        status = "✅" if obj_type else "❌"
        print(f"   {status} {obj_id} -> {obj_type if obj_type else '未找到匹配类型'}")


    if not_found_ids:
        print(f"\n⚠️  以下Object ID未找到对应的object_type：")
        print(f"   {not_found_ids}")


    found_count = sum(1 for v in result.values() if v is not None)
    print(f"\n📊 统计：找到匹配类型 {found_count} 个，未找到 {len(not_found_ids)} 个")

    return result


def get_merged_omap_list_with_activity(df: pd.DataFrame):

    target_event_ids = ['1.0', '4.0', '116.0', '125.0', '287.0', '751.0', '957.0', '988.0', '1022.0']

    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入的df必须是pandas.DataFrame类型")
    required_cols = ['event_id', 'activity', 'omap']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame缺少必要列: {', '.join(missing_cols)}")


    df['event_id'] = df['event_id'].astype(str)


    mask = df['event_id'].isin(target_event_ids)
    target_rows = df[mask].copy()


    target_rows_display = target_rows[['activity', 'omap']].copy()


    print("=" * 120)
    print(f"筛选出 event_id 为 {target_event_ids} 的行（共 {len(target_rows)} 行）：")
    print("=" * 120)

    if len(target_rows) == 0:
        print("⚠️  未找到任何匹配的行")
        return []


    with pd.option_context(
            'display.max_columns', None,
            'display.width', 2000,
            'display.max_colwidth', None
    ):
        print("📋 筛选结果（仅显示activity和omap列）：")
        print(target_rows_display)


    merged_omap = []
    for omap_list in target_rows['omap']:

        if isinstance(omap_list, list):
            merged_omap.extend(omap_list)


    seen = set()
    unique_omap = []
    for item in merged_omap:
        if item not in seen:
            seen.add(item)
            unique_omap.append(item)


    print(f"\n📊 合并结果统计：")
    print(f"   原始合并元素数量：{len(merged_omap)}")
    print(f"   去重后元素数量：{len(unique_omap)}")


    missing_ids = [eid for eid in target_event_ids if eid not in target_rows['event_id'].values]
    if missing_ids:
        print(f"\n⚠️  以下event_id未在表格中找到匹配行：")
        print(f"   {missing_ids}")

    return unique_omap

def print_target_event_rows(df: pd.DataFrame):

    target_event_ids = ['1.0', '4.0', '116.0', '125.0', '287.0', '751.0', '957.0', '988.0', '1022.0']


    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入的df必须是pandas.DataFrame类型")
    if 'event_id' not in df.columns:
        raise ValueError("DataFrame缺少必要列: event_id")


    df['event_id'] = df['event_id'].astype(str)


    mask = df['event_id'].isin(target_event_ids)
    target_rows = df[mask].copy()
    target_rows['event_id'] = pd.Categorical(target_rows['event_id'], categories=target_event_ids, ordered=True)
    target_rows = target_rows.sort_values('event_id').reset_index(drop=True)


    print("=" * 120)
    print(f"筛选出 event_id 为 {target_event_ids} 的行（共 {len(target_rows)} 行）：")
    print("=" * 120)

    if len(target_rows) == 0:
        print("⚠️  未找到任何匹配的行")
        return


    with pd.option_context(
            'display.max_columns', None,
            'display.width', 2000,
            'display.max_colwidth', None
    ):
        print(target_rows)


    missing_ids = [eid for eid in target_event_ids if eid not in target_rows['event_id'].values]
    if missing_ids:
        print("\n⚠️  以下event_id未在表格中找到匹配行：")
        print(f"   {missing_ids}")


def get_merged_omap_list(df: pd.DataFrame) :

    target_event_ids = ['1.0', '4.0', '116.0', '125.0', '287.0', '751.0', '957.0', '988.0', '1022.0']

    # 数据校验
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入的df必须是pandas.DataFrame类型")
    required_cols = ['event_id', 'omap']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame缺少必要列: {', '.join(missing_cols)}")

    df['event_id'] = df['event_id'].astype(str)

    mask = df['event_id'].isin(target_event_ids)
    target_rows = df[mask].copy()

    print("=" * 120)
    print(f"筛选出 event_id 为 {target_event_ids} 的行（共 {len(target_rows)} 行）：")
    print("=" * 120)

    if len(target_rows) == 0:
        print("⚠️  未找到任何匹配的行")
        return []


    merged_omap = []
    for omap_list in target_rows['omap']:

        if isinstance(omap_list, list):
            merged_omap.extend(omap_list)


    seen = set()
    unique_omap = []
    for item in merged_omap:
        if item not in seen:
            seen.add(item)
            unique_omap.append(item)

    # 打印结果统计
    print(f"\n📊 合并结果统计：")
    print(f"   原始合并元素数量：{len(merged_omap)}")
    print(f"   去重后元素数量：{len(unique_omap)}")


    missing_ids = [eid for eid in target_event_ids if eid not in target_rows['event_id'].values]
    if missing_ids:
        print(f"\n⚠️  以下event_id未在表格中找到匹配行：")
        print(f"   {missing_ids}")

    return unique_omap


def summarize_ocel_counts(df_events, df_objects):

    print("=" * 60)
    print("📊 OCEL 数据分布统计报告")
    print("=" * 60)


    print("\n🔹 【对象类型分布 (Object Type Distribution)】")
    if 'object_type' in df_objects.columns:
        obj_counts = df_objects['object_type'].value_counts().reset_index()
        obj_counts.columns = ['Object Type', 'Count']

        total_objs = len(df_objects)
        obj_counts['Percentage'] = (obj_counts['Count'] / total_objs * 100).round(2).astype(str) + '%'
        print(obj_counts.to_string(index=False))
    else:
        print("❌ 错误：对象表中未找到 'object_type' 列")

    print("\n🔹 【活动类型分布 (Activity Distribution)】")
    if 'activity' in df_events.columns:
        act_counts = df_events['activity'].value_counts().reset_index()
        act_counts.columns = ['Activity', 'Count']

        total_events = len(df_events)
        act_counts['Percentage'] = (act_counts['Count'] / total_events * 100).round(2).astype(str) + '%'
        print(act_counts.to_string(index=False))
    else:
        print("❌ 错误：事件表中未找到 'activity' 列")

    print("\n" + "=" * 60)


    return obj_counts, act_counts


def explore_jsonocel_structure(jsonocel_path, timestamp_format="%Y-%m-%dT%H:%M:%S.%fZ"):

    try:
        with open(jsonocel_path, "r", encoding="utf-8") as f:
            ocel = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 错误：未找到文件 {jsonocel_path}")
    except json.JSONDecodeError:
        raise ValueError(f"❌ 错误：{jsonocel_path} 不是合法的JSON文件")



    event_rows = []
    events_dict = ocel.get("ocel:events", {})
    for eid, e in events_dict.items():
        row = {"event_id": str(eid)}
        row["activity"] = e.get("ocel:activity", "未知活动")

        timestamp_raw = e.get("ocel:timestamp")
        try:
            row["timestamp"] = datetime.strptime(timestamp_raw, timestamp_format) if timestamp_raw else None
        except (ValueError, TypeError):
            row["timestamp"] = timestamp_raw

        row["omap"] = e.get("ocel:omap", [])

        vmap = e.get("ocel:vmap", {})
        for k, v in vmap.items():
            clean_k = k.replace(" ", "_").replace("-", "_").replace(".", "_")
            row[f"vmap_{clean_k}"] = v
        event_rows.append(row)
    df_events = pd.DataFrame(event_rows).reset_index(drop=True)


    object_rows = []
    objects_dict = ocel.get("ocel:objects", {})
    for oid, o in objects_dict.items():
        row = {"object_id": str(oid)}
        row["object_type"] = o.get("ocel:type", "未知类型")

        attrs = o.get("ocel:attributes", {})
        for k, v in attrs.items():
            clean_k = k.replace(" ", "_").replace("-", "_").replace(".", "_")
            row[f"obj_attr_{clean_k}"] = v
        object_rows.append(row)
    df_objects = pd.DataFrame(object_rows).reset_index(drop=True)


    rel_rows = []
    valid_oids = set(df_objects["object_id"])
    for eid, e in events_dict.items():
        for oid in e.get("ocel:omap", []):
            oid_str = str(oid)
            if oid_str in valid_oids:
                rel_rows.append({"event_id": str(eid), "object_id": oid_str})
    df_relations = pd.DataFrame(rel_rows).drop_duplicates().reset_index(drop=True)


    print("\n【1. 事件表（df_events）信息】")
    print(f"   📏 数据维度：{df_events.shape[0]} 行 × {df_events.shape[1]} 列")
    print(f"   📋 表头（列名）：{list(df_events.columns)}")
    print(f"   🎯 核心字段示例（前3行）：")
    core_event_cols = ["event_id", "activity", "timestamp"]
    show_cols = [col for col in core_event_cols if col in df_events.columns]
    print(df_events[show_cols].head(3).to_string(index=False))
    print(f"   📈 activity字段分布：\n{df_events['activity'].value_counts().head(5)}")

    print("\n【2. 对象表（df_objects）信息】")
    print(f"   📏 数据维度：{df_objects.shape[0]} 行 × {df_objects.shape[1]} 列")
    print(f"   📋 表头（列名）：{list(df_objects.columns)}")
    print(f"   🎯 核心字段示例（前3行）：")
    core_obj_cols = ["object_id", "object_type"]
    show_obj_cols = [col for col in core_obj_cols if col in df_objects.columns]
    print(df_objects[show_obj_cols].head(3).to_string(index=False))
    print(f"   📈 object_type字段分布：\n{df_objects['object_type'].value_counts().head(5)}")

    print("\n【3. 关系表（df_relations）信息】")
    print(f"   📏 数据维度：{df_relations.shape[0]} 行 × {df_relations.shape[1]} 列")
    print(f"   📋 表头（列名）：{list(df_relations.columns)}")
    print(f"   🎯 前3行数据：")
    print(df_relations.head(3).to_string(index=False))
    print(f"   🔗 事件-对象关联密度：平均每个事件关联 {df_relations.shape[0] / max(df_events.shape[0], 1):.2f} 个对象")

    print("\n【4. 数据类型概览】")
    print("   事件表数据类型：")
    print(df_events.dtypes.head(10))
    print("   对象表数据类型：")
    print(df_objects.dtypes.head(10))

    print("\n" + "=" * 80)

    return df_events, df_objects, df_relations


def sample_object_ids(df_objects, num_samples=3):



    obj_types = df_objects['object_type'].unique()

    for o_type in obj_types:

        samples = df_objects[df_objects['object_type'] == o_type]['object_id'].head(num_samples).tolist()


        total_count = len(df_objects[df_objects['object_type'] == o_type])


        print(f"🔹 类型: {o_type:<20} (总计: {total_count:>5})")
        if samples:
            for i, s_id in enumerate(samples):
                print(f"   [{i + 1}] {s_id}")
        else:
            print("   (无数据)")
        print("-" * 40)

    print("=" * 60)


def check_super_nodes(df_relations, threshold=0.05):

    degree_counts = df_relations['object_id'].value_counts()
    total_events = df_relations['event_id'].nunique()

    # 转换为百分比
    degree_pct = (degree_counts / total_events)

    super_nodes = degree_pct[degree_pct > threshold]

    if not super_nodes.empty:
        print("⚠️ 警告：发现以下超级节点！")
        for oid, pct in super_nodes.items():
            print(f"   - 对象 [{oid}] 关联了 {pct:.2% Rose} 的事件")
    else:
        print("✅ 恭喜：未发现明显的超级节点，对象分布比较健康。")


def verify_material_uniqueness(df_relations, df_objects):

    mats = df_objects[df_objects['object_type'] == 'material']['object_id'].unique()
    pos = df_objects[df_objects['object_type'] == 'purchase_order']['object_id'].unique()
    event_to_objs = df_relations.groupby('event_id')['object_id'].apply(set)


    mat_to_pos = {}
    for objs in event_to_objs:
        current_mats = objs.intersection(mats)
        current_pos = objs.intersection(pos)
        for m in current_mats:
            if m not in mat_to_pos: mat_to_pos[m] = set()
            mat_to_pos[m].update(current_pos)

    # 5. 分析结果
    counts = pd.Series([len(v) for v in mat_to_pos.values()])
    print(f"物料-订单 关联性分析：")
    print(f"   - 仅属于 1 个订单的物料占比: {(counts == 1).mean():.2%}")
    print(f"   - 跨越多个订单的物料数量: {(counts > 1).sum()}")
    if (counts > 1).any():
        print(f"   - 跨度最大的物料关联了 {counts.max()} 个订单")


def deep_dive_loan_structure(df_relations, df_objects):

    app_ids = df_objects[df_objects['object_type'] == 'application']['object_id']
    app_rel = df_relations[df_relations['object_id'].isin(app_ids)]

    events_per_app = app_rel.groupby('object_id')['event_id'].nunique()

    print(f"📈 申请单活跃度分析：")
    print(f"   - 平均每个申请包含 {events_per_app.mean():.1f} 个事件")
    print(f"   - 最复杂的申请经历了 {events_per_app.max()} 个事件")
    print(f"   - 事件数最少的申请仅有 {events_per_app.min()} 个事件")



import ast


def transform_ocel_flat_to_tables(file_path):

    df = pd.read_csv(file_path)


    def parse_id_list(x):
        if pd.isna(x) or x == "" or x == "[]":
            return []
        if isinstance(x, str) and x.startswith('['):
            try:
                return ast.literal_eval(x)
            except:
                return [x.strip("[]'\" ")]
        return [str(x)]


    df['parsed_apps'] = df['application'].apply(parse_id_list)
    df['parsed_offers'] = df['offer'].apply(parse_id_list)


    rel_data = []
    for _, row in df.iterrows():
        event_id = row['event_id']

        associated_objects = list(set(row['parsed_apps'] + row['parsed_offers']))
        rel_data.append({'event_id': event_id, 'object_ids': associated_objects})

    relationship_df = pd.DataFrame(rel_data)


    objects_set = {}


    for apps in df['parsed_apps']:
        for app_id in apps:
            if app_id: objects_set[app_id] = 'application'


    for offers in df['parsed_offers']:
        for o_id in offers:
            if o_id: objects_set[o_id] = 'offer'

    object_df = pd.DataFrame(list(objects_set.items()), columns=['object_id', 'object_type'])


    cols_to_exclude = ['application', 'offer', 'parsed_apps', 'parsed_offers']
    event_df = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])

    return event_df, object_df, relationship_df


def delete_columns(df, columns_to_delete):

    existing_cols = [col for col in columns_to_delete if col in df.columns]


    new_df = df.drop(columns=existing_cols, axis=1)


    print(f"实际删除的列: {existing_cols}")
    if len(existing_cols) < len(columns_to_delete):
        missing = set(columns_to_delete) - set(existing_cols)
        print(f"预设列表中不存在的列: {list(missing)}")

    return new_df


def remove_column_prefix(df, prefix="event_", exclude=['event_id']):


    def rename_logic(col_name):

        if col_name in exclude or not isinstance(col_name, str):
            return col_name


        return col_name.removeprefix(prefix)

    return df.rename(columns=rename_logic)



if __name__ == "__main__":
    list_eventlog = [
        #'OTC',
        #"BPI2017"
        "p2p"
    ]

    current_file = Path(os.getcwd()).resolve()
    current_dir = current_file.parent
    parent_dir_str = str(current_dir)
    for eventlog in list_eventlog:
        eventlog_path = parent_dir_str + "/data/" + eventlog + "/source/" + eventlog + ".jsonocel"
        df_events, df_objects, df_relations = explore_jsonocel_structure(eventlog_path)



        obj_counts, act_counts = summarize_ocel_counts(df_events, df_objects)
        sample_object_ids(df_objects)
        check_super_nodes(df_relations)

        deep_dive_loan_structure(df_relations, df_objects)



