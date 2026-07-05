import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import pandas as pd
from datetime import datetime




def load_jsonocel_to_dfs(jsonocel_path):
    """
    读取标准JSONOCEL文件（适配你的实际字段：ocel:activity/ocel:omap/ocel:vmap）
    拆分为事件表、对象表、关系表
    """
    with open(jsonocel_path, "r", encoding="utf-8") as f:
        ocel = json.load(f)

    # -------- 1. 构建事件表（核心：提取ocel:activity） --------
    event_rows = []
    # 读取标准OCEL事件字段：ocel:events
    events_dict = ocel.get("ocel:events", {})

    if not events_dict:
        print("警告：ocel:events字段为空！")
    else:
        for eid, e in events_dict.items():
            row = {"event_id": eid}  # 事件ID（如1.0）

            # 提取核心字段：活动类型（关键！解决activity列空值问题）
            row["activity"] = e.get("ocel:activity", None)

            # 提取时间戳（格式化为字符串，方便后续处理）
            row["timestamp"] = e.get("ocel:timestamp", None)

            # 提取ocel:omap（对象映射列表，可选保留）
            row["omap"] = e.get("ocel:omap", [])

            # 提取ocel:vmap（数值属性字典，展开为单独列）
            vmap = e.get("ocel:vmap", {})
            for k, v in vmap.items():
                row[f"vmap_{k}"] = v  # 如vmap_weight、vmap_price

            event_rows.append(row)

    df_events = pd.DataFrame(event_rows)
    print(f"事件表加载完成：{len(df_events)} 行，activity列非空值：{df_events['activity'].notna().sum()}")

    # -------- 2. 构建对象表（ocel:objects） --------
    object_rows = []
    objects_dict = ocel.get("ocel:objects", {})

    if not objects_dict:
        print("警告：ocel:objects字段为空！")
    else:
        for oid, o in objects_dict.items():
            row = {"object_id": oid}
            # 提取对象类型（OCEL标准字段）
            row["object_type"] = o.get("ocel:type", None)
            # 合并对象属性
            row.update(o.get("ocel:attributes", {}))
            object_rows.append(row)

    df_objects = pd.DataFrame(object_rows)
    print(f"对象表加载完成：{len(df_objects)} 行")

    # -------- 3. 构建事件-对象关系表（从ocel:omap提取） --------
    rel_rows = []
    if events_dict:
        for eid, e in events_dict.items():
            # 你的数据中，事件关联的对象在ocel:omap字段（而非ocel:objects）
            for oid in e.get("ocel:omap", []):
                rel_rows.append({
                    "event_id": eid,
                    "object_id": oid
                })

    df_relations = pd.DataFrame(rel_rows)
    print(f"关系表加载完成：{len(df_relations)} 行")

    return df_events, df_objects, df_relations


def encode_object_table(
    df_objects: pd.DataFrame,
    categorical_cols: list,
    numeric_cols: list,
):


    df = df_objects.copy()

    # 1. 建立 object 索引
    object_ids = df["object_id"].tolist()
    obj2idx = {oid: i for i, oid in enumerate(object_ids)}

    # 2. 特征子表
    df_features = df.drop(columns=["object_id"])

    # --- 安全检查（强烈建议保留） ---
    unknown_cols = set(df_features.columns) - set(categorical_cols) - set(numeric_cols)
    if unknown_cols:
        raise ValueError(
            f"以下列未声明为 categorical 或 numeric，请显式指定语义: {unknown_cols}"
        )

    # ===============================
    # 3. 数值特征
    # ===============================
    if numeric_cols:
        df_num = df_features[numeric_cols].copy()
        df_num = df_num.fillna(0.0)

        scaler = StandardScaler()
        X_num = scaler.fit_transform(df_num)

        num_feature_names = numeric_cols
    else:
        X_num = np.empty((len(df), 0))
        num_feature_names = []

    # ===============================
    # 4. 类别特征（独热）
    # ===============================
    if categorical_cols:
        df_cat = df_features[categorical_cols].copy()

        # 关键点：即使是 int，也强制当 category
        for c in categorical_cols:
            df_cat[c] = df_cat[c].astype("category")

        # NaN -> 全 0（不单独开 NaN 列）
        X_cat_df = pd.get_dummies(
            df_cat,
            columns=categorical_cols,
            dummy_na=False
        )

        X_cat = X_cat_df.values.astype(float)
        cat_feature_names = X_cat_df.columns.tolist()
    else:
        X_cat = np.empty((len(df), 0))
        cat_feature_names = []

    # ===============================
    # 5. 合并
    # ===============================
    X_obj = np.hstack([X_num, X_cat])
    feature_names = num_feature_names + cat_feature_names

    return X_obj, obj2idx, feature_names


def transform_objects_to_features(df, cat_cols, num_cols, id_col='object_id', type_col='activity'):
    """
    将对象表转换为特征矩阵，并提取对象类型的映射信息。

    新增参数:
    - type_col: 指定哪一列是对象类型 (例如 'object_type')，用于提取映射关系。

    返回:
    - feature_matrix: numpy数组
    - id_map: ID到索引的映射
    - feature_names: 特征列名列表
    - type_info: (新增) 包含类型映射和列范围的字典
    """

    # 1. 准备工作
    if id_col not in df.columns:
        raise ValueError(f"列 {id_col} 不在 DataFrame 中")

    # 确保 type_col 在 cat_cols 里，否则无法生成独热编码
    if type_col not in cat_cols:
        print(f"Warning: {type_col} 不在 cat_cols 中，已自动添加以便生成映射。")
        cat_cols.append(type_col)

    df_working = df.copy().set_index(id_col, drop=False)
    id_map = {oid: i for i, oid in enumerate(df_working.index)}
    processed_parts = []

    # 2. 处理数值列
    if num_cols:
        df_num = df_working[num_cols].fillna(0)
        scaler = MinMaxScaler()
        num_matrix = scaler.fit_transform(df_num)
        df_num_processed = pd.DataFrame(num_matrix, columns=num_cols, index=df_working.index)
        processed_parts.append(df_num_processed)

    # 3. 处理类别列 (独热编码)
    if cat_cols:
        df_cat = df_working[cat_cols]
        # dummy_na=False: NaN 变为全 0
        df_cat_processed = pd.get_dummies(df_cat, columns=cat_cols, dummy_na=False, dtype=int)
        processed_parts.append(df_cat_processed)

    # 4. 合并
    if not processed_parts:
        raise ValueError("未提供任何特征列")

    df_final = pd.concat(processed_parts, axis=1)
    feature_names = df_final.columns.tolist()
    feature_matrix = df_final.values

    # =======================================================
    # 【新增功能】提取并打印对象类型的映射与列范围
    # =======================================================
    print("-" * 30)
    print(f"正在提取 '{type_col}' 的特征映射信息...")

    # 1. 找到所有由 type_col 生成的独热列
    # pandas get_dummies 默认格式: "列名_值" (例如 object_type_Item)
    prefix = f"{type_col}_"
    type_columns = [col for col in feature_names if col.startswith(prefix)]

    if not type_columns:
        print(f"警告: 未找到前缀为 {prefix} 的列，请检查 type_col 参数是否正确。")
        type_info = {}
    else:
        # 2. 获取列索引范围
        # 假设这些列是连续的（通常是连续的），我们要找到它们在 feature_matrix 里的下标
        indices = [feature_names.index(col) for col in type_columns]
        start_idx = min(indices)
        end_idx = max(indices)  # 包含

        # 3. 构建映射字典 { 'Item': 索引, 'Order': 索引 }
        # col[len(prefix):] 用于去掉前缀 "object_type_" 得到 "Item"
        type_map = {col[len(prefix):]: idx for col, idx in zip(type_columns, indices)}

        # 4. 打印信息
        print(f"对象类型特征范围: 第 {start_idx} 列 到 第 {end_idx} 列")
        print(f"类型 -> 独热编码索引映射:")
        for name, idx in type_map.items():
            print(f"  - 类型 '{name}': 对应特征矩阵第 [{idx}] 列")

        # 5. 打包返回信息
        type_info = {
            'start_col': start_idx,
            'end_col': end_idx,
            'mapping': type_map,  # {'Item': 10, 'Package': 11}
            'reverse_mapping': {v: k for k, v in type_map.items()}  # {10: 'Item'} 方便反查
        }
    print("-" * 30)

    return feature_matrix, id_map, feature_names, type_info


def add_absolute_time_features(df_events):
    """
    给事件表添加绝对时间特征 (Sin/Cos 编码)
    去掉原始时间戳，但保留字典映射
    """
    df = df_events.copy()

    # 确保转换为 datetime 对象
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 生成 ID -> Timestamp 字典
    event_time_dict = dict(zip(df["event_id"], df["timestamp"]))

    # 提取时间组件
    # df["year"] = df["timestamp"].dt.year     # 暂时注释
    # df["month"] = df["timestamp"].dt.month   # 暂时注释

    weekday = df["timestamp"].dt.weekday  # 0-6
    hour = df["timestamp"].dt.hour
    minute = df["timestamp"].dt.minute
    second = df["timestamp"].dt.second

    # Sin / Cos 编码
    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    df["minute_cos"] = np.cos(2 * np.pi * minute / 60)

    df["second_sin"] = np.sin(2 * np.pi * second / 60)
    df["second_cos"] = np.cos(2 * np.pi * second / 60)

    # 移除原始时间戳和临时列
    df = df.drop(columns=["timestamp"])

    return df, event_time_dict


def add_absolute_time_features_v2(df_events):
    """
    统一抹除毫秒，解决格式不一致导致的报错
    """
    df = df_events.copy()

    # --- 1. 暴力统一字符串格式 ---
    # 逻辑：如果带小数点，只取小数点前面的部分；如果不带，保持原样
    # 这种方式比正则快，且能处理 2016-01-20T19:19:06 和 2016-01-01T13:34:53.911000
    df["timestamp"] = df["timestamp"].astype(str).str.split('.').str[0]

    if "vmap_start_timestamp" in df.columns:
        df["vmap_start_timestamp"] = df["vmap_start_timestamp"].astype(str).str.split('.').str[0]

    # --- 2. 统一解析 ---
    # 现在的格式全部统一成了 %Y-%m-%dT%H:%M:%S
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 建立 ID -> Timestamp 字典
    event_time_dict = dict(zip(df["event_id"], df["timestamp"]))

    # 计算时长 (可选)
    # if "vmap_start_timestamp" in df.columns:
    #     df["vmap_start_timestamp"] = pd.to_datetime(df["vmap_start_timestamp"])
    #     df["proc_duration_sec"] = (df["timestamp"] - df["vmap_start_timestamp"]).dt.total_seconds()
    #     df["proc_duration_sec"] = df["proc_duration_sec"].fillna(0).clip(lower=0)

    # --- 3. 提取循环编码特征 ---
    weekday = df["timestamp"].dt.weekday
    hour = df["timestamp"].dt.hour
    minute = df["timestamp"].dt.minute
    second = df["timestamp"].dt.second

    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    df["minute_cos"] = np.cos(2 * np.pi * minute / 60)

    # 既然抹掉了毫秒，秒级特征依然保留
    df["second_sin"] = np.sin(2 * np.pi * second / 60)
    df["second_cos"] = np.cos(2 * np.pi * second / 60)

    # --- 4. 移除原始列 ---
    df = df.drop(columns=["timestamp", "vmap_start_timestamp"], errors='ignore')

    print(f"✅ 格式已强行统一，毫秒已剔除。")
    return df, event_time_dict


def encode_event_table(df_events):
    """
    处理事件表，生成特征矩阵
    """
    df = df_events.copy()

    event_ids = df["event_id"].tolist()
    evt2idx = {eid: i for i, eid in enumerate(event_ids)}

    # 移除ID
    df_features = df.drop(columns=["event_id"])

    # 分离列
    numeric_cols = df_features.select_dtypes(include=["int64", "float64", "float32"]).columns.tolist()
    categorical_cols = df_features.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    # 1. 数值处理
    if numeric_cols:
        # 填充缺失值
        df_features[numeric_cols] = df_features[numeric_cols].fillna(0)
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df_features[numeric_cols])
    else:
        X_num = np.empty((len(df), 0))

    # 2. 类别处理
    if categorical_cols:
        # 使用 get_dummies 处理独热
        X_cat_df = pd.get_dummies(df_features[categorical_cols], columns=categorical_cols, dummy_na=False)
        X_cat = X_cat_df.values.astype(float)
        cat_feature_names = X_cat_df.columns.tolist()
    else:
        X_cat = np.empty((len(df), 0))
        cat_feature_names = []

    # 3. 合并
    X_evt = np.hstack([X_num, X_cat])

    feature_names = numeric_cols + cat_feature_names

    return X_evt, evt2idx, feature_names


def check_list_elements_in_df(df, check_columns=None):
    """
    检查 DataFrame 中包含列表（list）类型的元素，并输出详细信息

    参数：
        df: 待检查的 DataFrame（如 df_events_updated）
        check_columns: 可选，指定要检查的列列表；None 则检查所有列

    返回：
        list_cols_info: 字典，包含有列表元素的列名、行数、行索引和示例
    """
    # 确定要检查的列
    if check_columns is None:
        check_columns = df.columns.tolist()
    else:
        # 校验指定的列是否存在于DataFrame中
        check_columns = [col for col in check_columns if col in df.columns]
        if not check_columns:
            print("⚠️ 指定的列均不存在于DataFrame中！")
            return {}

    list_cols_info = {}

    print(f"========== 开始检查列表类型元素 ==========\n")
    print(f"待检查列数：{len(check_columns)} | DataFrame总行数：{len(df)}\n")

    for col in check_columns:
        # 1. 检查该列中每个元素是否是列表类型
        is_list = df[col].apply(lambda x: isinstance(x, list))
        list_count = is_list.sum()

        if list_count > 0:
            # 获取包含列表的行索引
            list_row_indices = df[is_list].index.tolist()
            # 获取前5个列表元素示例
            list_samples = df[col][is_list].head(5).tolist()

            # 存储信息到字典
            list_cols_info[col] = {
                "list_count": list_count,  # 列表类型元素数量
                "list_row_indices": list_row_indices,  # 包含列表的行索引
                "list_samples": list_samples  # 列表元素示例
            }

            # 打印详细信息
            print(f"📌 列名：{col}")
            print(f"   - 列表类型元素数量：{list_count} 个（占该列 {list_count / len(df) * 100:.2f}%）")
            print(f"   - 包含列表的行索引（前5个）：{list_row_indices[:5]}")
            print(f"   - 列表元素示例（前5个）：{list_samples}")
            print("-" * 80)

    # 汇总结果
    if list_cols_info:
        print(f"\n✅ 检查完成！共发现 {len(list_cols_info)} 列包含列表类型元素：")
        for col, info in list_cols_info.items():
            print(f"   - {col}：{info['list_count']} 个列表元素")
    else:
        print("\n✅ 检查完成！未发现任何列表类型元素")

    return list_cols_info

def process_omap(df):
    # 新增列存储omap列表长度
    df['omap_length'] = df['omap'].apply(lambda x: len(x))
    # 删除原omap列
    df = df.drop(columns=['omap'])
    return df


def build_object2type_mapping(df: pd.DataFrame,
                              object_id_col: str = 'object_id',
                              object_type_col: str = 'object_type') -> dict:
    """
    从DataFrame中构建object_id到object_type的映射字典

    参数:
        df: 包含object_id和object_type列的DataFrame
        object_id_col: DataFrame中表示对象ID的列名（默认：'object_id'）
        object_type_col: DataFrame中表示对象类型的列名（默认：'object_type'）

    返回:
        dict: 键为object_id，值为对应的object_type的映射字典

    异常处理:
        1. 若输入不是DataFrame，抛出TypeError
        2. 若指定的列不存在，抛出KeyError
        3. 空值会被过滤，重复的object_id保留最后一次出现的type
    """
    # 输入类型校验
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"输入的df必须是pandas.DataFrame类型，当前类型：{type(df)}")

    # 列存在性校验
    if object_id_col not in df.columns:
        raise KeyError(f"DataFrame中不存在指定的object_id列：{object_id_col}")
    if object_type_col not in df.columns:
        raise KeyError(f"DataFrame中不存在指定的object_type列：{object_type_col}")

    # 复制数据避免修改原DataFrame，过滤空值行
    df_clean = df[[object_id_col, object_type_col]].copy()
    df_clean = df_clean.dropna(subset=[object_id_col, object_type_col])  # 过滤任意一列空值的行

    # 处理重复的object_id：保留最后一次出现的type（也可改为保留第一次，只需将keep='last'改为keep='first'）
    df_clean = df_clean.drop_duplicates(subset=[object_id_col], keep='last')

    # 转换为字典（object_id为键，object_type为值）
    object2type = df_clean.set_index(object_id_col)[object_type_col].to_dict()

    return object2type