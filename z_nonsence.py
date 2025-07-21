# 02_analyze_t_values.py

import os
import pandas as pd
import seaborn as sns # 导入我们用于绘图的新工具：seaborn
import matplotlib.pyplot as plt # 导入 matplotlib，seaborn依赖于它
import numpy as np # 导入 numpy 用于处理无穷大值
from scipy.cluster import hierarchy # 导入我们用于层次聚类的新工具
from sklearn.preprocessing import StandardScaler # 导入用于数据标准化的新工具

# --- 1. 合并所有t值序列 (您已有的正确部分) ---

# 请确保这个路径是您存放t值CSV文件的正确位置
folder_directory = '/Users/zhanghongyi/Desktop/25_Summer/Quant_Finance/ricequant_strategies/Factor_test/d_短期 (1-3年)_tvalue_collection'

dataframes_to_merge = []
# ... (您之前用于合并文件的循环代码) ...
if os.path.isdir(folder_directory):
    all_filenames = os.listdir(folder_directory)
    for filename in all_filenames:
        if filename.endswith('.csv'):
            path = os.path.join(folder_directory, filename)
            t_value_df = pd.read_csv(path, index_col='date', parse_dates=True)
            factor_name = os.path.splitext(filename)[0]
            renamed_df = t_value_df.rename(columns={'t_value': factor_name})
            dataframes_to_merge.append(renamed_df)

if dataframes_to_merge:
    final_matrix = pd.concat(dataframes_to_merge, axis=1)
    
    # --- 【已修正】更稳健的数据清洗与准备流程 ---
    print(f"\n合并后初始形状: {final_matrix.shape}")
    
    # a) 替换无穷大值为NaN
    final_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # b) 丢弃完全为空的列 (如果某个因子t值全部计算失败)
    final_matrix.dropna(axis=1, how='all', inplace=True)
    print(f"丢弃空列后形状: {final_matrix.shape}")

    # c) 丢弃方差为0的列 (如果某个因子t值始终不变，会导致标准化时除以0)
    # 只有当列数大于0时才进行此操作
    if final_matrix.shape[1] > 0:
        # 使用一个很小的阈值以应对浮点数精度问题
        final_matrix = final_matrix.loc[:, final_matrix.std() > 1e-8]
    print(f"丢弃零方差列后形状: {final_matrix.shape}")

    # d) 用前一个有效值填充剩余的NaN值
    final_matrix.fillna(method='ffill', inplace=True)
    
    # e) 丢弃填充后仍然有NaN的行 (通常是开头几行)
    final_matrix.dropna(how='any', inplace=True)
    print(f"最终清洗后形状: {final_matrix.shape}")
    
    print("\n最终合并的t值矩阵信息:")
    final_matrix.info()
else:
    print(f"错误：在文件夹 '{folder_directory}' 中没有找到任何CSV文件。")
    final_matrix = pd.DataFrame() # 创建一个空的DataFrame以避免后续代码报错

if not final_matrix.empty:
    
    # --- 2. 可视化分析第一部分：相关矩阵热力图 (您已有的正确部分) ---

    # a) 设置中文字体
    # (这里使用 'Arial Unicode MS' 作为备选，因为它在Mac上更常见)
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 未找到 'Arial Unicode MS' 字体，尝试使用 'SimHei'。")
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("警告: 未找到 'SimHei' 字体，中文可能显示为乱码。")

    # b) 计算并绘制热力图
    print("\n正在计算相关系数矩阵并绘制热力图...")
    corr_matrix = final_matrix.corr()
    plt.figure(figsize=(25, 20)) 
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
    plt.title('因子t值相关矩阵热力图', fontsize=20)
    plt.xticks(fontsize=8, rotation=90) # 将x轴标签旋转90度，防止重叠
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

    # --- 3. 【新增】可视化分析第二部分：层次聚类分析 ---

    print("\n正在进行层次聚类分析...")

    # a) 数据准备：转置与标准化
    # 聚类分析的对象是“因子”，所以我们需要将DataFrame转置，让每一行代表一个因子。
    data_for_clustering = final_matrix.T
    
    # 标准化数据，消除不同因子t值波动大小（量纲）的影响，让算法更关注“变化模式”本身。
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_clustering)

    # b) 执行层次聚类
    # 'ward' 是一种常用的连接算法(linkage method)，它的目标是最小化各个簇内的方差。
    # 它会返回一个描述了因子之间如何一步步合并成簇的“关系树”结构。
    linked = hierarchy.linkage(data_scaled, method='ward')

    # c) 绘制树状图 (Dendrogram)
    print("正在绘制聚类树状图...")
    plt.figure(figsize=(20, 12))

    # 调用 dendrogram 函数来可视化 'linked' 结构
    hierarchy.dendrogram(
        linked,
        orientation='top', # 'top' 表示树从上往下生长，也可以设为 'left' 让它从左往右
        labels=data_for_clustering.index, # 使用因子名作为树叶的标签
        distance_sort='descending', # 按距离降序排列，让图结构更清晰
        show_leaf_counts=True, # 显示每个叶节点代表的因子数量（通常是1）
        leaf_rotation=90, # 将底部的因子标签旋转90度，防止重叠
        leaf_font_size=10 # 设置标签字体大小
    )

    plt.title('因子层次聚类树状图 (Dendrogram)', fontsize=16)
    plt.xlabel('因子名称', fontsize=12)
    plt.ylabel('距离 (Distance) - 衡量不相似度', fontsize=12)
    plt.tight_layout()
    plt.show()

    # d) (可选) 从聚类结果中提取分组标签
    # 我们可以通过设定一个“切割高度”或“目标簇数量”来获得每个因子的具体分组。
    # 例如，我们想把所有因子分成5个大类：
    num_clusters = 5
    clusters = hierarchy.fcluster(linked, num_clusters, criterion='maxclust')
    
    # 将结果整理成一个DataFrame，方便查看
    factor_clusters = pd.DataFrame({'因子': data_for_clustering.index, '类别': clusters})
    print("\n因子分类结果 (按类别排序):")
    print(factor_clusters.sort_values('类别'))
