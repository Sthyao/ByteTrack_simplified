import numpy as np
from scipy.optimize import linear_sum_assignment

def lapjv(cost_matrix, extend_cost=True, cost_limit=np.inf):
    """
    使用 SciPy 的 linear_sum_assignment 实现类似于 lap.lapjv 的功能。
    
    参数:
    - cost_matrix (2D array): 成本矩阵。
    - extend_cost (bool): 是否扩展成本矩阵以允许未匹配的分配。
    - cost_limit (float): 每个分配的成本上限，超过该值的分配将被视为未匹配。
    
    返回:
    - total_cost (float): 符合阈值的总匹配成本。
    - row_ind (1D array): 匹配的行索引。
    - col_ind (1D array): 匹配的列索引。
    """
    num_rows, num_cols = cost_matrix.shape
    orig_num_rows, orig_num_cols = num_rows, num_cols

    if extend_cost:
        # 扩展成本矩阵到方阵
        size = max(num_rows, num_cols)
        if num_rows != num_cols:
            extended_cost_matrix = np.full((size, size), fill_value=cost_limit, dtype=cost_matrix.dtype)
            extended_cost_matrix[:num_rows, :num_cols] = cost_matrix
        else:
            extended_cost_matrix = cost_matrix.copy()
    else:
        extended_cost_matrix = cost_matrix.copy()

    # 使用匈牙利算法进行匹配
    row_ind, col_ind = linear_sum_assignment(extended_cost_matrix)

    # 如果扩展了成本矩阵，过滤掉虚拟匹配
    if extend_cost:
        valid_indices = (row_ind < orig_num_rows) & (col_ind < orig_num_cols)
        row_ind = row_ind[valid_indices]
        col_ind = col_ind[valid_indices]

    # 应用成本阈值
    mask = cost_matrix[row_ind, col_ind] <= cost_limit
    filtered_row_ind = row_ind[mask]
    filtered_col_ind = col_ind[mask]
    total_cost = cost_matrix[filtered_row_ind, filtered_col_ind].sum()

    # 创建完整的匹配数组，未匹配的用 -1 表示
    x = -1 * np.ones(orig_num_rows, dtype=int)
    y = -1 * np.ones(orig_num_cols, dtype=int)
    x[filtered_row_ind] = filtered_col_ind
    y[filtered_col_ind] = filtered_row_ind

    return total_cost, x, y

def bbox_overlaps_python(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) 格式的边界框数组 [x1, y1, x2, y2]
    query_boxes: (K, 4) 格式的查询框数组 [x1, y1, x2, y2]
    
    Returns
    -------
    overlaps: (N, K) IoU矩阵
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    
    for k in range(K):
        query_box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        for n in range(N):
            box_area = (
                (boxes[n, 2] - boxes[n, 0]) *
                (boxes[n, 3] - boxes[n, 1])
            )
            
            # 计算交集区域
            ix1 = max(boxes[n, 0], query_boxes[k, 0])
            iy1 = max(boxes[n, 1], query_boxes[k, 1])
            ix2 = min(boxes[n, 2], query_boxes[k, 2])
            iy2 = min(boxes[n, 3], query_boxes[k, 3])
            
            # 如果没有重叠区域
            if ix2 < ix1 or iy2 < iy1:
                continue
                
            # 计算交集面积
            intersection = (ix2 - ix1) * (iy2 - iy1)
            
            # 计算IoU
            union = box_area + query_box_area - intersection
            overlaps[n, k] = intersection / union
            
    return overlaps
