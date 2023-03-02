import numpy as np
from numpy import inf
from operator import itemgetter
import copy

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    # 初始化各个节点的关联节点及其对应的距离
    # 其中A~G分别对应数字1~7
    node_dist = [[1, [2, 6, 7], [12, 16, 14]],
                 [2, [1, 3, 6], [12, 10, 7]],
                 [3, [2, 4, 5, 6], [10, 3, 5, 6]],
                 [4, [3, 5], [3, 4]],
                 [5, [3, 4, 6, 7], [5, 4, 2, 8]],
                 [6, [1, 2, 3, 5, 7], [16, 7, 6, 2, 9]],
                 [7, [1, 5, 6], [14, 8, 9]]]

    # 以D为起点，A为终点，寻找全局最优路径
    # 其中S为已经找到最短路径的节点集合+对应的最短路径值；U为尚未找到最短路径的节点集合+当前的最短值
    S = [[4, 0]]  # S集合初始化
    U = [[1, inf],
         [2, inf],
         [3, 3],
         [5, 4],
         [6, inf],
         [7, inf]]  # U集合初始化

    path_opt = [[4, [4]]]  # 最优路径初始化
    path_temp = [[1, []],
                 [2, []],
                 [3, [4, 3]],
                 [5, [4, 5]],
                 [6, []],
                 [7, []]]  # 其他当前路径初始化

    for r in range(len(U)):
        print(r)

        U0 = [i[0] for i in U]
        U1 = [i[1] for i in U]
        min_index, min_number = min(enumerate(U1), key=itemgetter(1))
        dist_index = U0[min_index] - 1  # 选出节点  对应node_dist的索引l

        path_opt.append(path_temp[min_index])  # 将该节点对应的路径加入最优
        path_temp.remove(path_temp[min_index])  # 从临时路径中删除已确定最优路径的节点

        # 将节点添加到S集中 + 从U集中删除
        S.append([U0[min_index], min_number])
        U.remove(U[min_index])

        print(S)
        #     print(path_opt)

        U0_new = [i[0] for i in U]
        U1_new = [i[1] for i in U]

        # 更新U集中对应的距离
        x = node_dist[dist_index][1]
        # 判断是否更新
        for temp_index, w in enumerate(x):
            temp_distance = inf
            if w in [s[0] for s in S]:
                pass
            else:
                temp_distance = min_number + node_dist[dist_index][2][temp_index]  # 新距离
                # 先找旧的在新U中的index
                old_index = U0_new.index(w)
                # 查找“旧”距离
                old_distance = U1_new[old_index]
                # 判断是否更新路径值
                if temp_distance < old_distance:
                    U[old_index][1] = temp_distance
                    # 更新临时距离
                    d = copy.deepcopy(path_opt[-1][1])
                    d.append(path_temp[old_index][0])
                    path_temp[old_index][1] = d
                #                 print(path_temp)
                else:
                    pass
        print(U)
