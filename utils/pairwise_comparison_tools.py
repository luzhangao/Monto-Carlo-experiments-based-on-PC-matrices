# coding:utf8

"""
@author: Zhangao Lu
@contact:
@time: 2021/2/27
@description:
1. Some special tools for pairwise comparisons.
"""

import numpy as np
import networkx as nx
from itertools import combinations


def vector_to_pc_matrix(v):
    """
    vector -> PC matrix
    vector = [a_1, a_2, ... a_n]
    matrix = [[1            a_1/a_2         a_1/a_3          ...]
              [a_2/a_1      1               a_2/a_3          ...]
              ...
               a_n/a_1      a_n/a_2         a_n/a_3          1  ]]
    Convect vector to PC matrix
    :param v: list
           vector
    :return: numpy.array
             The PC matrix
    """
    order = len(v)
    m = np.eye(order)  # np.eye() can return a 2-D array with ones on the diagonal and zeros elsewhere.
    # combinations(range(0, 2), 2) == > (0, 1), (0, 2), (1, 2).
    t = combinations(range(0, order), 2)
    for elm in t:
        m[elm[0], elm[1]] = v[elm[0]] / v[elm[1]] if v[elm[1]] != 0 else 0  # m_{ij} = v_i / v_j
        m[elm[1], elm[0]] = v[elm[1]] / v[elm[0]] if v[elm[0]] != 0 else 0  # m_{ji} = v_j / v_i
    return m


def pc_matrix_to_vector(m):
    """
    PC matrix -> vector
    matrix = [[1            a_1/a_2         a_1/a_3          ...]
              [a_2/a_1      1               a_2/a_3          ...]
              ...
               a_n/a_1      a_n/a_2         a_n/a_3          1  ]]
    vector = [a_1, a_2, ... a_n]
    :param m: numpy.array
              PC matrix
    :return: list
             vector
    """
    order = m.shape[0]  # m.shape = [0, 0]
    tmp = dict()
    ind = m.argmax()  # Returns the indices of the maximum values along an axis.
    ind_x = ind // order  # floor division (rounds down to nearest whole number)
    ind_y = ind % order
    if m[ind_x, ind_y] == 0:  # It must be ensure that all elements of the PC matrix is greater than zero.
        raise UserWarning("The maximum element of matrix with missing values is 0!")

    index_collection = np.argwhere(m > 0)  # find the index of elements which values are not equal to 0 and -1
    # Use graphic tools to rebuild the vector.
    g = nx.Graph()  # Create an empty graph with no nodes and no edges.
    for elm in index_collection:
        x = elm[0]
        y = elm[1]
        if x != y:
            g.add_edge(x, y)  # Add edges.
    first_node = list(g.nodes())[0]
    gpath = list(nx.dfs_tree(g, first_node).nodes())  # Find the path of nodes, e.g. [0, 3, 1, 4] or [4, 0, 3, 1]
    for i in range(len(gpath)):
        if i == 0:
            tmp[gpath[i]] = 1
        else:
            prev = gpath[i-1]
            now = gpath[i]
            if prev < now:  # prev = 0 and now =3
                tmp[now] = tmp[prev] / m[prev, now]  # v_3 = v_0 / m[0, 3]
            else:  # prev = 3 and now = 0
                tmp[now] = tmp[prev] * m[now, prev]  # v_0 = v_3 * m[0, 3]

    v = list()  # vector
    for elm in range(order):
        v.append(tmp.get(elm, -1))
    return v


def normalize_vector(v):
    """
    normalize vector, let the sum of vector values be 1
    sum([v_0, v_1, ... v_n]) = 1
    :param v: list
           vector, e.g. [1, 0.5, 0.2]
    :return: list
             normalized vector
    """
    new_v = v / np.sum(v)
    return new_v


def compute_kii(m):
    """
    Compute the Koczkodaj inconsistency index of any matrix.
    kii = max(min{|1-a_{ij}/(a_{ij}a_{jk})|, |1-(a_{ij}a_{jk})/a_{ij}|}), i < k < j
    :param m: np.array
           Square PC matrix
    :return: Kii float
    """
    order = m.shape[0]
    # combinations(range(0, 2), 2) == > (0, 1), (0, 2), (1, 2).
    triad_list = combinations(range(0, order), 3)
    res = list()
    for elm in triad_list:
        # i < j < k
        i = elm[0]
        j = elm[1]
        k = elm[2]
        a_ik = m[i, k]
        a_ij = m[i, j]
        a_jk = m[j, k]
        if a_jk > 0 and a_ik > 0 and a_ij > 0:  # Make sure all elements are greater than zero.
            r = abs(1 - a_ik / (a_ij * a_jk)), abs(1 - (a_ij * a_jk) / a_ik)
            res.append(min(r))
    kii = max(res)
    if res:
        if kii > 1e-10:  # It should be zero, but it shows 1.xxx e-16 sometime because of the error of precision.
            kii = max(res)
        else:
            kii = 0
    else:
        kii = 1
    return kii


if __name__ == '__main__':
    print(vector_to_pc_matrix([2, 3, 5]))
    print(compute_kii(vector_to_pc_matrix([2, 3, 5])))
    print(pc_matrix_to_vector(vector_to_pc_matrix([2, 3, 5])))
    # tmp = pc_matrix_to_vector(np.array([1, 2, 5, 1/2, 1, 3, 1/5, 1/3, 1]).reshape(3, 3))
    # print(tmp)
    # print(normalize_vector(tmp))
