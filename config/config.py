# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/2/28
@description:
1. Save some constants or constant list or constant dict.
"""
# A list to save colors.
color_list = ['lightcoral', 'indianred', 'brown', 'darkred', 'plum', 'violet', 'darkviolet', 'rebeccapurple',
                  'slateblue', 'blue', 'darkblue', 'midnightblue', ]

ft = {"fontsize": 16, "weight": "normal"}  # Change the font size to display better.
new_ft = {"fontsize": 11, "weight": "normal"}  # Change the font size to display better.

decimal_places = 4


# The dict is designed to convert metric names and is used in drawing graphs.
printed_metric = {"braycurtis": "Bray-Curtis Distance",
                  "canberra": "Canberra Distance",
                  "chebyshev": "Chebyshev Distance",
                  "cosine": "Cosine Similarity",
                  "euclidean": "Euclidean Distance",
                  "jensenshannon": "Jensen-Shannon Divergence",
                  "KLdivergence": "Kullback-Leibler Divergence"}

# The parameters used in experiments
mc_para_1 = {
    "metrics": ["braycurtis", "canberra", "chebyshev", "cosine", "euclidean", "jensenshannon", "KLdivergence"],
    "iterations": 100000,
    "is_show": True,
    "dpi": 200,
    "need_order": 4,
    # "need_order": 8,
}

mc_para_2 = {
    "metrics": ["braycurtis", "canberra", "jensenshannon"],
    "iterations": 100000,
    "is_show": True,
    "dpi": 400,
}

mc_para_3 = {
    "metrics": ["braycurtis", "canberra", "jensenshannon"],
    "iterations": 100000,
    "is_show": True,
    # "iterations": 1000
}

# To show the graph better, limit of y axis must be set in advance.
# At last, it is only applied for "chebyshev" and "euclidean" distance.
max_ylim = {"braycurtis": 0.06, "canberra": 1.2, "chebyshev": 25, "cityblock": 150, "cosine": 0.007,
            "euclidean": 25, "KLdivergence": 0.008, "correlation": 0.1}

# Use to zoom in the size of the bubble in the bubble chart.
size_dict = {"braycurtis": 75000, "canberra": 6000, "jensenshannon": 80000}

# Most of the experiments are base on threshold = 0.1.
threshold = 0.1

# A constant standard deviation of the normal distribution
sigma = 0.1

path_for_thesis = "../data/for_thesis/"

rho_table = {
    3: 0.0781,
    4: 0.0446,
    5: 0.0347,
    6: 0.0300,
    7: 0.0272,
    8: 0.0253,
    9: 0.0239,
    10: 0.0229
}

key_names = {
    "kn": "kii of NSI PC matrix",
    "knn": "kii of new NSI PC matrix",
    "dnp": "matrix distance between the NSI PC matrix and PC matrix",
    "dnnn": "matrix distance between the NSI PC matrix and new NSI PC matrix",
    "dnnp": "matrix distance between the new NSI PC matrix and PC matrix",
}

alpha_table = {
    "plan1": 1,
    "plan2": 10,
    "plan3": 50,
    "plan4": 1000
}


if __name__ == '__main__':
    pass
