# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/2/27
@description:
1. Use several methods to compute matrix distance.
Euclidean distance
Chebyshev distance
...
2. Analyze the results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from config import config
from utils.gerenal_tools import open_pickle, save_pickle
from pairwise_comparison.generate_NSI_PC_matrices import GenerateMatrices
from utils.pairwise_comparison_tools import compute_kii


class MatrixDistance(object):
    def __init__(self):
        pass

    @staticmethod
    def compute_distance(m1, m2, metric="euclidean", **kwargs):
        """
        Compute distance between each pair of the two collections of inputs.
        :param m1: ndarray
                   It is a np.array with shape (n, n) here.
        :param m2: ndarray
                   It is a np.array with shape (n, n) here.
        :param metric: string, default = "euclidean".
                      The distance function can be "braycurtis", "canberra", "chebyshev", "cityblock",
                      "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulsinski",
                      "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
                      "sokalmichener", "sokalsneath", "sqeuclidean", "wminkowski", "yule", "KLdivergence".
        :return: ds float
                 The value of distance.
        """
        if metric == "KLdivergence":
            """
            entropy(): Calculate the entropy of a distribution for given probability values.
            m1.shape = (n, n)
            m1.reshape(1, -1).shape = (1, n^2)
            np.squeeze(m1.reshape(1, -1)).shape = (n^2,)
            """
            ds = entropy(np.squeeze(m1.reshape(1, -1)), np.squeeze(m2.reshape(1, -1)))
        else:
            ds = cdist(m1.reshape(1, -1), m2.reshape(1, -1), metric=metric)
            ds = ds[0][0]  # cdist will return a ndarray, so use ds[0][0] to get a float number.
        return ds


def create_rho_table():
    """
    Create a table about matrix order, threshold and \rho, then save it in hard drive.
            threshold_1  threshold_2
    order_1  rho_{11}     rho_{12}
    order_2  rho_{21}     rho_{22}
    :return: None
    """
    temp_dict = dict()
    tmp = open_pickle(config.path_for_thesis + "rho.pkl")  # Access
    for key in range(3, 11):
        # print(key, tmp[key])
        # [{'rho': 0.038956560762328174, 'a': -1.4810377291603385, 'b': 1.3411769934668802, 'threshold': 0.05}, {}]
        for elm in tmp[key]:
            if elm["threshold"] not in temp_dict:
                temp_dict[elm["threshold"]] = [elm["rho"]]
            else:
                temp_dict[elm["threshold"]].append(elm["rho"])
    # The indices are orders, the columns are thresholds, and the elements are values of \rho.
    df = pd.DataFrame(temp_dict, index=range(3, 11))
    print(df)
    save_pickle(df, config.path_for_thesis + "rho and order table.pkl")


def generate_and_compute_the_distances(metric_list, iterations=1000):
    """
    Generate PC matrices and NSI PC matrices with different constraint: \rho (the mean of Kii). Then compute the
    distances between them. Finally, save the results.
    :param metric_list: list
           A list of metrics. Check config.mc_para_1.
    :param iterations: int, default = 1000
           Check config.mc_para_1.
    :return: None
    """
    md = MatrixDistance()
    tb = open_pickle(config.path_for_thesis + "rho and order table.pkl")  # Get the table about the \rho.
    print(tb)
    res = dict()
    for metric in metric_list:  # For each metric in the list
        res[metric] = dict()  # res = {metric1: {}, metric2: {}, ... }
        for ind in tb.index:  # tb is the table of \rho and order, and the index of tb is the matrix order.
            order = ind  # Rename it for better understanding.
            res[metric][order] = dict()
            # Threshold is the mean of Kii, which is set in advance.
            for threshold in np.array(range(8, 14, 1)) / 100.0:
                print(metric, ind, threshold)
                rho = tb.loc[ind][threshold]  # Select the \rho from the table.
                # Generate matrices.
                gm = GenerateMatrices(order=order, iterations=iterations, std_rate=rho)
                # Compute all the distances in loops
                res[metric][order][threshold] = list()  # res = {metric1: {order1:{threshold1: []}}}
                for _ in range(iterations):
                    # generate and get the PC and NSI PC matrix
                    tmp = gm.generate_matrix()
                    pc = tmp["PC"]
                    nsi_pc = tmp["NSI_PC"]
                    # Compute the distance between PC matrices and NSI PC matrices
                    dt = md.compute_distance(pc, nsi_pc, metric)
                    res[metric][order][threshold].append(dt)
    save_pickle(res, config.path_for_thesis + "distances for %d matrices" % iterations)


def analysis_results_chart1(metric_list, iterations, is_show=False, dpi=600):
    """
    Draw Letter-Value Plots for Section 4.2 in the thesis.
    :param metric_list: list
           A list of metrics. Check config.mc_para_1.
    :param iterations: int
           Check config.mc_para_1.
    :param is_show: boolean, default = False
           Show the plot or not. Check config.mc_para_1.
    :param dpi: int, default = 600
           When dpi is very high, the speed that latex compile the file is very slow. 200 is recommended for test.
           Check config.mc_para_1.
    :return: None
    """
    # Access the results computed through function: generate_and_compute_the_distances
    # res = {metric: {order: {threshold: []}}}
    res = open_pickle(config.path_for_thesis + "distances for %d matrices" % iterations)

    for metric in metric_list:
        tmp = dict()
        for order in res[metric]:  # res[metric] = {order: {threshold: []}}
            # All experiments are base on threshold = 0.1. So select the data where threshold = 0.1.
            tmp[order] = res[metric][order][config.threshold]
        df = pd.DataFrame(tmp)  # Convert to pandas.Dataframe.
        order = list(range(3, 11))  # The X-axis of the graph.
        plot_name = config.printed_metric[metric]  # Covert the metric names to print.
        """
        For Chebyshev distance and Euclidean distance, there are two graphs need to draw.
        The first one is the whole graph, and the second one the partial enlarged view.
        """
        if metric in ["chebyshev", "euclidean"]:
            sns.boxenplot(data=df, order=order)  # Letter-Value Plot
            plt.title("The Distribution of %s" % plot_name)  # Set the title of the graph.
            plt.xlabel('the order of matrices')  # Set the label of X-axis.
            plt.ylabel('distance/similarity/divergence')  # Set the label of Y-axis.
            # If is_show, then show the graph. Otherwise, save it.
            if is_show:
                plt.show()
            else:
                plt.savefig(config.path_for_thesis + "distribution_of_distances_%s_a.png" % metric,
                            format="png",
                            dpi=dpi)

            sns.boxenplot(data=df, order=order, showfliers=False)  # Letter-Value Plot
            plt.ylim([0, config.max_ylim[metric]])  # Limit the Y-axis to show more details.
            plt.title("The Distribution of %s" % plot_name)
            plt.xlabel('the order of matrices')
            plt.ylabel('distance/similarity/divergence')
            if is_show:
                plt.show()
            else:
                plt.savefig(config.path_for_thesis + "distribution_of_distances_%s_b.png" % metric,
                            format="png",
                            dpi=dpi)
        else:
            sns.boxenplot(data=df, order=order)  # Letter-Value Plot
            plt.title("The Distribution of %s" % plot_name)
            plt.xlabel('the order of matrices')
            plt.ylabel('distance/similarity/divergence')
            if is_show:
                plt.show()
            else:
                plt.savefig(config.path_for_thesis + "distribution_of_distances_%s.png" % metric, format="png", dpi=dpi)


def analysis_results_table(metric_list, iterations, need_order):
    """
    Generate the tables to show the statistical indicators for different orders, which is displayed in
    Section 4.2 of the thesis.
    :param metric_list: list
           A list of metrics. Check config.mc_para_1.
    :param iterations: int
           Check config.mc_para_1.
    :param need_order: int
           Check config.mc_para_1.
           I only set order=4 or order=8 for my thesis.
    :return: None
    """
    # Access the results computed through function: generate_and_compute_the_distances
    # res = {metric: {order: {threshold: []}}}
    res = open_pickle(config.path_for_thesis + "distances for %d matrices" % iterations)
    need_merge = list()
    for metric in metric_list:
        tmp = dict()
        for order in res[metric]:  # res[metric] = {order: {threshold: []}}
            # All experiments are base on threshold = 0.1. So select the data where threshold = 0.1.
            tmp[order] = res[metric][order][config.threshold]
        df = pd.DataFrame(tmp)
        """
        An example of df.describe()
                          3              4   ...             9              10
        count  100000.000000  100000.000000  ...  100000.000000  100000.000000
        mean        0.024043       0.014955  ...       0.009011       0.008663
        std         0.010873       0.005100  ...       0.001548       0.001369
        min         0.000922       0.000649  ...       0.001851       0.002561
        25%         0.016504       0.011468  ...       0.008016       0.007785
        50%         0.022294       0.014261  ...       0.008873       0.008547
        75%         0.029504       0.017615  ...       0.009832       0.009401
        max         0.141391       0.073217  ...       0.022877       0.022084
        """
        need_merge.append(df.describe()[need_order])
    mdf = pd.concat(need_merge, axis=1)  # Merge all pandas.Series and get a big matrix or pandas.Dataframe.
    mdf.columns = [config.printed_metric[elm] for elm in metric_list]  # Set the column names.
    mdf = mdf.T  # Transpose the matrix.
    mdf = mdf[["mean", "std", "min", "25%", "50%", "75%", "max"]]  # Select needed statistical measurements.
    # Print specific sentences, which is designed for writing the table in Latex.
    print(" name & " + " & ".join(list(mdf.columns)) + "\\\\" + " \\hline")
    for ind in mdf.index:
        print(ind.split()[0] + " & " + " & ".join(map(lambda x: str(round(x, 4)), list(mdf.loc[ind]))) + "\\\\" + " \\hline")


def analysis_results_chart2(metric_list, iterations, is_show=False, dpi=600):
    """
    Draw bubble charts for Section 4.3 in the thesis.
    :param metric_list: list
           A list of metrics. Check config.mc_para_2.
    :param iterations: int
           Check config.mc_para_2.
    :param is_show: boolean, default = False
           Show the plot or not. Check config.mc_para_2.
    :param dpi: int, default = 600
           When dpi is very high, the speed that latex compile the file is very slow. 200 is recommended for test.
           Check config.mc_para_2.
    :return: None
    """
    # Access the results computed through function: generate_and_compute_the_distances
    # res = {metric: {order: {threshold: []}}}
    res = open_pickle(config.path_for_thesis + "distances for %d matrices" % iterations)
    for metric in metric_list:
        fig = plt.figure(figsize=(12.00, 8.00))
        # Means are used to set the points of bubbles while standard deviations are used to set the size of bubbles.
        tmp1 = dict()  # To save the values of mean
        tmp2 = dict()  # To save the values of std
        plot_name = config.printed_metric[metric]
        for order in res[metric]:  # res[metric] = {metric: {order: {threshold: []}}}
            tmp1[order] = dict()
            tmp2[order] = dict()
            for threshold in res[metric][order]:  # res[metric][order] = {threshold: []}
                tmp1[order][threshold] = np.mean(res[metric][order][threshold])
                tmp2[order][threshold] = np.std(res[metric][order][threshold])
        df1 = pd.DataFrame(tmp1)
        """
        An example of df1
        The indices are the thresholds. The columns are matrices' orders. The elements are mean of distances.
                    3         4         5   ...        8         9         10
        0.08  0.019007  0.011864  0.009661  ...  0.007491  0.007152  0.006878
        0.09  0.021488  0.013406  0.010922  ...  0.008462  0.008074  0.007772
        0.10  0.024043  0.014955  0.012183  ...  0.009432  0.009011  0.008663
        0.11  0.026739  0.016543  0.013473  ...  0.010423  0.009945  0.009572
        0.12  0.029309  0.018190  0.014790  ...  0.011426  0.010894  0.010493
        0.13  0.031895  0.019769  0.016071  ...  0.012431  0.011862  0.011411
        """
        df2 = pd.DataFrame(tmp2)
        """
        An example of df2
        The indices are the thresholds. The columns are matrices' orders. The elements are mean of distances.
                    3         4         5   ...        8         9         10
        0.08  0.008581  0.004023  0.002682  ...  0.001405  0.001231  0.001086
        0.09  0.009759  0.004555  0.003033  ...  0.001592  0.001385  0.001232
        0.10  0.010873  0.005100  0.003373  ...  0.001766  0.001548  0.001369
        0.11  0.012127  0.005634  0.003737  ...  0.001955  0.001699  0.001509
        0.12  0.013237  0.006177  0.004099  ...  0.002150  0.001867  0.001656
        0.13  0.014382  0.006742  0.004443  ...  0.002327  0.002038  0.001806
        """
        cnt = 0  # counter
        for ind in df1.index:
            x = df1.columns
            y = df1.loc[ind]
            # The original size of the bubbles are too small. So set a ratio to zoom in it.
            size = df2.loc[ind] * config.size_dict[metric]
            plt.scatter(x, y, size, c=x, cmap=cm.get_cmap("coolwarm"))
            plt.plot(x, y, config.color_list[cnt], linestyle="--", label="the mean of Kii=%0.2f" % ind)
            cnt += 1
        plt.title("Distributions of %ss with Respect to Different Matrix Orders and Means of Kii" % plot_name)
        plt.xlabel('the order of matrices')
        plt.ylabel('the mean of distances/similarities/divergences')
        plt.legend()
        if is_show:
            plt.show()
        else:
            fig.savefig(config.path_for_thesis + "thresholds_%s_distribution.png" % metric, format="png", dpi=dpi)


def func_canberra_distance(error, m):
    """
    A quick method to compute the canberra distance between two numbers.
    n = m + error, m > 0 and n > 0
    d = |n-m| / (|m| + |n|) = |e| / (2m + error)
    :param error: float
    :param m: float
    :return: float
             The canberra distance.
    """
    return abs(error) / (2 * m + error)


def differences_between_canberra_distances(sigma, delta_sigma):
    """
    Generate two random errors from two different normal distributions with the original number m,
    then compute the canberra distances between two random samples and m.
    After that, return the differences between two canberra distances.
    :param sigma: float
           The standard deviation of the normal distribution.
    :param delta_sigma: float
           Measure the change of sigma.
    :return: float
            The differences between two canberra distances.
    """
    while 1:
        m = np.random.random() / np.random.random()  # m is the elements of any PC matrices.
        gm = GenerateMatrices()
        error = gm.random_numbers(sigma, m, 0)  # Generate a random error from original distribution.
        new_error = gm.random_numbers(sigma + delta_sigma, m, 0)  # Generate another error from the new distribution.
        if m > 0 and m + error > 0 and m + error + new_error > 0:  # All random values should be greater than zero.
            return func_canberra_distance(new_error, m) - func_canberra_distance(error, m)


def analysis_result_canberra_distance(iterations, is_show=False):
    """
    Draw a heat map for canberra distance which is also used in Section 4.3 of the thesis.
    The heat map demonstrates the distributions of the distances when order = 3.
    :param iterations: int
           Check config.mc_para_3.
    :param is_show: boolean, default = False
           Show the plot or not. Check config.mc_para_3.
    :return: None
    """
    # sigma = \rho * origin_num, [0.1, 0.2, ... 1]
    fig = plt.figure(figsize=(12.00, 8.00))
    cnt = 1
    # \rho - \kappa table when order = 3, kappa is defined in thesis as the mean of Kii.
    kappas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.56]
    rho_table = {0.1: 0.0781, 0.2: 0.1705, 0.3: 0.274, 0.4: 0.4003, 0.5: 0.5783, 0.56: 0.8608}
    for kappa in kappas:
        x = list(range(iterations))
        y = list()
        for _ in range(iterations):
            value = differences_between_canberra_distances(config.sigma, delta_sigma=rho_table[kappa])
            y.append(value)
        df = pd.DataFrame({'x': x, 'y': y, 'color': pd.cut(y, 10, labels=range(1, 11))})
        print(df)
        plt.subplot(2, 3, cnt)  # Set 6 sub-plots.
        cmap = sns.cubehelix_palette(start=0.1, light=1, as_cmap=True)
        sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5)  # Draw heat maps.
        plt.title("%s = %0.2f (%s = %0.2f)" % (chr(954), kappa, chr(961), rho_table[kappa]), config.ft)
        cnt += 1
    plt.suptitle("The Differences Distribution with Respect to %s" % chr(954))  # Set the sub titles.
    if is_show:
        plt.show()
    else:
        # fig.savefig(config.path_for_thesis + "distributions_of_differences_cd_highDPI.png", format="png", dpi=200)
        fig.savefig(config.path_for_thesis + "distributions_of_differences_cd.png", format="png", dpi=100)


def create_table_for_alpha(metric_list, iterations):
    """
    Create a alpha-order table, which will be used in reconstruct.py and Section 5.1 of the thesis.
    :param metric_list: list
           A list of metrics. Check config.mc_para_2.
    :param iterations: int
           Check config.mc_para_3.
    :return: None
    """
    table = dict()
    # Access the results computed through function: generate_and_compute_the_distances
    # res = {metric: {order: {threshold: []}}}
    res = open_pickle(config.path_for_thesis + "distances for %d matrices" % iterations)
    # Print specific sentences, which is designed for writing the table in Latex.
    print("Order & Bray-Curtis Distance & Canberra Distance & Jensen-Shannon Divergence" + "\\\\" + " \\hline")
    for order in range(3, 11):
        for metric in res:
            if metric in metric_list:
                table[metric] = dict()
                rm = GenerateMatrices(iterations=iterations, order=order)
                arrays = rm.read_array()  # Access the NPI PC matrices from hard drive.
                nsi_pc_list = arrays[1]
                tmp = list()
                for ind in range(len(nsi_pc_list)):
                    m_prime = np.squeeze(nsi_pc_list[ind])  # (n, n, 1) -> (n, n)
                    tmp.append(compute_kii(m_prime))  # Compute the kii of the matrices
                table[metric][order] = {"mean of kii": np.mean(tmp),
                                        "mean of distances": np.mean(res[metric][order][config.threshold]),
                                        "ratio": np.mean(tmp) / np.mean(res[metric][order][config.threshold])
                                        }
        # Print specific sentences, which is designed for writing the table in Latex.
        print(" & ".join([str(order), str(round(table["braycurtis"][order]["ratio"], config.decimal_places)),
                          str(round(table["canberra"][order]["ratio"], config.decimal_places)),
                          str(round(table["jensenshannon"][order]["ratio"], config.decimal_places))])
              + "\\\\" + " \\hline")
    save_pickle(table, config.path_for_thesis + "alpha table for %d matrices" % iterations)


if __name__ == '__main__':
    # m1 = np.array(range(1, 10)).reshape(3, 3)
    # m2 = m1 + np.random.randn()
    # print(m1, "\n", m2)
    # md = MatrixDistance()
    # # md.compute_distance(m1, m2)
    # create_rho_table()

    # generate_and_compute_the_distances(config.mc_para_1["metrics"], config.mc_para_1["iterations"])
    # analysis_results_chart1(config.mc_para_1["metrics"], config.mc_para_1["iterations"],
    #                         config.mc_para_1["is_show"], config.mc_para_1["dpi"])
    # analysis_results_table(config.mc_para_1["metrics"],
    #                        config.mc_para_1["iterations"],
    #                        config.mc_para_1["need_order"])

    # analysis_results_chart2(config.mc_para_2["metrics"], config.mc_para_2["iterations"],
    #                         config.mc_para_2["is_show"], config.mc_para_2["dpi"])

    # analysis_result_canberra_distance(config.mc_para_3["iterations"], config.mc_para_3["is_show"])
    create_table_for_alpha(config.mc_para_3["metrics"], config.mc_para_3["iterations"])


