# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/2/24
@description:
1. Generate NSI PC matrices, save and test them.
2. Fit the curve which is used to display the relations between rho and mean of Kii.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy
from itertools import combinations
from scipy.optimize import curve_fit
from collections import OrderedDict
from config import config
from utils.pairwise_comparison_tools import compute_kii
from utils.gerenal_tools import open_pickle, save_pickle, save_hickle, open_hickle


class GenerateMatrices(object):
    def __init__(self, order=3, iterations=1000, std_rate=1):
        """

        :param order: int, default = 3
               The order of generated matrices.
        :param iterations: int, default = 1000
               The number of generated matrices.
        :param std_rate: float, default = 1
               It is the parameter \rho in the thesis. sigma = self.std_rate * origin_num.
        """
        self.order = order
        self.iterations = iterations
        self.std_rate = std_rate
        self.result = OrderedDict()  # A ordered dict to save the results
        # Save the generated matrices with file name below.
        # The Kii threshold will always be 0.1, which is determined in the thesis.
        self.file_name_of_pc = "%d pc matrices with order=%d kii_threshold=%0.1f.pkl" % \
                               (self.iterations, self.order, 0.1)  # for PC matrices
        self.file_name_of_nsi_pc = "%d nsi pc matrices with order=%d kii_threshold=%0.1f.pkl" % \
                                   (self.iterations, self.order, 0.1)  # for NSI PC matrices

    @staticmethod
    def random_numbers(sigma, origin_num, mu=0):
        """
        A static method used to generate errors for the elements of the original PC matrices.
        Errors followed normal distribution with mean = mu, standard deviation = sigma.
        And make sure origin_num + error > 0
        :param sigma: float
               Standard deviation of the normal distribution
        :param origin_num: float
               The elements in the PC matrix. The value must be greater than 0.
        :param mu: float, default = 0
               Mean of the normal distribution. The value is zero and will not be changed during this experiment.
        :return: error, float
                 A float number which refers to the random error of the PC matrices' elements.
        """
        while 1:
            # numpy.random.randn() can return a sample from the standard normal distribution.
            # So for random samples from N(\mu, \sigma^2), they are sigma * np.random.randn() + mu.
            error = sigma * np.random.randn() + mu
            # The error must make sure the sum of the error and original element is greater than zero.
            # If the error meets the requirement, then break.
            if origin_num + error > 0:
                break
        return error

    def generate_matrix(self):
        """
        Generate PC matrices and NSI PC matrices.
        :return: dict
                {"NSI_PC": nsi_pc, "PC": pc}
        """
        # numpy.random.rand(n) can generate a random array with shape (n, 1).
        # However, the elements of this array can be zero. So if it happens, the array should be discarded.
        while 1:
            vector = np.random.rand(self.order)  # The range of the samples is [0, 1).
            if 0 not in vector:  # If 0 in the array, repeat the process. Otherwise, terminate the loop.
                break
        # np.eye() can return a 2-D array with ones on the diagonal and zeros elsewhere.
        pc = np.eye(self.order)
        # Use copy.deepcopy here to create another matrix.
        nsi_pc = copy.deepcopy(pc)
        """
        Permutations and combinations are itertools functions, which are designed to return successive elements 
        in the iterable.
        permutations(range(0, 2), 2) ==> (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1).
        combinations(range(0, 2), 2) ==> (0, 1), (0, 2), (1, 2).
        """
        temp = combinations(range(0, self.order), 2)
        """
        Generate the PC matrix according the array vector. The element of the PC matrix a_{ij} is equal to 
        vector_i / vector_j. Because I use combinations here, the iterable only has half of the needed elements. 
        So two elements of the PC matrices, a_{ij} and a_{ji}, must be generated in one loop.
        The O(n) for permutations is n(n-1).
        The O(n) for combinations is n(n-1)/2.
        """
        for elm in temp:
            i = elm[0]
            j = elm[1]
            tmp1 = vector[i] / vector[j]
            tmp2 = vector[j] / vector[i]
            pc[i, j] = tmp1  # The PC matrix's element: a_{ij} = vector_i / vector_j.
            pc[j, i] = tmp2  # The PC matrix's element: a_{ji} = vector_j / vector_i.
            # The NSI PC matrix's element: b_{ij} = vector_i / vector_j + error.
            nsi_pc[i, j] = tmp1 + self.random_numbers(self.std_rate * tmp1, tmp1) if self.std_rate else tmp1
            # The NSI PC matrix's element: b_{ji} = vector_j / vector_i + error.
            nsi_pc[j, i] = tmp2 + self.random_numbers(self.std_rate * tmp2, tmp2) if self.std_rate else tmp2
        return {"NSI_PC": nsi_pc, "PC": pc}

    def generate_with_rho(self, start=0, end=21, step=1):
        """
        Generate NSI PC matrices with different std_rate, then compute and compare the matrices' Kii.
        It is used to draw the graph in the thesis.
        :param start: int, default = 0
        :param end: int, default = 21
        :param step: int, default = 1
        :return: None
        """
        # np.array(range(0, 21, 1)) / 100.0 will create a numpy.array: [0, 0.01, 0.02, ..., 0.2]
        for self.std_rate in np.array(range(start, end, step)) / 100.0:
            kii_list = list()  # Store the values of Kii temporarily.
            for _ in range(self.iterations):
                m = self.generate_matrix()  # Generate a PC and NSI PC matrix.
                kii_list.append(compute_kii(m["NSI_PC"]))  # Choose the NSI one and compute its Kii.
            mean_of_kii = float(np.mean(kii_list))  # Compute the mean of all values of Kii.
            self.result[self.std_rate] = mean_of_kii  # Then save the mean into the ordered dict: self.result.

    def generate_and_save(self):
        """
        Generate NSI PC matrices and save them with pickle for further research.
        Generally, it is designed to generate matrices with Kii = threshold by setting different values
        for self.std_rate.
        In default, self.std_rate = 1
        :return: None
        """
        pc_list = list()  # To save PC matrices temporarily.
        nsi_pc_list = list()  # To save NSI PC matrices temporarily.
        for _ in range(self.iterations):
            m = self.generate_matrix()
            """
            np.expand_dims(arr, axis=2): add a new dimension for a two dimension np.array arr, then the np.array list 
            can be merged in to a big 3d array in next steps.
            The shape of arr is changed from shape (n, n) to shape (n, n, 1). 
            """
            pc_list.append(np.expand_dims(m["PC"], axis=2))
            nsi_pc_list.append(np.expand_dims(m["NSI_PC"], axis=2))
        """
        Concatenate a list of arrays, which the shape is (n, n, 1), into one big array,
        and its shape is (n, n, self.iterations). Then the big array can be saved by a faster tool: hickle, 
        """
        pc_array = np.concatenate(tuple(pc_list), axis=2)
        nsi_pc_array = np.concatenate(tuple(nsi_pc_list), axis=2)
        # If the file is a numpy array, I can use hickle to accelerate and save spaces and time.
        save_hickle(pc_array, config.path_for_thesis + self.file_name_of_pc)
        save_hickle(nsi_pc_array, config.path_for_thesis + self.file_name_of_nsi_pc)

    def read_array(self):
        """
        Read hickle files and decompose into array list.
        :return: pc_list, list
                 nsi_pc_list, list
                 The lists of PC matrices and NSI PC matrices.
        """
        self.file_name_of_pc = "%d pc matrices with order=%d kii_threshold=%0.1f.pkl" %\
                               (self.iterations, self.order, 0.1)
        self.file_name_of_nsi_pc = "%d nsi pc matrices with order=%d kii_threshold=%0.1f.pkl" %\
                                   (self.iterations, self.order, 0.1)
        pc_array = open_hickle(config.path_for_thesis + self.file_name_of_pc)
        nsi_pc_array = open_hickle(config.path_for_thesis + self.file_name_of_nsi_pc)
        pc_list = np.split(pc_array, pc_array.shape[2], axis=2)  # (n, n, iterations) -> [(n, n, 1)], len()=1000
        nsi_pc_list = np.split(nsi_pc_array, nsi_pc_array.shape[2], axis=2)
        return pc_list, nsi_pc_list

    def print_results(self):
        """
        Print all values of mean of Kii for further research.
        :return:
        """
        # self.result is a ordered dict to save different values for mean of kii. Their keys are self.std_rate.
        for key in self.result:
            mean_of_kii = self.result[key]
            if key == 0:
                print("the mean of %d %d by %d PC matrices' Kii is %f" %
                      (self.iterations, self.order, self.order, mean_of_kii))
            else:
                print("standard deviation == %0.2f * m_{ij}, the mean of %d %d X %d NSI PC matrices' Kii is %f" %
                      (key, self.iterations, self.order, self.order, mean_of_kii))


def draw_means_graph(is_fit=True, is_save=False, is_generate=False, iterations=100000):
    """
    Draw graphs to display the relations between mean of Kii and the ratio \rho (or self.std_rate) of
    the standard deviation.
    :param is_fit: boolean
           If True, fit the curve. Otherwise, draw the original curve.
    :param is_save: boolean
           If True, save the graph. Otherwise, show the graph.
    :param is_generate: boolean
           If True, generate these matrices and save. Otherwise, access from the hard drive.
    :param iterations: int, default = 1000
           The number of generated matrices.
    :return:
    """
    rho_collection = dict()
    # To display the graph better, set the figsize to (12, 8).
    fig = plt.figure(figsize=(12.00, 8.00))
    # fig = plt.figure(figsize=(19.20, 10.80))
    cnt = 0  # Counter of the loop, used to choose different colors.
    for order in range(3, 11, 1):  # Matrix order [3, 11]
        if is_generate:  # Run once.
            gm = GenerateMatrices(order=order, iterations=iterations)
            gm.generate_with_rho(end=15)  # Change end from 21 to 15 for better display.
            gm.print_results()
            # Save these matrices to hard drive.
            save_pickle(gm.result, config.path_for_thesis + "mean_of_kii_order=%d.pkl" % order)
            res = gm.result
        else:
            # Access the matrices from hard drive.
            res = open_pickle(config.path_for_thesis + "mean_of_kii_order=%d.pkl" % order)

        if is_fit:  # If fit, draw the original curve and the fit curve.
            x = list(res.keys())[: 15]  # Select some std_rate/rho to analyze.
            y = list(res.values())[: 15]  # Select some mean of Kii to analyze.
            # Draw the original curve at first.
            plt.plot(x, y, color=config.color_list[cnt], marker="o", linestyle="-", label="order=%d, original curve " % order)
            popt, pcov = compute_curve(x, y)  # popt = (x, y) = (std_rate/rho, mean of Kii)
            rho_collection[order] = list()  # rho_collection = {order: []}
            for threshold in np.array(range(5, 16, 1)) / 100.0:  # threshold = [0.05, 0.06, ..., 0.15]
                # scipy.optimize.fsolve is used to find the roots (\rho) of the lambda function:
                # a * \rho ^ 2 + b \rho - threshold = 0
                rho = scipy.optimize.fsolve(lambda k: popt[0] * (k ** 2) + popt[1] * k - threshold, np.array([0]))[0]
                # Print the results and round to four decimal places.
                print("order:", order, round(popt[0], config.decimal_places), round(popt[1], config.decimal_places),
                      round(rho, config.decimal_places), popt[0] * rho ** 2 + popt[1] * rho)
                # rho_collection = {order: [{"rho":, "a":, "b":, "threshold":}, {}]}
                rho_collection[order].append({"rho": rho, "a": popt[0], "b": popt[1], "threshold": threshold})
            # Draw the fit curve. *popt = popt[0], potp[1].
            plt.plot(x, fit_function(np.array(x), *popt), color=config.color_list[cnt], linestyle="--",
                     label="order=%d, fit curve" % order)
            # Draw a line for the threshold, 0.1.
            plt.axhline(y=config.threshold, color='grey', linestyle='--')
        else:  # If not fit, draw the original curve.
            x = list(res.keys())
            y = list(res.values())
            # For this case, only draw the original curve.
            plt.plot(x, y, color=config.color_list[cnt], marker="o", linestyle="-", label="order=%d" % order)
        cnt += 1

    # Set different titles
    if is_fit:
        title = 'The Original and Fit Curves'
    else:
        title = 'The Mean of %d NSI PC Matrices\' Kii' % iterations
    plt.title(title, config.ft)
    plt.xlabel('the ratio %s of the standard deviation' % chr(961), config.ft)  # The label of X-axis.
    plt.ylabel('mean of Kii', config.ft)  # The label of Y-axis.
    plt.legend()  # Display the legend of the graph.
    if is_save:
        if is_fit:
            # Eps file for latex. Pdf file for checking.
            fig.savefig(config.path_for_thesis + "the_original_and_fit_curve.pdf", format="pdf", dpi=1200)
            fig.savefig(config.path_for_thesis + "the_original_and_fit_curve.eps", format="eps", dpi=1200)
        else:
            fig.savefig(config.path_for_thesis + "the_mean_of_NSI_PC_matrices_Kii.pdf", format="pdf", dpi=1200)
            fig.savefig(config.path_for_thesis + "the_mean_of_NSI_PC_matrices_Kii.eps", format="eps", dpi=1200)
    else:
        plt.show()  # If not is_save, show the graph.
    # Print specific sentences, which is designed for writing the table in Latex.
    if rho_collection:
        # print(rho_collection)
        for key in rho_collection:  # {order: [{"rho":, "a":, "b":, "threshold":}, {}]}
            for elm in rho_collection[key]:
                if elm["threshold"] == 0.1:
                    print("0.1 & %d & %0.4f & %0.4f & %0.4f \\\\" % (key, elm["rho"], elm["a"], elm["b"]))
        save_pickle(rho_collection, config.path_for_thesis + "rho.pkl")


def fit_function(x, a, b):
    """
    y = a * x ^ 2 + b, the function used to fit the curve.
    :param x: float
           \rho/std_rate
    :param a: float
           coefficient
    :param b: float
           coefficient
    :return: y float
             y is the mean of Kii
    """
    y = a * x ** 2 + b * x
    return y


def compute_curve(x, y):
    """
    There are some relations between x (std_rate or \rho in the thesis) and y (mean of Kii).
    So fit a curve function for it.
    :param x: std_rate, list
    :param y: mean of Kii, list
    :return: popt float
             coefficient a
             pcov float
             coefficient b
    """
    popt, pcov = curve_fit(fit_function, x, y)
    return popt, pcov


if __name__ == '__main__':
    # gm = GenerateMatrices(order=3)
    # tmp = gm.generate_matrix()
    # for key in tmp:
    #     print(tmp[key])
    # for order in range(3, 11):
    #     # gm = GenerateMatrices(order=order, iterations=1000, std_rate=config.rho_table[order])
    #     # gm = GenerateMatrices(order=order, iterations=100000, std_rate=config.rho_table[order])
    #     gm = GenerateMatrices(order=order, iterations=10000, std_rate=config.rho_table[order])
    #     gm.generate_and_save()

    # draw_means_graph(is_fit=False, is_save=False, is_generate=False)
    draw_means_graph(is_fit=True, is_save=False, is_generate=False)

