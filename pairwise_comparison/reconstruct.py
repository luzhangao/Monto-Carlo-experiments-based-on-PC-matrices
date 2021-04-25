# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/3/16
@description:
1. Reconstruct PC matrix from NSI PC matrix.
"""

import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy import optimize
from pairwise_comparison.matrix_distance import MatrixDistance
from pairwise_comparison.generate_NSI_PC_matrices import GenerateMatrices
from utils.pairwise_comparison_tools import *
from multiprocessing import cpu_count
from utils.gerenal_tools import open_pickle, save_pickle, open_hickle, save_hickle, my_round
from utils.printing_format import PrintingFormat
from config import config
from config.config import key_names

np.set_printoptions(suppress=True)  # Do not use scientific notation when printing matrix.
np.set_printoptions(threshold=np.inf)  # Do not use Ellipsis when printing matrix.


class ReconstructMatrices(MatrixDistance, GenerateMatrices):

    def __init__(self):
        super().__init__()
        GenerateMatrices.__init__(self)
        self.alpha = 1.0  # The weight coefficient in the objective function, see Section 5.1 in the thesis.
        self.m_origin = np.array([])  # PC matrix
        self.v_origin = []  # original vector
        self.metric = "braycurtis"
        self.metric_list = config.mc_para_2["metrics"]  # The three metrics needed to analyze in section 5.
        self.alpha_plan = ""
        self.file_name_of_reconstruct_result = "%d matrices reconstructed result alpha plan=%s.pkl" % \
                                               (self.iterations, str(self.alpha_plan))
        self.file_name_of_new_pc = "%d new pc matrices with order=%d metric=%s alpha=%0.4f.pkl" % \
                                   (self.iterations, self.order, self.metric, self.alpha)
        self.is_show = False  # If is_show is True, show the graph.
        self.dpi = 200  # Set the dpi of the graphs.
        self.alpha_table = {}

    def objective_function(self, v):
        """
        The objective function: f(m') = Kii(m') + \alpha * D(m', m)
        :param v: list
               v is a vector, and will be converted in to the matrix m'
        :return: float
                 The number computed by the objective function.
        """
        m_prime = vector_to_pc_matrix(v)  # Convert the vector into a PC matrix
        kii = compute_kii(m_prime)  # Compute the Kii of the matrix.
        dt = self.compute_distance(m_prime, self.m_origin, self.metric)
        goal = kii + self.alpha * dt
        return goal

    def reconstruct_matrix(self, m, beta=0.2, maxiter=1000, disp=False):
        """
        Reconstruct the PC matrix based on DE.
        :param m: np.array
               original matrix, its shape is (n, n), cannot be (n, n, 1)
        :param beta: float, default = 0.2
               beta is the coefficient of the bounds.
               bounds = [v_i - beta * v_i, v_i + beta * v_i], v_i is a element of the original vector
        :param maxiter: int, default = 1000
               The maximum number of generations.
        :param disp: boolean
               Prints the evaluated function at every iteration.
        :return: np.array
               The reconstruct matrix.
        """
        self.m_origin = m  # Rename the matrix, then it can be printed in the loop.
        bounds = list()  # Bounds for variables.
        self.v_origin = pc_matrix_to_vector(self.m_origin)
        for elm in self.v_origin:
            bounds.append((elm - elm * beta, elm + elm * beta))
        r = optimize.differential_evolution(self.objective_function, bounds, workers=cpu_count(), maxiter=maxiter,
                                            updating="deferred", disp=disp)
        new_v = r.x
        return vector_to_pc_matrix(new_v)

    def run(self, iterations=1000, alpha_plan="plan1"):
        """
        Run the main function of this algorithm. It will reconstruct matrices and save the results.
        :param iterations: int, default = 1000
        :param alpha_plan: string, default = "plan1"
               It refers to a ratio used to change the values of \alpha.
        :return: None
        """
        self.alpha_table = self.read_alpha_table()
        self.alpha_plan = alpha_plan
        self.iterations = iterations
        for self.order in range(3, 11):
            arrays = super().read_array()
            pc_list = arrays[0]  # List of PC matrices.
            nsi_pc_list = arrays[1]  # List of NSI PC matrices.
            for self.metric in self.metric_list:
                # Select the values of \alpha with respect to different metrics and orders
                self.alpha = self.alpha_table[self.metric][self.order]["ratio"] * config.alpha_table[self.alpha_plan]
                self.alpha = round(self.alpha, config.decimal_places)
                print(self.order, self.metric, "%0.4f" % self.alpha)
                new_pc_list = list()
                for ind in range(len(nsi_pc_list)):
                    m_prime = np.squeeze(nsi_pc_list[ind])  # (n, n, 1) -> (n, n)
                    new_m_prime = self.reconstruct_matrix(m_prime)
                    new_pc_list.append(np.expand_dims(new_m_prime, axis=2))
                # Save the optimized matrices for further research.
                new_pc_array = np.concatenate(tuple(new_pc_list), axis=2)
                self.file_name_of_new_pc = "%d new pc matrices with order=%d metric=%s alpha=%0.4f.pkl" % (
                    self.iterations, self.order, self.metric, self.alpha)
                save_hickle(new_pc_array, config.path_for_thesis + self.file_name_of_new_pc)

    def read_new_array(self):
        """
        Access the reconstructed PC matrices from the hard drive.
        :return: list
                 The list of reconstructed PC matrices.
        """
        self.file_name_of_new_pc = "%d new pc matrices with order=%d metric=%s alpha=%0.4f.pkl" % \
                                   (self.iterations, self.order, self.metric, self.alpha)
        new_pc_array = open_hickle(config.path_for_thesis + self.file_name_of_new_pc)
        new_pc_list = np.split(new_pc_array, new_pc_array.shape[2], axis=2)  # (3, 3, 1000) -> [(3, 3, 1)], len()=1000
        return new_pc_list

    def read_alpha_table(self):
        """
        Access the \alpha table from hard drive.
        :return: None
        """
        # return open_pickle(config.path_for_thesis + "alpha table for %d matrices" % 1000)
        return open_pickle(config.path_for_thesis + "alpha table for %d matrices" % self.iterations)

    def check_and_draw(self, iterations=1000, readable=True, alpha_plan="plan1", is_show=True):
        """
        Generate a complicated dict for drawing graphs.
        res =
              {self.order:
                          {self.metric: {"kn": [],
                                         "knn": [],
                                         "dnp": [],
                                         "dnnn": [],
                                         "dnnp": [],
                                        }
                          }
              }
        :param iterations: int, default = 1000
        :param readable: int, default=True
               If True, access the results from the hard drive. Otherwise, compute them.
        :param alpha_plan: string, default = "plan1"
               It refers to a ratio used to change the values of \alpha.
        :param is_show: boolean, default = False
               Show the plot or not.
        :return: None
        """
        self.iterations = iterations
        self.is_show = is_show
        self.alpha_table = self.read_alpha_table()
        self.alpha_plan = alpha_plan
        res = dict()
        self.file_name_of_reconstruct_result = "%d matrices reconstructed result alpha plan=%s.pkl" % \
                                               (self.iterations, str(self.alpha_plan))
        if readable:
            res = open_pickle(config.path_for_thesis + self.file_name_of_reconstruct_result)
        else:  # If not readable, compute and save the data for further research.
            for self.order in range(3, 11):
                res[self.order] = dict()  # res = {self.order: {}}
                for self.metric in self.metric_list:
                    self.alpha = self.alpha_table[self.metric][self.order]["ratio"] * config.alpha_table[
                        self.alpha_plan]  # Select the values of \alpha with respect to different metrics and orders
                    new_pc_list = self.read_new_array()  # The list of Reconstructed PC matrices.
                    arrays = super().read_array()
                    pc_list = arrays[0]  # The list of original PC matrices.
                    nsi_pc_list = arrays[1]  # The list of NSI PC matrices.
                    res[self.order][self.metric] = dict()  # res = {self.order: {self.metric: {}}}
                    # For short, use "kn", "knn" and so on. The full name is displayed in config.py.
                    temp_dict = {key_names["kn"]: [],
                                 key_names["knn"]: [],
                                 key_names["dnp"]: [],
                                 key_names["dnnn"]: [],
                                 key_names["dnnp"]: []
                                 }
                    for ind in range(len(new_pc_list)):
                        m_prime = np.squeeze(nsi_pc_list[ind])  # (n, n, 1) -> (n, n)
                        m_origin = np.squeeze(pc_list[ind])
                        new_m_prime = np.squeeze(new_pc_list[ind])
                        # The full name of keys in config.py has explained the meaning of next 5 sentences.
                        temp_dict[key_names["kn"]].append(compute_kii(m_prime))
                        temp_dict[key_names["knn"]].append(compute_kii(new_m_prime))
                        temp_dict[key_names["dnp"]].append(self.compute_distance(m_origin, m_prime, self.metric))
                        temp_dict[key_names["dnnn"]].append(self.compute_distance(m_prime, new_m_prime, self.metric))
                        temp_dict[key_names["dnnp"]].append(self.compute_distance(m_origin, new_m_prime, self.metric))
                    res[self.order][self.metric] = temp_dict
            save_pickle(res, config.path_for_thesis + self.file_name_of_reconstruct_result)

        self.create_data_tables(res)

        self.choose_data_to_draw(res, "alpha+metric+order+key-name.dnp")
        self.choose_data_to_draw(res, "alpha+metric+order+key-name.dnnn")
        self.choose_data_to_draw(res, "alpha+metric+order+key-name.dnnp")

    def choose_data_to_draw(self, res, gtype):
        """
        Draw three letter-value plots for each metric. For example:
        metric=braycurtis. X-axis: order. Y-axis: key_names.dnp.
        metric=braycurtis. X-axis: order. Y-axis: key_names.dnnn.
        metric=braycurtis. X-axis: order. Y-axis: key_names.dnnp.
        :param res: dict
            A complicated dict. {self.order:
                                            {self.metric: {"kn": [],
                                                           "knn": [],
                                                           "dnp": [],
                                                           "dnnn": [],
                                                           "dnnp": [],
                                                          }
                                            }
                                }
        :param gtype: string
               Graph types. There are three gtypes here:
               1. alpha+metric+order+key-name.dnp
               2. alpha+metric+order+key-name.dnnn
               3. alpha+metric+order+key-name.dnnp
        :return: None
        """
        xlabel = "the order of matrices"
        plot_format = "png"

        order_list = list(range(3, 11))  # The X-axis of the graph.
        if gtype == "alpha+metric+order+key-name.dnp":
            for self.metric in self.metric_list:
                tmp = dict()
                for self.order in res:
                    tmp[self.order] = res[self.order][self.metric][key_names["dnp"]]
                df = pd.DataFrame(tmp)
                ylabel = "distance"
                plot_name = "The Distribution of %s Distances between \n the NSI PC Matrices and the Original PC " \
                            "Matrices" % self.metric.capitalize()
                # gtype.split(".")[1] = "dnp"
                plot_saved_path = "new_distribution_of_distances_%s_%s.png" % (self.metric, gtype.split(".")[1])
                self.draw_box_plots(data=df, xaxis=order_list, plot_name=plot_name, xlabel=xlabel, ylabel=ylabel,
                                    plot_saved_path=plot_saved_path, plot_format=plot_format)

        elif gtype == "alpha+metric+order+key-name.dnnn":
            for self.metric in self.metric_list:
                tmp = dict()
                for self.order in res:
                    tmp[self.order] = res[self.order][self.metric][key_names["dnnn"]]
                df = pd.DataFrame(tmp)
                ylabel = "distance"
                plot_name = "The Distribution of %s Distances between \n the NSI PC Matrix and Optimized Matrix" \
                            % self.metric.capitalize()
                plot_saved_path = "new_distribution_of_distances_%s_%s.png" % (self.metric, gtype.split(".")[1])
                self.draw_box_plots(data=df, xaxis=order_list, plot_name=plot_name, xlabel=xlabel, ylabel=ylabel,
                                    plot_saved_path=plot_saved_path, plot_format=plot_format)

        elif gtype == "alpha+metric+order+key-name.dnnp":
            for self.metric in self.metric_list:
                tmp = dict()
                for self.order in res:
                    tmp[self.order] = res[self.order][self.metric][key_names["dnnp"]]
                df = pd.DataFrame(tmp)
                ylabel = "divergence"
                # plot_name = "alpha=%0.4f x %d  metric=%s" % (self.alpha_table[self.metric][self.order]["ratio"],
                #                                             config.alpha_table[self.alpha_plan], self.metric)
                plot_name = "The Distribution of %s Divergences between \n the Original PC Matrix and Optimized " \
                            "Matrix" % self.metric.capitalize()
                plot_saved_path = "new_distribution_of_distances_%s_%s.png" % (self.metric, gtype.split(".")[1])
                self.draw_box_plots(data=df, xaxis=order_list, plot_name=plot_name, xlabel=xlabel, ylabel=ylabel,
                                    plot_saved_path=plot_saved_path, plot_format=plot_format)

    def create_data_tables(self, res):
        """
        create a complicated table, see table 6 in Section 5.2 of the thesis.
        :param res: dict
                    A complicated dict. {self.order:
                                                    {self.metric: {"kn": [],
                                                                   "knn": [],
                                                                   "dnp": [],
                                                                   "dnnn": [],
                                                                   "dnnp": [],
                                                                  }
                                                    }
                                        }
        :return: None.
        """
        for self.metric in self.metric_list:
            # Print the data with some specific format, which is used to create tables in the LaTex file.
            np1 = PrintingFormat()
            np2 = PrintingFormat()
            np3 = PrintingFormat()

            np1.for_reconstruct()
            np2.for_reconstruct()
            np3.for_reconstruct()
            for self.order in res:
                tmp1 = dict()
                tmp2 = dict()
                tmp3 = dict()
                """
                res = 
                      {self.order:
                                  {self.metric: {"kn": [],
                                                 "knn": [],
                                                 "dnp": [],
                                                 "dnnn": [],
                                                 "dnnp": [],
                                                }
                                  }
                      }
                """
                tmp1[self.order] = res[self.order][self.metric][key_names["dnp"]]
                tmp2[self.order] = res[self.order][self.metric][key_names["dnnn"]]
                tmp3[self.order] = res[self.order][self.metric][key_names["dnnp"]]

                df1 = pd.DataFrame(tmp1)
                df2 = pd.DataFrame(tmp2)
                df3 = pd.DataFrame(tmp3)
                # print(df1.describe().loc[["mean", "std", "min", "max"]])
                # print(df2.describe().loc[["mean", "std", "min", "max"]])
                # print(df3.describe().loc[["mean", "std", "min", "max"]])

                np1.for_reconstruct(mean=df1.describe()[self.order]["mean"],
                                    std=df1.describe()[self.order]["std"],
                                    min_value=df1.describe()[self.order]["min"],
                                    max_value=df1.describe()[self.order]["max"])
                np2.for_reconstruct(mean=df2.describe()[self.order]["mean"],
                                    std=df2.describe()[self.order]["std"],
                                    min_value=df2.describe()[self.order]["min"],
                                    max_value=df2.describe()[self.order]["max"])
                np3.for_reconstruct(mean=df3.describe()[self.order]["mean"],
                                    std=df3.describe()[self.order]["std"],
                                    min_value=df3.describe()[self.order]["min"],
                                    max_value=df3.describe()[self.order]["max"])
            np1.for_reconstruct(end=True)
            np2.for_reconstruct(end=True)
            np3.for_reconstruct(end=True)

            print(self.metric)
            print("=" * 100)
            np1.print_need_print()
            print("=" * 100)
            np2.print_need_print()
            print("=" * 100)
            np3.print_need_print()
            print("=" * 100)

    def draw_box_plots(self, data, xaxis, plot_name, xlabel, ylabel, plot_saved_path, plot_format):
        """
        Draw box plots according to different parameters.
        :param data: pandas.Dataframe
               The data used to draw the graph.
        :param xaxis: list
               The values of X-axis.
        :param plot_name: string
               The name of this graph.
        :param xlabel: string
               The label of X-axis.
        :param ylabel: string
               The label of Y-axis.
        :param plot_saved_path: string
               The file path to save the graph.
        :param plot_format: string
               png or other formats.
        :return: None
        """
        sns.boxenplot(data=data, order=xaxis)  # Draw the box plot.
        plt.title(plot_name, config.new_ft)
        plt.xlabel(xlabel, config.new_ft)
        plt.ylabel(ylabel, config.new_ft)
        if self.is_show:
            plt.show()
        else:
            plt.savefig(config.path_for_thesis + plot_saved_path, format=plot_format,
                        dpi=self.dpi)
            plt.close()

    def check_outliers_of_kii(self, alpha_plan):
        """
        Check if there is any Kii of reconstructed PC matrix != 0.
        :param alpha_plan: String
               plan1, plan2, plan3
        :return:
        """
        self.alpha_table = self.read_alpha_table()
        self.alpha_plan = alpha_plan
        self.file_name_of_reconstruct_result = "%d matrices reconstructed result alpha plan=%s.pkl" \
                                               % (self.iterations, str(self.alpha_plan))
        for self.order in range(3, 11):
            for self.metric in self.metric_list:
                self.alpha = self.alpha_table[self.metric][self.order]["ratio"] * config.alpha_table[
                    self.alpha_plan]
                new_pc_list = self.read_new_array()
                arrays = super().read_array()
                pc_list = arrays[0]
                nsi_pc_list = arrays[1]
                for ind in range(len(new_pc_list)):
                    m_prime = np.squeeze(nsi_pc_list[ind])  # (n, n, 1) -> (n, n)
                    new_m_prime = np.squeeze(new_pc_list[ind])
                    kii1 = compute_kii(m_prime)
                    kii2 = compute_kii(new_m_prime)
                    if kii2 != 0:  # If kii2 !=0, which means the DE algorithm hasn't been converged correctly.
                        print(kii1, kii2)


if __name__ == '__main__':
    rm = ReconstructMatrices()
    # rm.run(10000, "plan1")
    # rm.check_and_draw(iterations=10000, readable=False, alpha_plan="plan1", is_show=False)
    # rm.check_and_draw(readable=False, alpha_plan="plan3", is_show=True)

    # rm.check_outliers_of_kii("plan1")



