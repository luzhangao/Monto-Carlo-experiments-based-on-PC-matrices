# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/3/29
@description:
1. Print the data with some specific format, which is used to create tables in the LaTex file.
"""

from utils.gerenal_tools import my_round


class PrintingFormat(object):
    def __init__(self):
        self.need_print1 = ""
        self.need_print2 = ""
        self.need_print3 = ""
        self.need_print4 = ""

    def for_reconstruct(self, mean=0, std=0, min_value=0, max_value=0, end=False):
        """
        It will print the data with special format.
        :param mean: float, default = 0
        :param std: float, default = 0
        :param min_value: float, default = 0
        :param max_value: float, default = 0
        :param end: boolean
                    If True, it means this is the end of the sentence,
                    should be added "\cline{2-10}" or "\hline"
        :return: None
        """
        if not end:
            if not mean and not std and not min_value and not max_value:
                self.need_print1 = "& mean"
                self.need_print2 = "& SD"
                self.need_print3 = "& min"
                self.need_print4 = "& max"
            else:
                self.need_print1 += " & " + my_round(mean)
                self.need_print2 += " & " + my_round(std)
                self.need_print3 += " & " + my_round(min_value)
                self.need_print4 += " & " + my_round(max_value)
        else:
            self.need_print1 += r" \\ \cline{2-10}"
            self.need_print2 += r" \\ \cline{2-10}"
            self.need_print3 += r" \\ \cline{2-10}"
            self.need_print4 += r" \\ \hline"

    def print_need_print(self):
        print(self.need_print1)
        print(self.need_print2)
        print(self.need_print3)
        print(self.need_print4)


if __name__ == '__main__':
    pass
