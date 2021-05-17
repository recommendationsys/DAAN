import scipy.sparse as sp
import numpy as np
import pdb

class Dataset(object):

    def __init__(self, source_path, target_path):

        self.source_item_ratingdict, self.source_user_ratingdict, self.source_user_num, self.source_item_num1 = self.load_rating_file_as_dict(source_path + ".train.rating")
        self.source_testRatings, self.source_item_num2 = self.load_rating_file_as_list(source_path + ".test.rating")
        self.source_testNegatives = self.load_negative_file(source_path + ".test.negative")

        self.target_item_ratingdict, self.target_user_ratingdict, self.target_user_num, self.target_item_num1 = self.load_rating_file_as_dict(target_path + ".train.rating")
        self.target_testRatings, self.target_item_num2 = self.load_rating_file_as_list(target_path + ".test.rating")
        self.target_testNegatives = self.load_negative_file(target_path + ".test.negative")

        self.source_item_num = self.source_item_num1 if self.source_item_num1 >= self.source_item_num2 else self.source_item_num2
        self.target_item_num = self.target_item_num1 if self.target_item_num1 >= self.target_item_num2 else self.target_item_num2

    def load_rating_file_as_list(self, filename):
        ratingList = []
        item_list = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                item_list.append(item)
                ratingList.append([user, item])
                line = f.readline()
        item_num = max(item_list) + 1
        return ratingList, item_num
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_dict(self, filename):
        user_ratingdict = {}
        item_ratingdict = {}
        user_list = []
        item_list = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                if user not in user_list:
                    user_list.append(user)
                if item not in item_list:
                    item_list.append(item)
                item_ratingdict.setdefault(item, []).append(user)
                user_ratingdict.setdefault(user, []).append(item)
                line = f.readline()
        user_num = max(user_list) + 1
        item_num = max(item_list) + 1
        return item_ratingdict, user_ratingdict, user_num, item_num