import numpy as np
import random as rnd
from conjunction import *

class Teacher:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def get_correct_label(self, data):
        index = data_get_index(data)
        return self.Y[index]

    def is_correct_label(self, data, prediction_label):
        true_label = self.get_correct_label(data)
        res = true_label == prediction_label
        return res

    def _get_diff_features(self, data1, data2, existing_features):
        data1_features = data_get_features(data1)
        data2_features = data_get_features(data2)
        diff_features = np.where(data1_features != data2_features)[0]
        if len(existing_features):
            diff_features = np.setdiff1d(diff_features, existing_features)
        return diff_features
    
    def get_new_feature(self, data, conjunction):
        raise NotImplementedError  

# select random feature
class TeacherA(Teacher):
    def get_new_feature(self, data, conjunction):
        # find different features between data1 and data2
        existing_features = conjunction.get_predicate_indices()
        base_data = conjunction.get_base_data()
        diff_features = self._get_diff_features(data, base_data, existing_features)
        assert len(diff_features) > 0
        
        # choose random feature from the diff
        idx = rnd.randint(0, len(diff_features) - 1)
        feature_idx = diff_features[idx]
        return feature_idx

# select feature according to best delta in group stats
class TeacherB(Teacher):
    def __init__(self, X, Y):
        super().__init__(X, Y)
        # build stats for each type
        self.types_stats = {}
        self.types_count = {}
        features_shape = data_get_features(X[0]).shape
        for data in X:
            label = self.get_correct_label(data)
            if not label in self.types_stats:
                self.types_stats[label] = np.zeros(features_shape, dtype=np.int32)
                self.types_count[label] = 0    
            self.types_stats[label] = np.add(self.types_stats[label], data_get_features(data))
            self.types_count[label] += 1
        for label in self.types_stats:
            self.types_stats[label] = self.types_stats[label] / self.types_count[label]
    
    def get_new_feature(self, data, conjunction):
        # find different features between data1 and data2
        existing_features = conjunction.get_predicate_indices()
        base_data = conjunction.get_base_data()
        diff_features = self._get_diff_features(data, base_data, existing_features)
        assert len(diff_features) > 0
        
        # find the best feature that has the biggest delta between labels        
        data_stats = self.types_stats[self.get_correct_label(data)]
        base_data_stats = self.types_stats[self.get_correct_label(base_data)]
        diff_arr = np.abs(data_stats[diff_features] - base_data_stats[diff_features])
        max_idx = np.argmax(diff_arr)

        return diff_features[max_idx]

# find best feature that has the biggest delta from group center and the prediction
class TeacherC(Teacher):
    def get_new_feature(self, data, conjunction):
        # find different features between data1 and data2
        existing_features = conjunction.get_predicate_indices()
        base_data = conjunction.get_base_data()
        diff_features = self._get_diff_features(data, base_data, existing_features)
        assert len(diff_features) > 0
        
        # find best feature
        features = data_get_features(data)
        diff_arr = np.abs(conjunction.center[diff_features] - features[diff_features])
        max_idx = np.argmax(diff_arr)
        return diff_features[max_idx]

class TeacherD(Teacher):
    def __init__(self, X, Y):
        super().__init__(X, Y)
        self.types_stats = {}
        self.types_to_items = {}
        features_shape = data_get_features(X[0]).shape
        for data in X:
            label = self.get_correct_label(data)
            if not label in self.types_to_items:
                self.types_to_items[label] = [data]
            else:
                self.types_to_items[label].append(data)
            if not label in self.types_stats:
                self.types_stats[label] = np.zeros(features_shape, dtype=np.int32)
            self.types_stats[label] = np.add(self.types_stats[label], data_get_features(data))
    
    def _get_group_stats(self, conjunction):
        pred_indices = conjunction.get_predicate_indices()
        pred_values = conjunction.get_predicate_values()
        label = self.get_correct_label(conjunction.get_base_data())
        features_shape = data_get_features(self.X[0]).shape
        group_stats = np.zeros(features_shape, dtype=np.int32)
        for data in self.types_to_items[label]:
            item_features = data_get_features(data)
            if np.array_equal(item_features[pred_indices], pred_values):
                group_stats = np.add(group_stats, item_features)
        return group_stats

    def get_new_feature(self, data, conjunction):
        # find different features between data1 and data2
        existing_features = conjunction.get_predicate_indices()
        base_data = conjunction.get_base_data()
        diff_features = self._get_diff_features(data, base_data, existing_features)
        assert len(diff_features) > 0
        
        # find best feature
        predict_type_stats = self.types_stats[self.get_correct_label(data)]
        if len(existing_features):
            base_type_stats = self._get_group_stats(conjunction)
        else:
            base_type_stats = self.types_stats[self.get_correct_label(base_data)]
        
        diff_arr = np.abs(predict_type_stats[diff_features] - base_type_stats[diff_features])
        max_idx = np.argmax(diff_arr)
        return diff_features[max_idx]
    
class Node():
    def __init__(self):
        self.item_list = None
        self.feature = None
        self.left_son = None
        self.right_son = None
    
    def set_feature(self, feature):
        self.feature = feature

    def set_items(self, items):
        self.item_list = items
        features_shape = data_get_features(self.item_list[0]).shape
        self.stats = np.zeros(features_shape, dtype=np.int32)
        for data in self.item_list:
            self.stats = np.add(self.stats, data_get_features(data))
        self.stats = self.stats / len(self.item_list)

    def get_stats_of_group(self, features):
        if (self.left_son is None) and (self.right_son is None):
            return self.stats
        
        if features[self.feature]:
            return self.right_son.get_stats_of_group(features)
        else:
            return self.left_son.get_stats_of_group(features)

    def print(self):
        if (self.left_son is None) and (self.right_son is None):
            print("items", self.item_list)
            print("stats", self.stats)
        else:
            print("feature", self.feature)

        if (self.right_son is not None):
            print("right:")
            self.right_son.print()
        if (self.left_son is not None):
            print("left:")
            self.left_son.print()

class TeacherE(Teacher):
    def build_node(self, items_list, features_list, max_level_num):
        node = Node()
        if len(items_list) < self.min_group_size or (max_level_num <= 1):
            node.set_items(items_list)
            return node

        features_shape = data_get_features(items_list[0]).shape
        items_stats = np.zeros(features_shape, dtype=np.int32)
        for data in items_list:
            items_stats = np.add(items_stats, data_get_features(data))
        idx_array = list(range(0, len(items_stats)))
        valid_idx = np.delete(idx_array, features_list)
        valid_stats = np.delete(items_stats, features_list)
        mid_value = len(items_list) / 2
        mid_idx = np.abs(valid_stats - mid_value).argmin()
        mid_idx = valid_idx[mid_idx]
        
        upper_limit = int(len(items_list) * ((50 + self.percentage_from_mid) / 100))
        lower_limit = int(len(items_list) * ((50 - self.percentage_from_mid) / 100))
        if (items_stats[mid_idx] < lower_limit) or (items_stats[mid_idx] > upper_limit):
            node.set_items(items_list)
            return node

        right_list = []
        left_list = []
        for data in items_list:
            if (data_get_features(data)[mid_idx]):
                right_list.append(data)
            else:
                left_list.append(data)
        node.set_feature(mid_idx)
        features_list = np.append(features_list, mid_idx)
        node.right_son = self.build_node(right_list, features_list, max_level_num - 1)
        node.left_son = self.build_node(left_list, features_list, max_level_num - 1)
        
        return node
    
    def __init__(self, X, Y, percentage_from_mid, max_tree_level, min_group_size):
        super().__init__(X, Y)
        label_items = {}
        self.label_node = {}
        self.percentage_from_mid = percentage_from_mid
        self.min_group_size = min_group_size
        for data in X:
            label = self.get_correct_label(data)
            if not label in label_items:
                label_items[label] = [data]
            else:
                label_items[label].append(data)

        for label in label_items: 
            self.label_node[label] = self.build_node(label_items[label],
                                                     np.empty([0], dtype=np.int32),
                                                     max_tree_level)

    def get_new_feature(self, data, conjunction):
        # find different features between data1 and data2
        existing_features = conjunction.get_predicate_indices()
        base_data = conjunction.get_base_data()
        diff_features = self._get_diff_features(data, base_data, existing_features)
        assert len(diff_features) > 0
        
        # find the best feature that has the biggest delta between groups
        data_stats = self.label_node[self.get_correct_label(data)].get_stats_of_group(data_get_features(data))
        base_data_stats = self.label_node[self.get_correct_label(base_data)].get_stats_of_group(data_get_features(base_data))
        diff_arr = np.abs(data_stats[diff_features] - base_data_stats[diff_features])
        max_idx = np.argmax(diff_arr)
        return diff_features[max_idx]
    