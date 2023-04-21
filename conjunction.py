import numpy as np
import random as rnd
import sys

def data_get_index(data):
    return data[0]

def data_get_features(data):
    return data[1]

def data_get_name(data):
    return data[2]

class Conjunction():
    def __init__(self, index, label, data, max_pred_num, max_miss_num):
        self.index = index
        self.data = data
        self.label = label # the label of the instances
        self.predicates_indices = np.empty([0], dtype=np.int32)
        self.predicates_values = np.empty([0], dtype=np.int8)
        self.predicates_miss = np.empty([0], dtype=np.int32)
        self.fit_instances = [data] # the instances that fit the conjunction
        self.center = np.copy(data_get_features(data))
        self.max_pred_num = max_pred_num
        self.max_miss_num = max_miss_num

    def add_new_predicate(self, index, value):
        feature_idx = index
        feature_val = value
        found_idx_arr = np.where(self.predicates_indices == feature_idx)[0]
        if len(found_idx_arr):
            assert len(found_idx_arr) == 1
            assert self.predicates_values[found_idx_arr[0]] == feature_val
        else:
            self.predicates_indices = np.append(self.predicates_indices, feature_idx)
            self.predicates_values = np.append(self.predicates_values, feature_val)
            self.predicates_miss = np.append(self.predicates_miss, 0)

    def delete_predicate(self, index):
        self.predicates_miss = np.delete(self.predicates_miss, index)
        self.predicates_indices = np.delete(self.predicates_indices, index)
        self.predicates_values = np.delete(self.predicates_values, index)
                    
    def add_matched_item(self, data):
        self.fit_instances.append(data)
        new_val = np.add(self.center, data_get_features(data))
        temp = np.divide(np.subtract(new_val , self.center), len(self.fit_instances))
        self.center = np.add(self.center, temp)

    def get_predicate_indices(self):
        return self.predicates_indices

    def get_predicate_values(self):
        return self.predicates_values
    
    def get_base_data(self):
        return self.data

    def update_miss_on_data(self, data):
        if (not self.max_pred_num) and (not self.max_miss_num):
            return
    
        # update misses
        pred_indices = self.get_predicate_indices()
        pred_values = self.get_predicate_values()
        diff_arr = data_get_features(data)[pred_indices] != pred_values
        self.predicates_miss = self.predicates_miss + diff_arr
        
        # delete predicate with max misses in case we have one
        if (len(pred_values) > 1):
            max_idx = np.argmax(self.predicates_miss)
            if (len(pred_values) > self.max_pred_num) or \
                (self.predicates_miss[max_idx] > self.max_miss_num):
                self.delete_predicate(max_idx)

class ConjunctionFinder():
    def __call__(self, conj_vector, data):
        raise NotImplementedError
    
class ConjunctionFinderRandom(ConjunctionFinder):
    def __call__(self, conj_vector, data):
        item_features = data_get_features(data)
        conj_found = []
        for c in conj_vector:
            pred_indices = c.get_predicate_indices()
            pred_values = c.get_predicate_values()
            if np.array_equal(item_features[pred_indices], pred_values):
                conj_found.append(c)
        
        if (len(conj_found) > 0):
            idx = rnd.randint(0, len(conj_found) - 1)
            return conj_found[idx]
        return None

class ConjunctionFinderFirst(ConjunctionFinder):
    def __call__(self, conj_vector, data):
        item_features = data_get_features(data)
        for c in conj_vector:
            pred_indices = c.get_predicate_indices()
            pred_values = c.get_predicate_values()
            if np.array_equal(item_features[pred_indices], pred_values):
                return c
        return None

class ConjunctionFinderFeatureFit(ConjunctionFinder):
    def __call__(self, conj_vector, data):
        item_features = data_get_features(data)
        conj_found = []
        for c in conj_vector:
            pred_indices = c.get_predicate_indices()
            pred_values = c.get_predicate_values()
            if np.array_equal(item_features[pred_indices], pred_values):
                conj_found.append(c)
        if len(conj_found) > 0:
            c_found = None
            c_min_val = sys.maxsize
            for c in conj_found:
                base_data = c.get_base_data()
                base_features = data_get_features(base_data)
                score = np.sum(np.abs(np.subtract(item_features, base_features)))
                if score < c_min_val:
                    c_min_val = score
                    c_found = c
            return c_found
        return None
 
class ConjunctionFinderPredicatFit(ConjunctionFinder):
    def __call__(self, conj_vector, data):   
        item_features = data_get_features(data)
        max_pred_num = 0
        conj = None
        for c in conj_vector:
            pred_indices = c.get_predicate_indices()
            pred_values = c.get_predicate_values()
            if np.array_equal(item_features[pred_indices], pred_values):
                if len(pred_values) > max_pred_num:
                    max_pred_num = len(pred_values)
                    conj = c
        return conj
