import numpy as np
import os
import random as rnd
from conjunction import *
import matplotlib.pyplot as plt

SEED = 10
TRAIN_NUM_PER_TEACHER = 4
OUTPUT_DIR = "./tmp"

def update_conjection_miss(conj_vector, teacher, data):
    data_features = data_get_features(data)
    selected_conj = None
    best_fit_score = 1
    for c in conj_vector:
        if teacher.is_correct_label(data, c.label):
            pred_indices = c.get_predicate_indices()
            pred_values = c.get_predicate_values()
            score = np.sum(data_features[pred_indices] != pred_values) / len(pred_values)
            if score < best_fit_score:
                score = best_fit_score
                selected_conj = c
    
    if selected_conj is not None:
        selected_conj.update_miss_on_data(data)


def train(X, teacher, conj_finder, max_pred_num, max_miss_num, feature_names):
    # get default prediction
    x0 = X[0]
    y0 = teacher.get_correct_label(x0)
    
    conj_vector = []
    misses = 0
    total_count = 0
    stats = []
    
    # iterate over all the dataset
    for data in X:
        # find a group that satisfies this item
        conj = conj_finder(conj_vector, data)
        if (conj is not None):
            # found matched conjunction lets try to predict
            if teacher.is_correct_label(data, conj.label):
                # predicted correctly
                conj.add_matched_item(data)
                print("correct prediction: instance=%s predict=%s similar=%s" % (data_get_name(data),
                                                                                 conj.label,
                                                                                 data_get_name(conj.data)))
            else:
                # did not predict correctly so need to add new predicate
                feature_idx = teacher.get_new_feature(data, conj)
                feature_value = data_get_features(conj.data)[feature_idx]
                conj.add_new_predicate(feature_idx, feature_value)
                print("wrong prediction: instance=%s predict=%s similar=%s feature=%s val=%d" % (data_get_name(data),
                                                                                                 conj.label,
                                                                                                 data_get_name(conj.data),
                                                                                                 feature_names[feature_idx],
                                                                                                 feature_value))
                
                # update conjenctions on miss
                if max_pred_num or max_miss_num:
                    update_conjection_miss(conj_vector, teacher, data)

                misses += 1
        else:
            # did not found any group
            # lets try to predict the default predication
            prediction = teacher.is_correct_label(data, y0)
            if not prediction:
                # did not predict correctly need to open new group
                correct_label = teacher.get_correct_label(data)
                conj = Conjunction(len(conj_vector), correct_label, data, max_pred_num, max_miss_num)
                conj_vector.append(conj)
                
                # add new feature to the group
                feature_idx = teacher.get_new_feature(x0, conj)
                feature_value = data_get_features(data)[feature_idx]
                conj.add_new_predicate(feature_idx, feature_value)
                print("wrong prediction: instance=%s predict=%s similar=%s feature=%s val=%d" % (data_get_name(data),
                                                                                                 y0,
                                                                                                 data_get_name(x0),
                                                                                                 feature_names[feature_idx],
                                                                                                 feature_value))
                # update conjunction on miss
                if max_pred_num or max_miss_num:
                    update_conjection_miss(conj_vector, teacher, data)

                misses += 1
            else:
                pass
                print("correct prediction: instance=%s predict=%s similar=%s" % (data_get_name(data),
                                                                                 y0,
                                                                                 data_get_name(x0)))
        
        total_count += 1
        stats.append((misses / total_count ) * 100)    
        
    return conj_vector, stats


def add_results(res_arr, conj_vector, stats):
    x = np.arange(1, len(stats) + 1)
    y = stats
    accuracy = "%.3f%%" % y[-1]
    print(f"The algorithm was incorrect in {accuracy}% of the times")
    res_arr.append(y[-1])
    plt.plot(x, y, label=accuracy)
    
    print("Final conjunction list:")
    idx = 0
    for c in conj_vector:
       print(f"{list(zip(c.get_predicate_indices(),c.get_predicate_values()))} ==> Label: {c.label}")

def save_results(res_arr, label):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    avg_res = sum(res_arr) / len(res_arr)
    plt.title("Training Results %s: avg_acc=%.3f" % (label, avg_res)) 
    plt.xlabel("No. of examples") 
    plt.ylabel("Percent of mistakes") 
    plt.legend(loc="upper right")
    output_file = '%s/%s.png' % (OUTPUT_DIR, label)
    plt.savefig(output_file)
    plt.close()
    print("END of run: saved result to %s" % output_file)

def run_training_session(X,
                         teacher,
                         conj_finder,
                         feature_names,
                         label,
                         max_pred_num=0,
                         max_miss_num=0):
    res_arr = []
    for i in range(TRAIN_NUM_PER_TEACHER):
        rnd.seed(SEED + i)
        X_i = rnd.sample(X, len(X))
        conj_vec, stats = train(X_i, teacher, conj_finder, max_pred_num, max_miss_num, feature_names)
        add_results(res_arr, conj_vec, stats)
    save_results(res_arr, label)
