from conjunction import ConjunctionFinderRandom, ConjunctionFinderFirst, \
                        ConjunctionFinderFeatureFit, ConjunctionFinderPredicatFit
from teacher import TeacherB, TeacherC, TeacherD, TeacherE
from datasets import nursery_dataset, mushroom_dataset, chess_dataset
from learner import run_training_session

if __name__ == "__main__":
    conj_find_random = ConjunctionFinderRandom()
    conj_find_first = ConjunctionFinderFirst()
    conj_find_feature_fit = ConjunctionFinderFeatureFit()
    conj_find_predicate_fit = ConjunctionFinderPredicatFit()
    
    # create mushroom dataset
    X, Y, feature_names = mushroom_dataset()
    
    # teacherB for compare of results
    teacher = TeacherB(X, Y)
    run_training_session(X, teacher, conj_find_random, feature_names, "mushroom_teacherB_conj_random")
    run_training_session(X, teacher, conj_find_first, feature_names, "mushroom_teacherB_conj_first")
    run_training_session(X, teacher, conj_find_feature_fit, feature_names, "mushroom_teacherB_conj_feature_fit")
    run_training_session(X, teacher, conj_find_predicate_fit, feature_names, "mushroom_teacherB_conj_pred_fit")
    
    # teacher C
    teacher = TeacherC(X, Y)
    run_training_session(X, teacher, conj_find_feature_fit, feature_names, "mushroom_teacherC_conj_feature_fit")
    
    # teacher D
    teacher = TeacherD(X, Y)
    run_training_session(X, teacher, conj_find_feature_fit, feature_names, "mushroom_teacherD_conj_feature_fit")

    # teacherE
    teacher = TeacherE(X, Y, percentage_from_mid=2, max_tree_level=3, min_group_size=20)
    run_training_session(X, teacher, conj_find_feature_fit, feature_names, "mushroom_teacherE_2_0")

    # teacherB + delete
    teacher = TeacherB(X, Y)
    run_training_session(X, teacher, conj_find_feature_fit, feature_names, "mushroom_teacherB_delete", max_pred_num=5, max_miss_num=5)
    
    #best
    teacher = TeacherE(X, Y, percentage_from_mid=2, max_tree_level=3, min_group_size=20)
    run_training_session(X, teacher, conj_find_feature_fit, feature_names, "mushroom_teacherB_best", max_pred_num=5, max_miss_num=5)

    # nursery dataset
    X, Y, feature_names = nursery_dataset()
 
     # teacherB for compare
    teacher = TeacherB(X, Y)
    run_training_session(X, teacher, conj_find_first, feature_names, "nursery_teacherB_conj_first")
   
    # teacherC
    teacher = TeacherC(X, Y)
    run_training_session(X, teacher, conj_find_first, feature_names, "nursery_teacherC_conj_first")
    
    # teacherD
    teacher = TeacherD(X, Y)
    run_training_session(X, teacher, conj_find_first, feature_names, "nursery_teacherD_conj_first")

    # teacherB + delete
    teacher = TeacherB(X, Y)
    run_training_session(X, teacher, conj_find_first, feature_names, "nursery_teacherB_delete", max_pred_num=10, max_miss_num=5)

    #chess dataset
    X, Y, feature_names = chess_dataset()

    #teacherB
    teacher = TeacherB(X, Y)
    run_training_session(X, teacher, conj_find_first, feature_names, "chess_teacherB_conj_first")

    #teacherE
    teacher = TeacherE(X, Y, percentage_from_mid=25, max_tree_level=15, min_group_size=20)
    run_training_session(X, teacher, conj_find_first, feature_names, "chess_teacherE_25p_15L_conj_find")