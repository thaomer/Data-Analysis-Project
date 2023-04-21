from conjunction import ConjunctionFinderRandom, ConjunctionFinderFirst, \
                        ConjunctionFinderFeatureFit, ConjunctionFinderPredicatFit
from teacher import TeacherA, TeacherB
from datasets import zoo_dataset, nursery_dataset
from learner import run_training_session

if __name__ == "__main__":
    conj_find_random = ConjunctionFinderRandom()
    conj_find_first = ConjunctionFinderFirst()
    conj_find_feature_fit = ConjunctionFinderFeatureFit()
    conj_find_predicate_fit = ConjunctionFinderPredicatFit()

    # create zoo dataset
    X, Y, feature_names = zoo_dataset()
    
    # train zoo on teachers
    teacher = TeacherA(X, Y)
    run_training_session(X, teacher, conj_find_first, feature_names, "zoo_teacherA_conj_first")
    
    teacher = TeacherB(X, Y)
    run_training_session(X, teacher, conj_find_first, feature_names, "zoo_teacherB_conj_first")
   
    
    # create nursery dataset
    X, Y, feature_names = nursery_dataset()
    
    # train nursery on teachers
    teacher = TeacherA(X, Y)
    run_training_session(X, teacher, conj_find_first, feature_names, "nursery_teacherA_conj_first")
    
    teacher = TeacherB(X, Y)
    run_training_session(X, teacher, conj_find_first, feature_names, "nursery_teacherB_conj_first")
