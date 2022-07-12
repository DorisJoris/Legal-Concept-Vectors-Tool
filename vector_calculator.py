# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:05:58 2022

@author: bejob
"""

#%% Import

import numpy as np
import copy
import random
from progress.bar import Bar
from progress.spinner import Spinner

#%% Concept_vector initializer (only based on the children)
def concept_vector_init(legal_concepts, hierachical_label_list): #The heirachical_label_list should be sorted lowest to highest hierachy.
    #candidat_keys = list(legal_concepts.keys())

    bar_cvi = Bar('Processing concept vector init', max=len(hierachical_label_list))     

    for label in hierachical_label_list:
        
        for legal_concept_key in legal_concepts.keys():
            if label in legal_concepts[legal_concept_key]['labels']:
                parent_concept_bow = dict()
                sum_of_child_vec = np.array([0]*768, dtype='float32')
                n_of_child = 0
                
                
                for neighbour in legal_concepts[legal_concept_key]['neighbours']:
                    if neighbour['type'] == 'child':
                        child_id = neighbour['neighbour']
                        
                        sum_of_child_vec += legal_concepts[child_id]['concept_vector']
                        
                        child_bow = legal_concepts[child_id]['concept_bow']
            
                        n_of_child += 1
                        
                        legal_concepts[legal_concept_key]['raw_text'] += '\n' + legal_concepts[child_id]['raw_text']
                        
                        for word in child_bow.keys():
                                
                            if word in parent_concept_bow.keys():
                                parent_concept_bow[word][0] += child_bow[word][0]
                                parent_concept_bow[word][1] += child_bow[word][1]
                                parent_concept_bow[word][3] += child_bow[word][3]
                            else:
                                parent_concept_bow[word] = copy.copy(child_bow[word])

                if n_of_child > 0:        
                    
                    legal_concepts[legal_concept_key]['concept_bow'] = parent_concept_bow
                    legal_concepts[legal_concept_key]['concept_vector'] = sum_of_child_vec/n_of_child
                #candidat_keys.remove(legal_concept_key)
                
        bar_cvi.next()
                
    bar_cvi.finish()          
    return legal_concepts

  

#%% Concept_vector calculator (iterative calculation the mean of all neighbours for every concept)    
    
def concept_vector_calculator(legal_concepts, aver_dist_threshold):
    spinner = Spinner('Loading ')
    candidat_keys = list(legal_concepts.keys())
    candidat_change_count = {}
    
    iteration_count = 0
    neighbours_not_found = []
    neighbours_without_concept_vector = []
    while len(candidat_keys) > 0 and iteration_count < 1000:
        spinner.next()
        for legal_concept_key in candidat_keys:
            prior_concept_vector = legal_concepts[legal_concept_key]['concept_vector']
            if type(prior_concept_vector) != np.ndarray:
                if legal_concept_key not in neighbours_not_found:
                    neighbours_without_concept_vector.append(legal_concept_key)
                    candidat_keys.remove(legal_concept_key)
                continue
            
            sum_of_neighbours_vec = np.array([0]*768, dtype='float32')
            n_of_neighbours = 0
            
            for neighbour in legal_concepts[legal_concept_key]['neighbours']:
                neighbour_id = neighbour['neighbour']
                if neighbour_id in legal_concepts.keys():
                    neighbour_vec = legal_concepts[neighbour_id]['concept_vector']
                    if type(neighbour_vec) == np.ndarray:
                        sum_of_neighbours_vec = sum_of_neighbours_vec + neighbour_vec
                        n_of_neighbours += 1
                    else:
                        if neighbour_id not in neighbours_without_concept_vector:
                            neighbours_without_concept_vector.append(neighbour_id)
                            
                else:
                    if neighbour_id not in neighbours_not_found:
                        neighbours_not_found.append(neighbour_id)
                        
                        
            if n_of_neighbours > 0:        
                ultimo_concept_vector = sum_of_neighbours_vec/n_of_neighbours
                
                dist_prior_ultimo_concept_vector = np.linalg.norm(prior_concept_vector-ultimo_concept_vector)
                
                if dist_prior_ultimo_concept_vector > aver_dist_threshold:
                    legal_concepts[legal_concept_key]['concept_vector'] = ultimo_concept_vector
                    candidat_change_count[legal_concept_key] = {'cv_unchanged':0}
                
                else:
                    if legal_concept_key not in candidat_change_count.keys():
                        candidat_change_count[legal_concept_key] = {'cv_unchanged':1}
                    else:
                        candidat_change_count[legal_concept_key]['cv_unchanged'] += 1
            
            if candidat_change_count[legal_concept_key]['cv_unchanged'] > 1:
                candidat_keys.remove(legal_concept_key)
                
        iteration_count += 1    
                
    if iteration_count < 1000:   
        print(f"{iteration_count} numbers of iteration where needed to calculate the concept vectors.") 
        print(f"The applied average distance threshold was {aver_dist_threshold}.")
        if len(neighbours_not_found) > 0:
            print("---")
            print(f"The following {len(neighbours_not_found)} neighbour ID's could not be found:")
            for not_found_id in neighbours_not_found:
                print(not_found_id)
                
        if len(neighbours_without_concept_vector) > 0:
            print("---")
            print(f"The following {len(neighbours_without_concept_vector)} neighbour ID's had no concept vector:")
            for not_found_id in neighbours_without_concept_vector:
                print(not_found_id)
    else:
        print("Stopped after 1000 iterations!")
    return legal_concepts

#%% concept bow calculator
def concept_bow_calculator(legal_concepts, min_tf_threshold):
    spinner = Spinner('Loading ')
    candidat_keys = list(legal_concepts.keys())
    
    candidat_change_count = {}
    iteration_count = 0
    neighbours_not_found = []
    neighbours_with_empty_bows = []
    while len(candidat_keys) > 0 and iteration_count < 1000:
        spinner.next()
        for key in candidat_keys:
            if key not in candidat_change_count.keys():
                candidat_change_count[key] = {'cb_unchanged':0}
                
            neighbourhood = legal_concepts[key]['neighbours']
            
            neighbourhood_bow = dict()
            
            
            for neighbour in neighbourhood:
                neighbour_id = neighbour['neighbour']
                try:
                    neighbour_bow = legal_concepts[neighbour_id]['concept_bow']
                    for word in neighbour_bow.keys():
                        if word in legal_concepts[key]['concept_bow'].keys():
                            if word in neighbourhood_bow.keys():
                                neighbourhood_bow[word][0] = neighbourhood_bow[word][0] + neighbour_bow[word][0]
                                neighbourhood_bow[word][1] = neighbourhood_bow[word][1] + neighbour_bow[word][1]
                            else:
                                neighbourhood_bow[word] = copy.copy(neighbour_bow[word])
                        else:
                            continue
                            
                except:
                    if neighbour_id not in neighbours_not_found:
                        neighbours_not_found.append(neighbour_id)
            
            
            vec_change_sum = 0
            
            for word in neighbourhood_bow.keys():
                old_word_vec = legal_concepts[key]['concept_bow'][word][1]/legal_concepts[key]['concept_bow'][word][0]
                new_word_vec = (old_word_vec + (neighbourhood_bow[word][1]/neighbourhood_bow[word][0]))/2
                
                legal_concepts[key]['concept_bow'][word][1] = new_word_vec
                legal_concepts[key]['concept_bow'][word][0] = 1
                
                vec_change_sum += np.linalg.norm(old_word_vec-new_word_vec)
            
            if len(legal_concepts[key]['concept_bow']) > 0:
                if vec_change_sum/len(legal_concepts[key]['concept_bow']) < min_tf_threshold:
                    candidat_change_count[key]['cb_unchanged'] += 1
                    
                if candidat_change_count[key]['cb_unchanged'] == 2:
                    candidat_keys.remove(key)
            else:
                print(f"{key} empty BOW")
        
        iteration_count += 1
    
    if iteration_count < 1000:         
        print(f"{iteration_count} numbers of iteration where needed to calculate the concept bow.")
        if len(neighbours_not_found) > 0:
            print("---")
            print("The following neighbour ID's could not be found:")
            for not_found_id in neighbours_not_found:
                print(not_found_id)
        if len(neighbours_with_empty_bows) > 0:
            print("---")
            print("The following concepts have empty bow's':")    
            for empty_id in neighbours_with_empty_bows:
                print(empty_id)
    else:
        print("Stopped after 1000 iterations!")
    return legal_concepts


