# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 18:07:25 2022

@author: bejob
"""
#%% import
import pickle
import numpy as np
import pandas as pd
import random
import copy
import re
import json

from sklearn.neighbors import NearestNeighbors

from datetime import datetime

import text_cleaning as lc_text_cleaning
import wmd


#%% input convertion
def calculate_input_bow_meanvector(input_dict, database):
    input_dict['input_bow'] = dict()
    input_dict['input_cv'] = np.array([0]*768, dtype='float32')
    #input_dict['input_oov_list'] = []
    
    for sentence_dict in input_dict['sentence_dicts']:
        input_dict['input_cv'] += sentence_dict['input_cv']
        
        for word in sentence_dict['input_bow'].keys():
            if word in input_dict['input_bow'].keys():
                input_dict['input_bow'][word][0] +=  sentence_dict['input_bow'][word][0]
                
                
                input_dict['input_bow'][word][1] +=  sentence_dict['input_bow'][word][1]
                
                
                input_dict['input_bow'][word][3] +=  sentence_dict['input_bow'][word][3]
            else:
                input_dict['input_bow'][word] = copy.copy(sentence_dict['input_bow'][word])
                
                
    input_dict['input_cv'] = input_dict['input_cv']/len(input_dict['sentence_dicts']) 
    
    
    return input_dict

#%% 
def calculate_min_dist(input_dict, database, nbrs_cv):
    
    bow_type = ('wmd_concept_bow','concept_bow', 'reverse_wmd_concept_bow')
    
    max_level = database.concept_vector_df['level'].max()
    
    search_vector = input_dict['input_cv']
    
    cv_distances, cv_neighbours = nbrs_cv.kneighbors(search_vector.reshape(1,-1))
    
    
    input_dict['input_min_dist'] = {
        'concept_vector':dict(),
        'wmd_concept_bow':{
            '1. closest': ('',{'wmd':1000.00}),
            '2. closest': ('',{'wmd':1000.00}),
            '3. closest': ('',{'wmd':1000.00})
            },
        'reverse_wmd_concept_bow':{
            '1. closest': ('',{'wmd':1000.00}),
            '2. closest': ('',{'wmd':1000.00}),
            '3. closest': ('',{'wmd':1000.00})
            },
        }
    
    for i in range(10):
        key = f"{i+1}. closest"
        input_dict['input_min_dist']['concept_vector'][key] = (database.concept_vector_df.iloc[cv_neighbours[0][i]].name,
                                                               cv_distances[0][i])
    

    changed = False
    for level in range(max_level+1):
        search_df = database.concept_vector_df[database.concept_vector_df.level == level]
        if input_dict['input_min_dist'][bow_type[0]]['1. closest'][0] != '':
             search_parent_one = input_dict['input_min_dist'][bow_type[0]]['1. closest'][0]
             search_parent_two = input_dict['input_min_dist'][bow_type[0]]['2. closest'][0]
             search_parent_three = input_dict['input_min_dist'][bow_type[0]]['3. closest'][0]
             search_df = search_df[(search_df.parent == search_parent_one) | 
                                   (search_df.parent == search_parent_two) |
                                   (search_df.parent == search_parent_three)]
        changed = False    
        for key in search_df.index.values.tolist():
        
            #try:
                wmd_dict = wmd.wmd(input_dict['input_bow'], 
                                                 database.legal_concepts[key][bow_type[1]])
                
                
                if wmd_dict[0]['wmd'] <= input_dict['input_min_dist'][bow_type[0]]['1. closest'][1]['wmd']:
                    input_dict['input_min_dist'][bow_type[0]]['3. closest'] = input_dict['input_min_dist'][bow_type[0]]['2. closest']
                    input_dict['input_min_dist'][bow_type[0]]['2. closest'] = input_dict['input_min_dist'][bow_type[0]]['1. closest']
                    input_dict['input_min_dist'][bow_type[0]]['1. closest'] = (
                        str(key),
                        wmd_dict[0]
                        )
                    changed = True
                    
                elif wmd_dict[0]['wmd'] <=  input_dict['input_min_dist'][bow_type[0]]['2. closest'][1]['wmd']:  
                    input_dict['input_min_dist'][bow_type[0]]['3. closest'] = input_dict['input_min_dist'][bow_type[0]]['2. closest']
                    input_dict['input_min_dist'][bow_type[0]]['2. closest'] = (
                        str(key),
                        wmd_dict[0]
                        )
                    changed = True
                    
                elif wmd_dict[0]['wmd'] <=  input_dict['input_min_dist'][bow_type[0]]['3. closest'][1]['wmd']:
                    input_dict['input_min_dist'][bow_type[0]]['3. closest'] =  (
                        str(key),
                        wmd_dict[0]
                        )
                    changed = True
                
                if wmd_dict[1]['wmd'] <= input_dict['input_min_dist'][bow_type[2]]['1. closest'][1]['wmd']:
                    input_dict['input_min_dist'][bow_type[2]]['3. closest'] = input_dict['input_min_dist'][bow_type[2]]['2. closest']
                    input_dict['input_min_dist'][bow_type[2]]['2. closest'] = input_dict['input_min_dist'][bow_type[2]]['1. closest']
                    input_dict['input_min_dist'][bow_type[2]]['1. closest'] = (
                        str(key),
                        wmd_dict[1]
                        )
                    changed = True
                    
                elif wmd_dict[1]['wmd'] <=  input_dict['input_min_dist'][bow_type[2]]['2. closest'][1]['wmd']:  
                    input_dict['input_min_dist'][bow_type[2]]['3. closest'] = input_dict['input_min_dist'][bow_type[2]]['2. closest']
                    input_dict['input_min_dist'][bow_type[2]]['2. closest'] = (
                        str(key),
                        wmd_dict[1]
                        )
                    changed = True
                    
                elif wmd_dict[1]['wmd'] <=  input_dict['input_min_dist'][bow_type[2]]['3. closest'][1]['wmd']:
                    input_dict['input_min_dist'][bow_type[2]]['3. closest'] =  (
                        str(key),
                        wmd_dict[1]
                        )
                    changed = True
                    
            # except:
            #     pass
            
        if changed == False:
            break
                        
            
    return input_dict

#%% 2d distance preserving plotter
def two_d_distance_preserving_plotter(list_of_points, list_of_distances):
    """
    Parameters
    ----------
    list_of_points : list of numpy array
        Should be a list of numpy arrays of equal lenght.
        The first numpy array will be the focus point and the others will be the neighbours.
    
    list_of_distances : list of int
        Should be a list of distances between the points and the focus point.

    Returns
    -------
    None.

    """
    dim = len(list_of_points[0])
    
    new_focus_point = np.array([0,0])
    new_neighbour_points = list()
    
    for a in range(1,len(list_of_points)):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        new_neighbour_point = np.array([x,y])
        

        goal_dist = (list_of_distances[a]/dim)*2
            
        start_dist = np.linalg.norm(new_focus_point-new_neighbour_point)
        
        new_x = new_neighbour_point[0]+(((new_focus_point[0]-new_neighbour_point[0])/start_dist)*((start_dist-goal_dist))) 
        new_y = new_neighbour_point[1]+(((new_focus_point[1]-new_neighbour_point[1])/start_dist)*((start_dist-goal_dist))) 
        new_neighbour_point = np.array([new_x,new_y])
        
        new_neighbour_points.append(new_neighbour_point)
    
        
    
    
    for k in range(4):    
        for i in range(1,len(list_of_points)):
            focus_neighbour_point = list_of_points[i]
            new_focus_neighbour_point = new_neighbour_points[i-1]
            
            for j in range(1,len(list_of_points)):
                if i == j:
                    continue
                else:
                    neighbour_point = list_of_points[j]
                    
                    goal_dist = (np.linalg.norm(focus_neighbour_point-neighbour_point)/dim)*2
                    start_dist = np.linalg.norm(new_focus_neighbour_point-new_neighbour_points[j-1])
                    
                    if start_dist != 0:
                        new_x = new_neighbour_points[j-1][0]+(((new_focus_neighbour_point[0]-new_neighbour_points[j-1][0])/start_dist)*((start_dist-goal_dist))) 
                        new_y = new_neighbour_points[j-1][1]+(((new_focus_neighbour_point[1]-new_neighbour_points[j-1][1])/start_dist)*((start_dist-goal_dist))) 
                        new_neighbour_points[j-1] = np.array([new_x,new_y])
                    else:
                        new_x = new_neighbour_points[j-1][0]+(((new_focus_neighbour_point[0]-new_neighbour_points[j-1][0]))*((start_dist-goal_dist))) 
                        new_y = new_neighbour_points[j-1][1]+(((new_focus_neighbour_point[1]-new_neighbour_points[j-1][1]))*((start_dist-goal_dist))) 
                        new_neighbour_points[j-1] = np.array([new_x,new_y])
                    
                    
            for g in range(1,len(list_of_points)):
                
                goal_dist = (list_of_distances[g]/dim)*2
                    
                start_dist = np.linalg.norm(new_focus_point-new_neighbour_points[g-1])
                
                new_x = new_neighbour_points[g-1][0]+(((new_focus_point[0]-new_neighbour_points[g-1][0])/start_dist)*((start_dist-goal_dist))) 
                new_y = new_neighbour_points[g-1][1]+(((new_focus_point[1]-new_neighbour_points[g-1][1])/start_dist)*((start_dist-goal_dist))) 
                new_neighbour_points[g-1] = np.array([new_x,new_y])
            
        
    
    return [new_focus_point]+new_neighbour_points

#%% Create visualization dataframe

def get_visual_df(input_dict, database):
    distance_labels = ['concept_vector',
                       'reverse_wmd_concept_bow',
                       'wmd_concept_bow']
    
    visual_dfs = dict()
    
    zero_vector = np.array([0]*768, dtype='float32')
    for label in distance_labels:
        lc_names = ['Input text','Zero vector']
        list_of_points = [input_dict['input_cv'],
                          zero_vector]
        
        list_of_types = ['Input','Zero']
        list_of_types_two = ['Text', 'Vector']
        list_of_raw_text = [input_dict['full_text'],'']
        list_of_word_pair_ranks = [0,0]
        list_of_bow_size = [len(input_dict['input_bow'].keys()),0]
        
        list_of_distances = [0,
                             np.linalg.norm(input_dict['input_cv']-zero_vector)]
        
        for key in input_dict['input_min_dist'][label].keys():
            tup = input_dict['input_min_dist'][label][key]
            
            lc_names.append(tup[0].replace('§ ', '§\xa0'))
            if label == 'concept_vector':
                list_of_points.append(database.legal_concepts[tup[0].replace('§ ', '§\xa0')]['concept_vector'])
                list_of_bow_size.append(len(database.legal_concepts[tup[0].replace('§ ', '§\xa0')]['concept_bow'].keys()))
            else:
                list_of_points.append(database.legal_concepts[tup[0].replace('§ ', '§\xa0')]['concept_vector'])
    
                list_of_bow_size.append(len(database.legal_concepts[tup[0].replace('§ ', '§\xa0')]['concept_bow'].keys()))
                
            list_of_types.append(key)
            list_of_types_two.append('Legal concept')
            raw_text = database.legal_concepts[tup[0].replace('§ ', '§\xa0')]['raw_text']
            raw_text = re.sub('  ', ' ', raw_text)
            list_of_raw_text.append(raw_text)
            list_of_word_pair_ranks.append(0)
                        
            if type(tup[1]) != dict:
                list_of_distances.append(tup[1])
            else:
                list_of_distances.append(tup[1]['wmd'])
                traveldistances = sorted(input_dict['input_min_dist'][label][key][1]['travel_distance_pairs'], key=lambda tup: tup[2])
                rank = 1
                for traveldistance in traveldistances[0:10]:
                    if traveldistance[0][0] not in lc_names:
                        lc_names.append(traveldistance[0][0])
                        vector = traveldistance[0][1][1]
                        list_of_points.append(vector)
                        list_of_types.append('Input')
                        list_of_types_two.append('word')
                        list_of_raw_text.append(traveldistance[0][0])
                        list_of_word_pair_ranks.append(rank)
                        dist = np.linalg.norm(input_dict['input_cv']-vector)
                        list_of_distances.append(dist)
                        list_of_bow_size.append(1)
                    if traveldistance[1][0] not in lc_names:
                        lc_names.append(traveldistance[1][0])
                        vector = traveldistance[1][1][1]
                        list_of_points.append(vector)
                        list_of_types.append(key)
                        list_of_types_two.append('word')
                        list_of_raw_text.append(traveldistance[1][0])
                        list_of_word_pair_ranks.append(rank)
                        dist = np.linalg.norm(input_dict['input_cv']-vector)
                        list_of_distances.append(dist)
                        list_of_bow_size.append(1)
                    rank += 1
                for traveldistance in traveldistances[-10:]:
                    if traveldistance[0][0] not in lc_names:
                        lc_names.append(traveldistance[0][0])
                        vector = traveldistance[0][1][1]
                        list_of_points.append(vector)
                        list_of_types.append('Input')
                        list_of_types_two.append('word')
                        list_of_raw_text.append(traveldistance[0][0])
                        list_of_word_pair_ranks.append(rank)
                        dist = np.linalg.norm(input_dict['input_cv']-vector)
                        list_of_distances.append(dist)
                        list_of_bow_size.append(1)
                    if traveldistance[1][0] not in lc_names:
                        lc_names.append(traveldistance[1][0])
                        vector = traveldistance[0][1][1]
                        list_of_points.append(vector)
                        list_of_types.append(key)
                        list_of_types_two.append('word')
                        list_of_raw_text.append(traveldistance[1][0])
                        list_of_word_pair_ranks.append(rank)
                        dist = np.linalg.norm(input_dict['input_cv']-vector)
                        list_of_distances.append(dist)
                        list_of_bow_size.append(1)
                    rank += 1
        
        
        xy_df = pd.DataFrame(two_d_distance_preserving_plotter(list_of_points, list_of_distances),
                             columns = ['X','Y'])
        label_df = pd.DataFrame()
        label_df['Name'] = lc_names
        label_df['Neighbour type'] = list_of_types
        label_df['Point type'] = list_of_types_two
        label_df['Text'] = list_of_raw_text
        label_df['wordpair rank'] = list_of_word_pair_ranks
        label_df['Distance to input'] = list_of_distances
        label_df['BoW size'] = list_of_bow_size
        label_df = pd.concat([label_df, xy_df], axis =1)
        
        visual_dfs[label] = label_df
    return visual_dfs


#%% Get input sentence dict
def get_input_sentence_dicts(text, database):
    start=datetime.now()
    
    nbrs_cv = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(database.concept_vector_df.iloc[:,0:768])

      
    input_dict = dict()
    input_dict['full_text'] = text
    input_sentences = lc_text_cleaning.split_text_into_sentences(text)

    input_dict['sentence_dicts'] = []
    for sentence in input_sentences:
        sentence_dict = lc_text_cleaning.get_sentence_bow_meanvector(sentence,
                                                                      database)
        
        if len(sentence_dict['input_bow'].keys())>0:
            sentence_dict = calculate_min_dist(sentence_dict, database, nbrs_cv)
            
            input_dict['sentence_dicts'].append(sentence_dict)
    
    input_dict = calculate_input_bow_meanvector(input_dict, database)
    
    input_dict = calculate_min_dist(input_dict, database, nbrs_cv)
    
    visual_dfs = [get_visual_df(input_dict, database)]
    
    for sentence_dict in input_dict['sentence_dicts']:
        visual_dfs.append(get_visual_df(sentence_dict, database))
    
    print(datetime.now()-start)
    return visual_dfs, input_dict

#%% serialize

def _convert_array_to_list_in_input_dict(input_dict):
    input_dict['input_cv'] = input_dict['input_cv'].tolist()
    
    for word in input_dict['input_bow'].keys():
        input_dict['input_bow'][word][1] = input_dict['input_bow'][word][1].tolist()
    
    for distance in ['reverse_wmd_concept_bow','wmd_concept_bow']:
        for neighbour in input_dict['input_min_dist'][distance]:
            input_dict['input_min_dist'][distance][neighbour][1]['wmd'] = float(input_dict['input_min_dist'][distance][neighbour][1]['wmd'])
            new_travel_distance_pairs = list()
            for travel_dist_pair in input_dict['input_min_dist'][distance][neighbour][1]['travel_distance_pairs']:
                
                travel_dist_pair[0][1][1] = travel_dist_pair[0][1][1].tolist()
                travel_dist_pair[1][1][1] = travel_dist_pair[1][1][1].tolist()
                
                new_travel_distance_pairs.append((travel_dist_pair[0], travel_dist_pair[1], float(travel_dist_pair[2])))
            
            input_dict['input_min_dist'][distance][neighbour][1]['travel_distance_pairs'] = new_travel_distance_pairs
    
    for neighbour in input_dict['input_min_dist']['concept_vector']:
        input_dict['input_min_dist']['concept_vector'][neighbour] = (input_dict['input_min_dist']['concept_vector'][neighbour][0], 
                                                                     float(input_dict['input_min_dist']['concept_vector'][neighbour][1]))
                
    return input_dict

#%%
def serialize_input_visual_tup(visual_dfs_list, input_dict):
    
    for visual_dfs in visual_dfs_list:
        for key in visual_dfs.keys():
            visual_dfs[key] = visual_dfs[key].to_json(date_format='iso', orient='split')
        
    input_dict = _convert_array_to_list_in_input_dict(input_dict)
    
    for sentence_dict in input_dict['sentence_dicts']:
        sentence_dict = _convert_array_to_list_in_input_dict(sentence_dict)
        
    return visual_dfs_list, input_dict
    
#%% 
def get_inputs_visual_dfs(input_list, database):
    input_visual_dfs = dict()
    for input_tup in input_list:
        visual_dfs, input_dict = get_input_sentence_dicts(input_tup[0], database)
        input_visual_dfs[input_tup[1]] = (serialize_input_visual_tup(visual_dfs, input_dict))
        
    return input_visual_dfs

#%%
def get_input_visual_dfs(text, database):
    visual_dfs, input_dict = get_input_sentence_dicts(text, database)
    visual_dfs, input_dict = serialize_input_visual_tup(visual_dfs, input_dict)
    return json.dumps(visual_dfs), json.dumps(input_dict)

#%% 
def add_inputs_visual_dfs(input_text, input_visual_dfs, database):
    nr = 1
    input_id = f"User Input {nr}"
    while input_id in input_visual_dfs.keys():
        nr += 1
        input_id = f"User Input {nr}"
    
    visual_dfs, input_dict = get_input_sentence_dicts(input_text, database)
    input_visual_dfs[input_id] = (serialize_input_visual_tup(visual_dfs, input_dict))
        
    return input_visual_dfs



#%% Open database
if __name__ == "__main__":
    #open data
    with open("databases/test_database.p", "rb") as pickle_file:
        test_database = pickle.load(pickle_file) 
        
    example_lc = test_database.random_lc()



#%% Input text
    test_input_list = [("En funktionær er en lønmodtager, som primært er ansat "
                       +"inden for handel- og kontorområdet. " 
                       +"Du kan også være funktionær, hvis du udfører "
                       +"lagerarbejde eller tekniske og kliniske ydelser.",
                       "En funktionær er en lønmodtager... (Funktionærloven)"),#Funktionærloven -> https://www.frie.dk/find-svar/ansaettelsesvilkaar/funktionaerloven/#:~:text=En%20funktion%C3%A6r%20er%20en%20l%C3%B8nmodtager,arbejdstid%20er%20minimum%208%20timer.
                       
                       ("Der er en lang række fleksible muligheder - specielt for de forældre, "
                       +"som gerne vil vende tilbage til arbejdet efter for eksempel seks eller "
                       +"otte måneders orlov og gemme resten af orloven, til barnet er blevet lidt ældre. "
                       +"Eller for de forældre, der ønsker at dele orloven eller starte på arbejde på "
                       +"nedsat tid og dermed forlænge orloven. Fleksibiliteten forudsætter i de "
                       +"fleste tilfælde, at der er indgået en aftale med arbejdsgiveren.", 
                       "Der er en lang række fleksible... (Barselsloven)"),#Barselsloven -> https://bm.dk/arbejdsomraader/arbejdsvilkaar/barselsorlov/barselsorloven-hvor-meget-orlov-og-hvordan-kan-den-holdes/
                       
                       ("Når du er tidsbegrænset ansat, gælder der et princip om, at du ikke "
                       +"må forskelsbehandles i forhold til virksomhedens fastansatte, "
                       +"medmindre forskelsbehandlingen er begrundet i objektive forhold. "
                       +"Du har altså de samme rettigheder som de fastansatte.", 
                       "Når du er tidsbegrænset ansat... (Lov om tidsbegrænset ansættelse"),#Tidsbegrænset ansættelse -> https://sl.dk/faa-svar/ansaettelse/tidsbegraenset-ansaettelse
                       
                       ("Person A er bogholder.", 
                        "Person A er bogholder."),
                       
                       ("Bogholderen Anja venter sit første barn. Hendes termin nærmer sig med hastige skridt.",
                        "Bogholderen Anja venter første barn..."),
                       
                       ("Jan's kone er gravid. Han glæder sig meget til at være hjemmegående og bruge tid med hans søn.",
                        "Jan's kone er gravid..."),
                       
                       ("Den nye malersvend blev fyret efter en uge.",
                        "Den nye malersvend blev fyret efter en uge."),
                       
                       ("Den nye salgschef blev fyret efter en uge.", 
                        "Den nye salgschef blev fyret efter en uge.")
                       ]   





#%% create input_visual_dfs

    #test_visual_dfs, test_input_dict = get_input_sentence_dicts(test_input_list[0][0], test_input_list[0][1], test_database)
    
    input_visual_dfs = get_inputs_visual_dfs(test_input_list[:2], test_database)
    
    input_visual_dfs = (get_input_visual_dfs(test_input_list[0][0], test_database))
    input_visual_dfs = (get_input_sentence_dicts(test_input_list[0][0],test_database))

#%% json serialization

    
    input_visual_jsons_json = json.dumps(input_visual_dfs)

#%% Save input_visual_dfs

    with open("visualization_data/input_visual_dfs_new.p", "wb") as pickle_file:
        pickle.dump(input_visual_dfs, pickle_file) 
    
    #with open("visualization_data/word_idf.p", "wb") as pickle_file:
    #    pickle.dump(test_database.word_idf, pickle_file) 
    
    #with open("visualization_data/stopwords_list.p", "wb") as pickle_file:
    #    pickle.dump(test_database.stopwords, pickle_file) 

#%% Open pickled input visual_dfs

    with open("visualization_data/input_visual_dfs.p", "rb") as pickle_file:
        input_visual_dfs = pickle.load(pickle_file) 




#%% latex export

    
    wmd_bow_1_td_pairs_df = pd.DataFrame(sorted(input_dict['input_min_dist']['wmd_bow']['1. closest'][1]['travel_distance_pairs'], key=lambda tup: tup[2]),
                                         columns=['Input word','Legal concept word','Distance'])

    print(wmd_bow_1_td_pairs_df.to_latex(index=False,
                                         caption=('Travel distance pairs of the 1. closets legal concept BOW for the WMD.'),
                                         label='wmd_bow_1_td_pais',
                                         float_format="%.4f"))   

    test_latex = input_visual_dfs['En funktionær er en lønmodtager... (Funktionærloven)'][0]['concept_bow_meanvector'].to_latex(index=False,
                                    caption=(''),
                                    label='tab:add label',
                                    float_format="%.4f")
    test_latex_list = test_latex.split("\n")
    
    
    