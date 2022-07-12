#%% import
import numpy as np

#%% WMD function
def wmd(input_bow, lc_bow):
    #max_input_bow_value = max(input_bow.values())
    #max_lc_bow_value = max(lc_bow.values())
    
    wmd = 0
    min_travel_distance_pairs = []
    
    reverse_wmd = 0
    reverse_min_travel_distance_pairs= []


    for word in input_bow.keys():
        input_word_vec = input_bow[word][1]/input_bow[word][0]
        word_min_travel_distance = 1000.000
        word_min_td_lc_word = ''
        
        for lc_word in lc_bow.keys():
            lc_word_vec = lc_bow[lc_word][1]/lc_bow[lc_word][0]
            current_travel_distance = np.linalg.norm(input_word_vec - lc_word_vec)
            
            if current_travel_distance < word_min_travel_distance:
                word_min_travel_distance = current_travel_distance
                word_min_td_lc_word = lc_word
                
        
        wmd += word_min_travel_distance
        min_travel_distance_pairs.append(
                (
                (word, 
                 [input_bow[word][0],
                  input_bow[word][1],
                  input_bow[word][2],
                  input_bow[word][3]]
                 ),
                (word_min_td_lc_word, 
                 [lc_bow[word_min_td_lc_word][0],
                  lc_bow[word_min_td_lc_word][1],
                  lc_bow[word_min_td_lc_word][2],
                  lc_bow[word_min_td_lc_word][3]]
                 ),
                word_min_travel_distance
                )
            )
    
    

    for lc_word in lc_bow.keys():
        lc_word_vec = lc_bow[lc_word][1]/lc_bow[lc_word][0]
        lc_word_min_travel_distance = 1000.000
        
        lc_word_min_td_word = ''
        

        
        for word in input_bow.keys():
            input_word_vec = input_bow[word][1]/input_bow[word][0]
            current_travel_distance = np.linalg.norm(input_word_vec - lc_word_vec)
            
            if current_travel_distance < lc_word_min_travel_distance:
               lc_word_min_travel_distance = current_travel_distance
               lc_word_min_td_word = word
               
        if lc_word_min_td_word != '':
            reverse_wmd += lc_word_min_travel_distance
            reverse_min_travel_distance_pairs.append(
                    (
                    (lc_word_min_td_word, 
                     [input_bow[lc_word_min_td_word][0],
                      input_bow[lc_word_min_td_word][1],
                      input_bow[lc_word_min_td_word][2],
                      input_bow[lc_word_min_td_word][3]]
                     ),
                    (lc_word, 
                     [lc_bow[lc_word][0],
                      lc_bow[lc_word][1],
                      lc_bow[lc_word][2],
                      lc_bow[lc_word][3]]
                     ),
                    lc_word_min_travel_distance
                    )
                )   
   
    return ({'wmd':wmd/len(input_bow.keys()), 
            'travel_distance_pairs': min_travel_distance_pairs
            },
            {'wmd':reverse_wmd/len(lc_bow.keys()), 
             'travel_distance_pairs': reverse_min_travel_distance_pairs
             })
        
            
    
    