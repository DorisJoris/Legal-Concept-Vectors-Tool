# -*- coding: utf-8 -*-

#%% Import
import pickle

from pympler import asizeof

import database

    
#%% App
if __name__ == "__main__":
    
    
    
    ## LBK-documents
    #funktionærloven
    url1 = 'https://www.retsinformation.dk/api/document/eli/lta/2017/1002'
    test_database = database.lc_database(url1)
    
    #lov om finansiel virksomhed (Ram explosion)
    #url4 = 'https://www.retsinformation.dk/api/document/eli/lta/2022/406'
    #test_database.add_retsinfo_doc(url4)
    
    
    
#%% Add documents
if __name__ == "__main__":    
    url_list = [
         'https://www.retsinformation.dk/api/document/eli/lta/2021/235', #barselsloven
         'https://www.retsinformation.dk/api/document/eli/lta/2008/907', #lov om tidsbegrænset ansaettelse
         'https://www.retsinformation.dk/api/document/eli/lta/2022/336', #lov om investeringsforeninger m.v.
         'https://www.retsinformation.dk/api/document/eli/lta/2016/193', #Bekendtgørelse af lov om aftaler og andre retshandler på formuerettens område
         'https://www.retsinformation.dk/api/document/eli/lta/2013/1457', #Lov om forbrugeraftaler
        'https://www.retsinformation.dk/api/document/eli/lta/2021/1284',#Bekendtgørelse af lov om indkomstskat for personer m.v.
        'https://www.retsinformation.dk/api/document/eli/lta/2021/25' #Bekendtgørelse af lov om godkendte revisorer og revisionsvirksomheder (revisorloven)1)
        ]
    
    
    
    
    for url in url_list:
        test_database.add_retsinfo_doc(url)
  
#test_database.add_retsinfo_doc('https://www.retsinformation.dk/api/document/eli/lta/2021/25')
#%%
if __name__ == "__main__":
    
    #test_database.concept_bow_vector_init()
    
    #print(len(test_database.external_ref))
    #print(test_database.concept_count())
    test_database.connect_ext_ref()  
    
    test_database.calculate_concept_vector(aver_dist_threshold = 0.1)
    test_database.calculate_concept_bow(min_tf_threshold = 0.1)
    
    test_database.get_vector_dfs()
    
    # cbowm_df = test_database.concept_bow_meanvector_df
    # cv_df = test_database.concept_vector_df
    # bm_df = test_database.bow_meanvector_df
    
    example_lc = test_database.random_lc()
    example_lc = test_database.legal_concepts['3. pkt._Stk. 3._§\xa019._Kapitel 3_LBK nr 25 af 08/01/2021']
    
    asizeof.asizeof(test_database)/1e+9
    
    with open("databases/test_database.p", "wb") as pickle_file:
        pickle.dump(test_database, pickle_file) 




    


    


    