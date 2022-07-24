# -*- coding: utf-8 -*-

#%% Import
import pickle

from pympler import asizeof

import database_bert as database

    
#%% App
if __name__ == "__main__":
    
    
    
    ## LBK-documents
    #funktionærloven
    url1 = 'https://www.retsinformation.dk/api/document/eli/lta/2017/1002'
    bert_database = database.lc_database(url1)
    
    
    
    
#%% Add documents
if __name__ == "__main__":    
    url_list = [
        'https://www.retsinformation.dk/api/document/eli/lta/2021/235', #barselsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2008/907', #lov om tidsbegrænset ansaettelse
        'https://www.retsinformation.dk/api/document/eli/lta/2022/336', #lov om investeringsforeninger m.v.
        'https://www.retsinformation.dk/api/document/eli/lta/2016/193', #Bekendtgørelse af lov om aftaler og andre retshandler på formuerettens område
        'https://www.retsinformation.dk/api/document/eli/lta/2013/1457', #Lov om forbrugeraftaler
        'https://www.retsinformation.dk/api/document/eli/lta/2021/1284',#Bekendtgørelse af lov om indkomstskat for personer m.v.
        'https://www.retsinformation.dk/api/document/eli/lta/2021/25', #Bekendtgørelse af lov om godkendte revisorer og revisionsvirksomheder (revisorloven)1)
        'https://www.retsinformation.dk/api/document/eli/lta/2010/240', #Ansættelsesbevisloven
        'https://www.retsinformation.dk/api/document/eli/lta/2021/242', #Afskrivningsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2018/1070', #Erstatningsansvarsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2022/956', #Erhvervsuddannelsesloven
        'https://www.retsinformation.dk/api/document/eli/lta/2021/824', #Kildeskatteloven
        'https://www.retsinformation.dk/api/document/eli/lta/2011/645', #Ligebehandlingsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2019/156', #Ligelønsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2022/866', #Markedsføringsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2021/1853', #Købeloven
        'https://www.retsinformation.dk/api/document/eli/lta/2014/332', #Kommissionsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2015/1123', #lov om forbrugerbeskyttelse ved erhvervelse af fast ejendom m.v.
        'https://www.retsinformation.dk/api/document/eli/lta/2021/510', #lov om formidling af fast ejendom m.v.
        'https://www.retsinformation.dk/api/document/eli/lta/2022/866', #Markedsføringsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2014/459', #Renteloven
        'https://www.retsinformation.dk/api/document/eli/lta/2021/242', #Afskrivningsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2015/47', #Boafgiftsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2019/132', #Ejendomsavancebeskatningsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2019/984', #Erhvervsfondsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2014/1175', #Etableringskontoloven
        'https://www.retsinformation.dk/api/document/eli/lta/2020/1590', #Ejendomsværdiskatteloven
        'https://www.retsinformation.dk/api/document/eli/lta/2020/2020', #Fondsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2021/743', #Fusionsskatteloven
        'https://www.retsinformation.dk/api/document/eli/lta/2014/433', #Forvaltningsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2015/48', #Opkrævnings- og inddrivelsesloven
        'https://www.retsinformation.dk/api/document/eli/lta/2019/353', #Konkursskatteloven
        'https://www.retsinformation.dk/api/document/eli/lta/2021/1735', #Ligningsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2013/349', #Ombudsmandsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2019/1121', #Retssikkerhedsloven
        'https://www.retsinformation.dk/api/document/eli/lta/2019/774', #Ægtefælleloven
        'https://www.retsinformation.dk/api/document/eli/lta/2021/251', #Selskabsskatteloven
        
        
        ]
    
    
    
    
    for url in url_list:
        bert_database.add_retsinfo_doc(url)

#%%
if __name__ == "__main__":
    url ='https://www.retsinformation.dk/api/document/eli/lta/2021/251'
    
    bert_database.add_retsinfo_doc(url)

#%% Save Bert database
if __name__ == "__main__":
   with open("databases/bert_database.p", "wb") as pickle_file:
       pickle.dump(bert_database, pickle_file)  


#%% Open pickled bert database

if __name__ == "__main__": 
    with open("databases/bert_database.p", "rb") as pickle_file:
        bert_database = pickle.load(pickle_file)      


#test_database.add_retsinfo_doc('https://www.retsinformation.dk/api/document/eli/lta/2021/25')
#%%
if __name__ == "__main__":
    
    bert_database.concept_bow_vector_init()
    
    #print(len(test_database.external_ref))
    #print(test_database.concept_count())
    bert_database.connect_ext_ref()  
    
    bert_database.calculate_concept_vector()
    bert_database.calculate_concept_bow()
    
    bert_database.get_vector_dfs()

    
    # cbowm_df = test_database.concept_bow_meanvector_df
    cv_df = bert_database.concept_vector_df
    # bm_df = test_database.bow_meanvector_df
    
    example_lc = bert_database.random_lc()
    example_lc = bert_database.legal_concepts['3_Stk. 1._§\xa02._Kapitel 2_Afsnit 1_LBK nr 336 af 11/03/2022']
    
    asizeof.asizeof(bert_database)/1e+9
    
    bert_database.number_of_doc()
    bert_database.concept_count()

    

    keys = list(bert_database.legal_concepts.keys())

    url = 'https://www.retsinformation.dk/api/document/eli/lta/2022/336'
    test_database = database.lc_database(url)
    example_lc = test_database.legal_concepts['a)_2)_Stk. 1._§\xa02._Kapitel 2_Afsnit 1_LBK nr 336 af 11/03/2022']
