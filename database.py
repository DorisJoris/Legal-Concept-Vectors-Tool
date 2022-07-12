# -*- coding: utf-8 -*-
#%% Import
import random
import numpy as np
import pandas as pd



from transformers import AutoTokenizer, AutoModelForPreTraining

import extractor_MAIN as lc_em
import vector_calculator as lc_vc
import text_cleaning as lc_text_cleaning
import wmd
import tf_idf as lc_tf_idf

from resources import hierachical_label_list

#%% Database class

class lc_database:
    def __init__(self, init_url):
        print('Loading BERT model')
        self.hierachical_label_list = hierachical_label_list
        self.tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
        self.model = AutoModelForPreTraining.from_pretrained("Maltehb/danish-bert-botxo")
        
        print('Creating legal concepts')
        init_lc_doc = lc_em.get_law_document_dict(init_url, 
                                                    self.tokenizer, 
                                                    self.model)
        
        
        
        #self.legal_concepts = init_lc_doc['legal_concepts']
        self.legal_concepts = lc_vc.concept_vector_init(init_lc_doc['legal_concepts'], 
                                                            self.hierachical_label_list)
        
        self.external_ref = init_lc_doc['external_ref']
    
    # Helper functions
    def concept_count(self):
        return len(self.legal_concepts)
    
    def random_lc(self):
        key = random.choice(list(self.legal_concepts.keys()))
        return self.legal_concepts[key]
    
    #Add new retsinfo documents
    def add_retsinfo_doc(self, url):
        to_be_added_doc = lc_em.get_law_document_dict(url, 
                                                        self.tokenizer, 
                                                        self.model)
        
        
        # to_be_added_doc = lc_vc.concept_vector_init(to_be_added_doc, 
        #                                                 self.hierachical_label_list, 
        #                                                 self.word_embeddings,
        #                                                 self.word_idf)
        
       
        
        #self.legal_concepts.update(to_be_added_doc['legal_concepts'])
        self.legal_concepts.update(lc_vc.concept_vector_init(to_be_added_doc['legal_concepts'], 
                                                            self.hierachical_label_list))
        
        for ref in to_be_added_doc['external_ref']:
            if ref not in self.external_ref:
                self.external_ref.append(ref)
                
    
    # Conncet the external refs
    def connect_ext_ref(self):
        for ex_ref in self.external_ref:
            ref_instance = ex_ref[1]
            if ref_instance['parent_law'] in self.legal_concepts.keys():
                
                referee_key = ex_ref[0]['name']
                for parent in ex_ref[0]['parent']:
                    referee_key = referee_key + '_' + parent
                
                ref_keys = []    
                for ref in ref_instance['p_ref_list']:
                    parents = ref['partial_parent']
                    parents.append(ref_instance['parent_law'])
                    
                    for ref_name in ref['ref_names']:
                        # remember "i"-problem -> if ref_name[-2] =='i':
                        for key in self.legal_concepts.keys():
                            if ref_name == self.legal_concepts[key]['name']:
                                suspect = True
                                for parent in parents:
                                    if parent not in self.legal_concepts[key]['parent']:
                                        suspect = False
                                        break
                                if suspect == True:
                                    ref_keys.append(key)
                 
                for ref_key in ref_keys:
                    self.legal_concepts[referee_key]['neighbours'].append(
                        {'neighbour':ref_key,'type':'ref_to'})
                    self.legal_concepts[ref_key]['neighbours'].append(
                        {'neighbour':referee_key,'type':'ref_from'})
                
                self.external_ref.remove(ex_ref)
    
    # def concept_bow_vector_init(self):
    #     self.legal_concepts = lc_vc.concept_vector_init(self.legal_concepts, 
    #                                                         self.hierachical_label_list)
    
    # Calculate the concept vectors and bows
    def calculate_concept_vector(self, aver_dist_threshold = 30):
        self.legal_concepts = lc_vc.concept_vector_calculator(self.legal_concepts, 
                                                                aver_dist_threshold)
        
    def calculate_concept_bow(self, min_tf_threshold = 30):
        
        self.legal_concepts = lc_vc.concept_bow_calculator(self.legal_concepts,
                                                           min_tf_threshold)
    
    # Calculate concept dfs
    def get_vector_dfs(self):
        columns = []
        for i in range(768):
            columns.append(f"Dim_{i+1}")
            
        
        self.concept_vector_df = pd.DataFrame()
        
        for key in self.legal_concepts.keys():
            if type(self.legal_concepts[key]['parent']) != list:
                level = 0
                parents = self.legal_concepts[key]['parent']
            else:
                level = len(self.legal_concepts[key]['parent'])
                parents = self.legal_concepts[key]['parent']
            
            key_cv_df = pd.DataFrame([self.legal_concepts[key]['concept_vector']], 
                                  index=[self.legal_concepts[key]['id']],
                                  columns=columns)
            key_cv_df['level'] = level
            key_cv_df['parent'] = '_'.join([str(item) for item in parents])
            
            self.concept_vector_df = self.concept_vector_df.append(key_cv_df)

    
    
        
    
    
 