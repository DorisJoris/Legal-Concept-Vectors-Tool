# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 19:36:17 2022

@author: bejob
"""
# Initial comment:
# The purpose of this code is to extract and segment the text of a danish law into its subparts,  
# such as paragraphs, stk (sections of a paragraph) and so on and organize these in a neo4j database. 
# The code has been writen for Funktionærloven, LBK nr 1002 af 24/08/2017, and will need to be
# modified to be used on other sources of law. 
    
#%% Import
import urllib.request
import json
import re
from datetime import datetime
from progress.bar import Bar
import numpy as np

import copy
import torch


from resources import lov_label_list
from resources import section_label_list
from resources import chapter_label_list
from resources import paragraph_label_list
from resources import stk_label_list
from resources import litra_label_list
from resources import nr_label_list
from resources import sentence_label_list

import extractor_LBK as LBK_extractor
import extractor_LOV as LOV_extractor
import reference_extractor as ref_extractor

#%% Get law json
def _get_law_json_doc_type(url):
    with urllib.request.urlopen(url) as page:
        data = json.loads(page.read().decode())
    
    lov_json = data[0]
    
    doc_type = lov_json['shortName'][0:lov_json['shortName'].find(' ')]
    
    return lov_json, doc_type

#%% Get law_document_dict
def concatenate_lists(list_of_lists):
    output_list = []
    for l in list_of_lists:
        output_list = output_list + l
    return output_list

def _get_legal_concept_id(name, parent_list):
    lc_id = name
    for parent in parent_list:
        lc_id = lc_id + '_' + parent
    return lc_id
    
def _get_sentence_bow_meanvector(sentence, tokenizer, model):
    start_embedding = datetime.now()
    
    tokenized = tokenizer(sentence, max_length=512, truncation=True, return_tensors='pt')
    
    tokens = tokenized['input_ids'][0]
    
    with torch.no_grad():
        output = model(**tokenized, output_hidden_states=True)
        
    # extract hidden staytes        
    hidden_states = torch.cat(output[2])
    
    # mean last layer to create sentence embedding
    sentence_embedding = torch.mean(hidden_states[-1:,:,:].squeeze(), dim=0).numpy()
    
    token_embeddings = list(hidden_states[-1:,:,:].squeeze().numpy())
    
    embedding_time = datetime.now() - start_embedding
    start_bow = datetime.now()
    
    words = list()
    
    for tensor, embedding in zip(tokens, token_embeddings):
        token_id = int(tensor)
        token = list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(token_id)]
        
        if re.match('[a-ø]',token[0]) != None:
            word = token
            
            vector = embedding 
            
            words.append([word,vector,[token_id]])
        
        elif token[0:2] == '##' and re.match('[a-ø]',token[2]) != None:
            
            if len(words) > 0:
                previous_tup = words[-1]
            else:
                print(sentence)
                print(token)
                previous_tup = ['', np.array([0]*768, dtype='float32'),[]]
            
            word = previous_tup[0] + token[2:]
            vector = (previous_tup[1] + embedding)/2
            
            words = words[:-1]
            words.append([word,vector,previous_tup[2]+[token_id]])
            
        else:
            continue
        
    bow = dict()
    for word_list in words:
        if word_list[0] in bow.keys():
            bow[word_list[0]][0] += 1
            bow[word_list[0]][1] = bow[word_list[0]][1]+word_list[1]
            bow[word_list[0]][3] += 1
            
        else:
            bow[word_list[0]] = [1, word_list[1], word_list[2], 1]
            
        
    bow_time = datetime.now() - start_bow
    
    return bow, sentence_embedding, (embedding_time, bow_time)

def _get_law_document_dict_raw(url):
    lov_json, doc_type = _get_law_json_doc_type(url)
    
    if doc_type == 'LBK':
        #Lov
        lov_soup, lov_property_dict, lov_name, lov_shortname = LBK_extractor.law_property_gen(lov_json)
        
        
        #Paragraphs
        (paragraph_property_list, chapter_property_list, section_property_list) = LBK_extractor.paragraph_property_gen(lov_soup, lov_shortname)
       
        
        # Stk   
        stk_property_list = LBK_extractor.stk_property_gen(paragraph_property_list)
        
        
        #Litra, Nr and sentences
        (litra_property_list,
        nr_property_list,
        sentence_property_list) = LBK_extractor.sentence_litra_nr_property_gen(stk_property_list)
        
    
    if doc_type == 'LOV':
        try:
            #Lov
            lov_soup, lov_property_dict, lov_name, lov_shortname = LOV_extractor.law_property_gen(lov_json)
            
            
            #Paragraphs
            (paragraph_property_list, chapter_property_list) = LOV_extractor.paragraph_property_gen(lov_soup, lov_shortname)
            
            
            # Stk   
            stk_property_list = LOV_extractor.stk_property_gen(paragraph_property_list)
            
            
            #Litra, Nr and sentences
            (litra_property_list,
            nr_property_list,
            sentence_property_list) = LOV_extractor.sentence_litra_nr_property_gen(stk_property_list)
            
            
        except:
            #Lov
            lov_soup, lov_property_dict, lov_name, lov_shortname = LBK_extractor.law_property_gen(lov_json)
            
            
            #Paragraphs
            (paragraph_property_list, chapter_property_list, section_property_list) = LBK_extractor.paragraph_property_gen(lov_soup, lov_shortname)
           
            
            # Stk   
            stk_property_list = LBK_extractor.stk_property_gen(paragraph_property_list)
            
            
            #Litra, Nr and sentences
            (litra_property_list,
            nr_property_list,
            sentence_property_list) = LBK_extractor.sentence_litra_nr_property_gen(stk_property_list)
            

    

    #----------------------------------------------------------------------------                                                             
    ## References

    stk_internal_ref_query_list, stk_internal_ref_dict_list = ref_extractor.stk_internal_references(sentence_property_list)
    
    
    paragraph_internal_references_query_list, paragraph_internal_ref_dict_list = ref_extractor.paragraph_internal_references(sentence_property_list)
    
    relative_stk_ref_sets_list = ref_extractor.get_relative_ref_sets(sentence_property_list) #only in funktionærloven
    relative_stk_ref_dict_list = ref_extractor.get_relative_stk_ref_dict_list(relative_stk_ref_sets_list)     
    
    list_internal_ref_query_list, list_internal_ref_dict_list = ref_extractor.list_internal_ref(sentence_property_list) #only in funktionærloven
    
    (law_internal_paragraph_specific_ref_query_list,
     law_internal_paragraph_specific_ref_dict_list,
     law_external_paragraph_specific_ref_list) = ref_extractor.paragraph_specific_references(sentence_property_list, paragraph_property_list)
    
    
    (internal_ref_to_whole_law_list, 
     law_external_paragraph_specific_ref_list) = ref_extractor.ref_to_whole_law(sentence_property_list, law_external_paragraph_specific_ref_list)
    internal_ref_to_whole_law_dict_list = ref_extractor.get_internal_ref_to_whole_law_dict_list(internal_ref_to_whole_law_list, lov_name, lov_shortname)

    
    
    # internal ref dict list
    internal_ref_dict_list = concatenate_lists([stk_internal_ref_dict_list,
                                                paragraph_internal_ref_dict_list,
                                                relative_stk_ref_dict_list,
                                                law_internal_paragraph_specific_ref_dict_list,
                                                internal_ref_to_whole_law_dict_list
                                                ])

                            
    #missing external references to document as a whole or chapters.
    
    law_document_dict_raw = {'law': lov_property_dict, 'law_label': lov_label_list,
                             'section': section_property_list, 'section_label': section_label_list,
                         'chapter': chapter_property_list, 'chapter_label': chapter_label_list,
                         'paragraph': paragraph_property_list, 'paragraph_label': paragraph_label_list,
                         'stk': stk_property_list, 'stk_label': stk_label_list,
                         'litra': litra_property_list, 'litra_label': litra_label_list,
                         'nr': nr_property_list, 'nr_label': nr_label_list,
                         'sentence': sentence_property_list, 'sentence_label': sentence_label_list,

                         'internal_ref': internal_ref_dict_list,
                         'external_ref': law_external_paragraph_specific_ref_list

                         }
    
    return law_document_dict_raw
    
def get_law_document_dict(url, tokenizer, model):
    law_document_dict_raw = _get_law_document_dict_raw(url)
    legal_concepts = {}

    bar_other = Bar('Processing other elements', max=7)
    # law
    law = law_document_dict_raw['law']
    law_id = _get_legal_concept_id(law['shortName'],[])
    new_law_property_dict = {
        'id': law_id,
        'name': law['name'],
        'shortName': law['shortName'],
        'title': law['title'],
        'date_of_publication': law['date_of_publication'],
        'parent': 'document',
        'ressort': law['ressort'],
        'retsinfo_id': law['id'],
        'url': url,
        'labels': law_document_dict_raw['law_label'],
        'raw_text': '',
        #'bow': dict(),
        #'bow_meanvector': None,#np.array([0]*word_embeddings.vector_size, dtype='float32'),
        'concept_bow': dict(),
        'concept_vector': None,#np.array([0]*word_embeddings.vector_size, dtype='float32'),
        'neighbours': []
        }
    
    legal_concepts[law_id] = new_law_property_dict
    bar_other.next()
    
    # section
    section_list = law_document_dict_raw['section']
    for section in section_list:
        section_id = _get_legal_concept_id(section['name'],section['parent'])
        new_section_property_dict = {
            'id': section_id,
            'name': section['name'],
            'shortName': section['shortName'],
            'position': section['position'],
            'parent': section['parent'],
            'labels': law_document_dict_raw['section_label'],
            'raw_text': '',
            #'bow': dict(),
            #'bow_meanvector': None,#np.array([0]*word_embeddings.vector_size, dtype='float32'),
            'concept_bow': dict(),
            'concept_vector': None#np.array([0]*word_embeddings.vector_size, dtype='float32')
            }
        
        parent_id = _get_legal_concept_id(section['parent'][0],section['parent'][1:])
        new_section_property_dict['neighbours'] = [{'neighbour':parent_id, 'type': 'parent'}]
        legal_concepts[parent_id]['neighbours'].append({'neighbour':section_id, 'type': 'child'})
        
        
        legal_concepts[section_id] = new_section_property_dict
    bar_other.next()
    # chapter
    chapter_list = law_document_dict_raw['chapter']
    for chapter in chapter_list:
        chapter_id = _get_legal_concept_id(chapter['name'],chapter['parent'])
        new_chapter_property_dict = {
            'id': chapter_id,
            'name': chapter['name'],
            'shortName': chapter['shortName'],
            'position': chapter['position'],
            'parent': chapter['parent'],
            'labels': law_document_dict_raw['chapter_label'],
            'raw_text': '',
            #'bow': dict(),
            #'bow_meanvector': None,#np.array([0]*word_embeddings.vector_size, dtype='float32'),
            'concept_bow': dict(),
            'concept_vector':None# np.array([0]*word_embeddings.vector_size, dtype='float32')
            }
        
        parent_id = _get_legal_concept_id(chapter['parent'][0],chapter['parent'][1:])
        new_chapter_property_dict['neighbours'] = [{'neighbour':parent_id, 'type': 'parent'}]
        legal_concepts[parent_id]['neighbours'].append({'neighbour':chapter_id, 'type': 'child'})
        
        
        legal_concepts[chapter_id] = new_chapter_property_dict
    bar_other.next()
    # paragraph
    paragraph_list = law_document_dict_raw['paragraph']
    for paragraph in paragraph_list:
        paragraph_id = _get_legal_concept_id(paragraph['name'],paragraph['parent'])
        new_paragraph_property_dict = {
            'id': paragraph_id,
            'name': paragraph['name'],
            'position': paragraph['position'],
            'parent': paragraph['parent'],
            'labels': law_document_dict_raw['paragraph_label'],
            'raw_text': '',
            #'bow': dict(),
            #'bow_meanvector': None,#np.array([0]*word_embeddings.vector_size, dtype='float32'),
            'concept_bow': dict(),
            'concept_vector': None#np.array([0]*word_embeddings.vector_size, dtype='float32')
            }
        
        parent_id = _get_legal_concept_id(paragraph['parent'][0],paragraph['parent'][1:])
        new_paragraph_property_dict['neighbours'] = [{'neighbour':parent_id, 'type': 'parent'}]
        legal_concepts[parent_id]['neighbours'].append({'neighbour':paragraph_id, 'type': 'child'})
        
        legal_concepts[paragraph_id] = new_paragraph_property_dict
    bar_other.next()
    # stk
    stk_list = law_document_dict_raw['stk']
    for stk in stk_list:
        stk_id = _get_legal_concept_id(stk['name'],stk['parent'])
        new_stk_property_dict = {
            'id': stk_id,
            'name': stk['name'],
            'position': stk['position'],
            'parent': stk['parent'],
            'labels': law_document_dict_raw['stk_label'],
            'raw_text': '',
            #'bow': dict(),
            #'bow_meanvector':None,# np.array([0]*word_embeddings.vector_size, dtype='float32'),
            'concept_bow': dict(),
            'concept_vector': None#np.array([0]*word_embeddings.vector_size, dtype='float32')
            }
        
        parent_id = _get_legal_concept_id(stk['parent'][0],stk['parent'][1:])
        new_stk_property_dict['neighbours'] = [{'neighbour':parent_id, 'type': 'parent'}]
        legal_concepts[parent_id]['neighbours'].append({'neighbour':stk_id, 'type': 'child'})
        
        legal_concepts[stk_id] = new_stk_property_dict
    bar_other.next()    
    # litra
    litra_list = law_document_dict_raw['litra']
    for litra in litra_list:
        litra_id = _get_legal_concept_id(litra['name'],litra['parent'])
        new_litra_property_dict = {
            'id': litra_id,
            'name': litra['name'],
            'position': litra['position'],
            'parent': litra['parent'],
            'labels': law_document_dict_raw['litra_label'],
            'raw_text': '',
            #'bow': dict(),
            #'bow_meanvector': None,#np.array([0]*word_embeddings.vector_size, dtype='float32'),
            'concept_bow': dict(),
            'concept_vector': None#np.array([0]*word_embeddings.vector_size, dtype='float32')
            }
        
        parent_id = _get_legal_concept_id(litra['parent'][0],litra['parent'][1:])
        new_litra_property_dict['neighbours'] = [{'neighbour':parent_id, 'type': 'parent'}]
        legal_concepts[parent_id]['neighbours'].append({'neighbour':litra_id, 'type': 'child'})
        
        legal_concepts[litra_id] = new_litra_property_dict
    bar_other.next()    
    # nr
    nr_list = law_document_dict_raw['nr']
    for nr in nr_list:
        nr_id = _get_legal_concept_id(nr['name'],nr['parent'])
        new_nr_property_dict = {
            'id': nr_id,
            'name': nr['name'],
            'position': nr['position'],
            'parent': nr['parent'],
            'labels': law_document_dict_raw['nr_label'],
            'raw_text': '',
            #'bow': dict(),
            #'bow_meanvector': None,#np.array([0]*word_embeddings.vector_size, dtype='float32'),
            'concept_bow': dict(),
            'concept_vector': None#np.array([0]*word_embeddings.vector_size, dtype='float32')
            }
        
        parent_id = _get_legal_concept_id(nr['parent'][0],nr['parent'][1:])
        new_nr_property_dict['neighbours'] = [{'neighbour':parent_id, 'type': 'parent'}]
        legal_concepts[parent_id]['neighbours'].append({'neighbour':nr_id, 'type': 'child'})
        
        legal_concepts[nr_id] = new_nr_property_dict
    bar_other.next()    
    bar_other.finish()
    # sentence
    
    
    sentence_list = law_document_dict_raw['sentence']
    
    bar_sentence = Bar('Processing sentences', max=len(sentence_list))
    embed_bow_times = list()
    
    
    for sentence in sentence_list:
        bow, meanvector, time_tup = _get_sentence_bow_meanvector(sentence['raw_text'], tokenizer, model)
        
        if len(bow) == 0:
            continue
        
        embed_bow_times.append(time_tup)
        
        
        sentence_id = _get_legal_concept_id(sentence['name'],sentence['parent'])
        new_sentence_property_dict = {
            'id': sentence_id,
            'name': sentence['name'],
            'position': sentence['position'],
            'parent': sentence['parent'],
            'raw_text': sentence['raw_text'],
            'labels': law_document_dict_raw['sentence_label'],
            #'bow': bow,
            'concept_bow': bow,
            #'bow_meanvector': "see concept_vector", #meanvector,
            'concept_vector': meanvector
            }
        
        
        
        parent_id = _get_legal_concept_id(sentence['parent'][0],sentence['parent'][1:])
        new_sentence_property_dict['neighbours'] = [{'neighbour':parent_id, 'type': 'parent'}]
        legal_concepts[parent_id]['neighbours'].append({'neighbour':sentence_id, 'type': 'child'})
        
        legal_concepts[sentence_id] = new_sentence_property_dict
        
        bar_sentence.next()
    bar_sentence.finish()
        

    for ref in law_document_dict_raw['internal_ref']:
         ref_from_id = _get_legal_concept_id(ref['ref_from']['name'], ref['ref_from']['parent'])
         try:
             ref_to_id = _get_legal_concept_id(ref['ref_to']['name'], ref['ref_to']['parent'])
         except:
             ref_to_id =  ref['ref_to']['shortName']
         try:
             legal_concepts[ref_to_id]['neighbours'].append({'neighbour':ref_from_id, 'type': 'ref_from'})
             legal_concepts[ref_from_id]['neighbours'].append({'neighbour':ref_to_id, 'type': 'ref_to'})
         except:
             legal_concepts[ref_from_id]['neighbours'].append({'neighbour':ref_to_id, 'type': 'ref_to_UNKNOWN'})
            
    law_doc_dict = {
        'legal_concepts': legal_concepts,
        'external_ref': law_document_dict_raw['external_ref']
        }
    
    return law_doc_dict
    
#%% test
if __name__ == "__main__":

    url = 'https://www.retsinformation.dk/api/document/eli/lta/2022/406'
    
    with urllib.request.urlopen(url) as page:
        data = json.loads(page.read().decode())
    
    lov_json = data[0]
    
    lov_soup, lov_property_dict, lov_name, lov_shortname = LBK_extractor.law_property_gen(lov_json)
    
    #Paragraphs
    (paragraph_property_list, chapter_property_list, section_property_list) = LBK_extractor.paragraph_property_gen(lov_soup, lov_shortname)
    
    # Stk   
    stk_property_list = LBK_extractor.stk_property_gen(paragraph_property_list)
    
    #Litra, Nr and sentences
    (litra_property_list,
    nr_property_list,
    sentence_property_list) = LBK_extractor.sentence_litra_nr_property_gen(stk_property_list)
    
    stk_internal_ref_query_list, stk_internal_ref_dict_list = ref_extractor.stk_internal_references(sentence_property_list)
        
    paragraph_internal_references_query_list, paragraph_internal_ref_dict_list = ref_extractor.paragraph_internal_references(sentence_property_list)

    relative_stk_ref_sets_list = ref_extractor.get_relative_ref_sets(sentence_property_list) #only in funktionærloven
    relative_stk_ref_dict_list = ref_extractor.get_relative_stk_ref_dict_list(relative_stk_ref_sets_list)
        
    list_internal_ref_query_list, list_internal_ref_dict_list = ref_extractor.list_internal_ref(sentence_property_list) #only in funktionærloven

    (law_internal_paragraph_specific_ref_query_list,
     law_internal_paragraph_specific_ref_dict_list,
     law_external_paragraph_specific_ref_list) = ref_extractor.paragraph_specific_references(sentence_property_list, paragraph_property_list)

    (internal_ref_to_whole_law_list, 
     law_external_paragraph_specific_ref_list) = ref_extractor.ref_to_whole_law(sentence_property_list, law_external_paragraph_specific_ref_list)
    internal_ref_to_whole_law_dict_list = ref_extractor.get_internal_ref_to_whole_law_dict_list(internal_ref_to_whole_law_list, lov_name, lov_shortname)
    
    # internal ref dict list
    internal_ref_dict_list = concatenate_lists([stk_internal_ref_dict_list,
                                                paragraph_internal_ref_dict_list,
                                                relative_stk_ref_dict_list,
                                                law_internal_paragraph_specific_ref_dict_list,
                                                internal_ref_to_whole_law_dict_list
                                                ])
                            
    #missing external references to document as a whole or chapters.
    
    law_document_dict_raw = {'law': lov_property_dict, 'law_label': lov_label_list,
                             'section': section_property_list, 'section_label': section_label_list,
                         'chapter': chapter_property_list, 'chapter_label': chapter_label_list,
                         'paragraph': paragraph_property_list, 'paragraph_label': paragraph_label_list,
                         'stk': stk_property_list, 'stk_label': stk_label_list,
                         'litra': litra_property_list, 'litra_label': litra_label_list,
                         'nr': nr_property_list, 'nr_label': nr_label_list,
                         'sentence': sentence_property_list, 'sentence_label': sentence_label_list,
                         'internal_ref': internal_ref_dict_list,
                         'external_ref': law_external_paragraph_specific_ref_list
                         }
