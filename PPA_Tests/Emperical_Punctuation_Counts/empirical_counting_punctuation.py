# import transformers
# from textattack.models.wrappers import HuggingFaceModelWrapper
# from textattack.attack_results import FailedAttackResult, SkippedAttackResult
import sys
from textattack.datasets import HuggingFaceDataset

import numpy as np
# import random
import torch
# from textattack import Attacker
import argparse
import pandas as pd
pd.set_option("display.max_columns", None)
from tabulate import tabulate

# import os
# https://huggingface.co/textattack

# def set_seed(random_seed):
#     random.seed(random_seed)
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
# set_seed(765)

def punctuation_dictionarization(list_punc,punc_dic):
   for j, (p) in enumerate(list_punc):
       # print ('jp',j,p)
       if p in punc_dic:
           punc_dic[p] +=1
       else:
           punc_dic[p] = 1
   return punc_dic

def percentage_dic_fun(punc_dic,total_number_letters):
   total_number_punctuation = 0
   for symbol,numb in punc_dic.items():
       total_number_punctuation += numb
   # print ('total numb punc',total_number_punctuation, 'total numb char',total_number_letters)

   total_chars =total_number_punctuation + total_number_letters
   percentage_dic = {}
   for symbol,numb in punc_dic.items():
       percentage_dic[symbol] = (numb/total_chars)*100

   return percentage_dic

def sorted_dics(punc_dic,top=10):

   sorted_punc_dic = {k: v for k, v in sorted(punc_dic.items(), key=lambda item: item[1], reverse=True)[:top]}
   return sorted_punc_dic



def concatinate_strings(example):
   str = []
   for i in example:
       str.append(example[i])
   return ' '.join(str)

def brian_tokenizer_sanitizer(example_concat):
    # example_concat = '_..?_sdlf__dsf___d__s__ never_mind ..actu.aly'
    split_sentance = example_concat.split()
    tokenized_sentance = []

    for i in split_sentance:
        # cleaned = re.sub('[^a-zA-Z0-9_.-]|(?<!\d)\.(?!\d)|(?<!\w)-(?!\w)','',i)

        cleaned = re.sub(r'^[^a-zA-Z0-9À-ÿ]+|[^a-zA-Z0-9À-ÿ]+$', '', i)
        # cleaned = re.sub(r'[a-zA-Z]([^]*)[a-zA-Z]','', i)
        # cleaned = re.findall('\w+(?:\W+\w+)*',i)

        tokenized_sentance.append(cleaned)
    # print ('str:',split_sentance)
    return tokenized_sentance

parser = argparse.ArgumentParser(description='punctuation count')


parser.add_argument('--global_dataset',default="rotten_tomatoes",
                    help='attacked dataset')
parser.add_argument('--max_num_samples',default=2000,
                    help='number of samples to check for number of punctuation')

args = parser.parse_args()
# from textattack import Attack
# from textattack.search_methods import GreedySearch,GreedyWordSwapWIR

# from textattack.constraints.grammaticality import PartOfSpeech
# from textattack.constraints.pre_transformation import (
#     InputColumnModification,
#     RepeatModification,
#     StopwordModification,
# )
# from textattack.goal_functions import UntargetedClassification
# from textattack.constraints.pre_transformation import RepeatModification
# from textattack.constraints.pre_transformation import StopwordModification
# from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
# from textattack.constraints.semantics import WordEmbeddingDistance

#
# from textattack.transformations import (
#     CompositeTransformation,
#     WordSwapEmbedding,
#     WordSwapHomoglyphSwap,
#     WordSwapNeighboringCharacterSwap,
#     WordSwapRandomCharacterDeletion,
#     WordSwapRandomCharacterInsertion,
#     WordSwapSpecialSymbolInsertion,
#     WordSwapSpecialSymbolSubstitution,
#     WordSwapSampleSpecialSymbolInsertion,
#     WordSwapSampleSpecialSymbolInsertionExtended,
#
# )

# python empirical_counting_punctuation.py --global_dataset "rotten_tomatoes"
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
# need to figure out how to get all the datasets
dataset_list = ['MR','IMDB','MNLI','SNLI','QNLI','QQP']

dataset_final =[]
punctuation_final =[]
punctuation_count_final = []
punctuation_perc_final =[]

for d in dataset_list:

    args.global_dataset=d

    if args.global_dataset == 'MR':
        dataset = HuggingFaceDataset('rotten_tomatoes', None, "train",shuffle=True)
    elif args.global_dataset == 'QNLI':
        dataset = HuggingFaceDataset('glue', 'qnli', "train",shuffle=True)
    elif args.global_dataset == 'QQP':
        dataset = HuggingFaceDataset('glue', 'qqp', "train",shuffle=True)
    elif args.global_dataset == 'IMDB':
        dataset = HuggingFaceDataset( 'imdb',None, "train",shuffle=True)
    elif args.global_dataset == 'SNLI':
        dataset = HuggingFaceDataset( 'snli', None, "train", None, None,shuffle=True)
    elif args.global_dataset == 'MNLI':
        dataset = HuggingFaceDataset("glue",'mnli',"train",None,None,shuffle=True)


    punc_dic = {}
    total_number_letters = 0
    for i,(example, label) in enumerate(dataset):

        if i == args.max_num_samples:
            break
        example_concat = concatinate_strings(example)
        clean_letters =  re.sub(r'[^a-zA-Z0-9À-ÿ\t\n\r\f]','',example_concat)

        total_number_letters += len(clean_letters)
        text_under_analysis =  re.sub(r'[a-zA-Z0-9À-ÿ\t\n\r\f]','',example_concat)
        text_under_analysis = "".join(text_under_analysis.split())
        if len(text_under_analysis) > 0:
            list_punctuation = list(text_under_analysis)
        else:
            continue

        punc_dic = punctuation_dictionarization(list_punctuation,punc_dic)



    percentage_dic = percentage_dic_fun(punc_dic,total_number_letters)
    sorted_dic = sorted_dics(punc_dic,top=10)
    sorted_percentage_dic = sorted_dics(percentage_dic,top=10)



    punc_middle_word = {}
    for i,(example, label) in enumerate(dataset):
        if i == args.max_num_samples:
            break

        example_concat = concatinate_strings(example)
        tokenized_input = brian_tokenizer_sanitizer(example_concat) # mine tokenizer

        sentence_with_no_outer_punc = "".join(tokenized_input)
        remaining_punc = re.sub(r'[a-zA-Z0-9À-ÿ\t\n\r\f]','',sentence_with_no_outer_punc)
        remaining_punc = "".join(remaining_punc.split())
        if len(remaining_punc) > 0:
            list_punctuation_mid = list(remaining_punc)
        else:
            continue
        punc_middle_word = punctuation_dictionarization(list_punctuation_mid,punc_middle_word)


    percentage_mid_dic = percentage_dic_fun(punc_middle_word,total_number_letters)
    sorted_mid_dic = sorted_dics(punc_middle_word,top=5)
    sorted_percentage_mid_dic = sorted_dics(percentage_mid_dic,top=5)

    # put into pandas
    # do first table for normal dic
    # do second table for sec dic

    dataset = ['Total punctuation count'] + [args.global_dataset for i in range(len(list(sorted_dic.keys())))]
    punctuation =  ['Total punctuation count'] + list(sorted_dic.keys())
    punctuation_count =  ['Total punctuation count'] + list(sorted_dic.values())
    punctuation_perc =  ['Total punctuation count'] + ["{:.2E}%".format(i) for i in list(sorted_percentage_dic.values())]

    #add total punc count to all

    dataset_mid = ['Punctuation within word count'] + [args.global_dataset for i in range(len(list(sorted_mid_dic.keys())))]
    punctuation_mid =  ['Punctuation within word count'] + list(sorted_mid_dic.keys())
    punctuation_count_mid =  ['Punctuation within word count'] + list(sorted_mid_dic.values())
    punctuation_perc_mid =  ['Punctuation within word count'] + ["{:.2E}%".format(i) for i in list(sorted_percentage_mid_dic.values())]

    dataset_final = dataset_final+dataset+dataset_mid
    punctuation_final = punctuation_final+ punctuation + punctuation_mid
    punctuation_count_final = punctuation_count_final+ punctuation_count + punctuation_count_mid
    punctuation_perc_final = punctuation_perc_final + punctuation_perc + punctuation_perc_mid

data = {'Dataset':dataset_final,'Punctuation':punctuation_final,'Counts':punctuation_count_final,'Percentage':punctuation_perc_final}


print(tabulate(data, tablefmt="plain",headers=[ 'Dataset','Punctuation','Counts','Percentage']))
