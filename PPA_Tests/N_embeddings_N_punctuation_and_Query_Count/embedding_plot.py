
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
import matplotlib.pyplot as plt
# import seaborn as sns

import os
import sys

import re

import argparse

name = 'PAA'
parser = argparse.ArgumentParser(description='plot')

parser.add_argument('--dataset',default="mr",
                    help='which dataset plotting')

parser.add_argument('--model',default="bert-base-uncased",
                    help='which dataset plotting')

parser.add_argument('--random',default=False,action='store_true',
                    help='random position picking for pucntuation insertion')


args = parser.parse_args()

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):

    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
def test_performance(lines):

    lines_of_interest = lines[-11:]
    lines_of_interest = [l.strip() for l in lines_of_interest]
    lines_of_interest = [l[1:-1] for l in lines_of_interest ]
    lines_of_interest = [l.replace("'", "") for l in lines_of_interest ]
    lines_of_interest = [l.split(', ') for l in lines_of_interest ]
    lines_of_interest_split = []

    for l in lines_of_interest:
        if "%" in l[1]:
            l[1] = l[1][:-1]
        else:
            pass


        if 'Number Of Successful Attacks' in l[0]:
            continue

        if 'Number Of Failed Attacks' in l[0]:
            continue

        if 'Number Of Skipped Attacks' in l[0]:
            continue
        #
        # if 'Attack Success Rate' in l[0]:
        #     continue


        lines_of_interest_split.append([l[0], l[1]])

    return lines_of_interest_split

sample_number = 10

word_budget = [i for i in range(1,sample_number+1,1)]


if args.dataset == 'mr':
    args.dataset = 'MR'
    dataset = 'rotten_tomatoes'
elif args.dataset =='mnli':
    args.dataset = 'MNLI'
    dataset = 'mnli'

if args.random == True:
    random_extension = '_random_false'
else:
    random_extension = ''

folder_treie = f'./punctuation_attack{random_extension}/{dataset}/{args.model}'
folder_textbugger = f'./textfooler/{dataset}/{args.model}'
folder_random_false = f'./punctuation_attack_random_false/{dataset}/{args.model}'
folder_all_char_false = f'./punctuation_attack_all_char_false/{dataset}/{args.model}'
folder_all_false = f'./punctuation_attack_all_false/{dataset}/{args.model}'


file_name_treie = 'punctuation_attack{random_extension}_embedding_size_'
file_name_textbugger = 'textfooler_embedding_size_'
file_name_random_false = f'punctuation_attack_random_false_embedding_size_'
file_name_all_char_false = f'punctuation_attack_all_char_false_embedding_size_'
file_name_all_false = f'punctuation_attack_all_false_embedding_size_'


# open folder,  file and read these values
textbugger_results = os.listdir(folder_textbugger)
textbugger_results.remove('Models')
textbugger_results.sort(key=natural_keys)
treie_results = os.listdir(folder_treie)
treie_results.remove('Models')
treie_results.sort(key=natural_keys)
random_false_results = os.listdir(folder_random_false)
random_false_results.remove('Models')
random_false_results.sort(key=natural_keys)
all_char_false_results = os.listdir(folder_all_char_false)
all_char_false_results.remove('Models')
all_char_false_results.sort(key=natural_keys)
all_false_results = os.listdir(folder_all_false)
all_false_results.remove('Models')
all_false_results.sort(key=natural_keys)

print (textbugger_results, treie_results, random_false_results,all_char_false_results,all_false_results)

treie_results = treie_results[:sample_number]
textbugger_results = textbugger_results[:sample_number]
random_false_results = random_false_results[:sample_number]
all_char_false_results = all_char_false_results[:sample_number]
all_false_results = all_false_results[:sample_number]
# #temp
# treie_results = textbugger_results
# folder_treie = folder_textbugger
####

def after_attack_and_words(folder,results):
    query_number = []
    after_attack = []
    for i in range(len(results)):
        name = os.path.join(folder,results[i])
        f = open(name, "r")
        lines = f.readlines()
        cols_treie = test_performance(lines)
        print (cols_treie)
        query_number.append([float(a[1]) for a in cols_treie if a[0] == 'Avg Number Queries:'][0])
        after_attack.append([float(a[1]) for a in cols_treie if a[0] == 'Attack Success Rate [%]:'][0])

    return {'attack':after_attack,'query':query_number}


# for i in range(2):#len(textbugger_results)):
#     name = os.path.join(folder_textbugger,textbugger_results[i])
#     f = open(name, "r")
#     lines = f.readlines()
#     cols_textbugger = test_performance(lines)
#     print (cols_textbugger)
#
# treie_actual_words = []
# treie_after_attack = []
# for i in range(2):#len(treie_results)):
#     name = os.path.join(folder_treie,treie_results[i])
#     f = open(name, "r")
#     lines = f.readlines()
#     cols_treie = test_performance(lines)
#     print (cols_treie)
#     treie_actual_words.append([a[1] for a in cols_treie if a[0] == 'Number Changed Words'][0])
#     treie_after_attack.append([a[1] for a in cols_treie if a[0] == 'After Attack Acc [%]'][0])

# print (treie_actual_words,treie_after_attack)

treie_results = after_attack_and_words(folder_treie,treie_results)
textbugger_results = after_attack_and_words(folder_textbugger,textbugger_results)
random_false_results = after_attack_and_words(folder_random_false,random_false_results)
all_char_false_results = after_attack_and_words(folder_all_char_false,all_char_false_results)
all_false_results = after_attack_and_words(folder_all_false,all_false_results)

# print (treie_results,textbugger_results)
#
# sys.exit()
# treie_actual_words =[0, 1, 1.37, 1.77, 2.01, 2.29, 2.42, 2.67, 2.87, 3.06, 3.2]
# treie_actual_words = [a[1] for a in cols_treie if a[0] == 'Number Changed Words']
# treie_after_attack = [83.8, 65.8, 55.4, 45.8, 42.4, 39.4, 37, 34.6, 31.8, 30.2, 28.2]
# treie_after_attack = [a[1] for a in cols_treie if a[0] == 'After Attack Acc [%]']
# print (treie_actual_words,treie_after_attack)
# sys.exit()
#
# insert_actual_words = [0, 1, 1.38, 1.58, 1.76, 1.76, 1.85, 1.89, 1.93, 1.93, 1.93]
# isnert_after_attack =[83.8, 68.6, 61.8, 58.2, 55.2, 56.6, 55.2, 55.4, 56.2, 56.2, 56.2]

print (treie_results, textbugger_results,random_false_results,all_char_false_results,all_false_results)

plt.figure(figsize=(10,6))
# plt.plot(word_budget, treie_results['attack'], color='r',marker='o', lw=2, label=f'{name}')
txtf = plt.plot(word_budget, textbugger_results['attack'], color='b',marker='>',linestyle='dashed', lw=2, label=f'TextFooler',alpha=0.5,markersize=15)



dwbpTF=plt.plot(word_budget, treie_results['attack'], color='r',marker='o', lw=2, label=f'DWBP RPos=T/RPunc=F',alpha=0.5,markersize=15)
# for i, txt in enumerate(treie_results['query']):
#     txt = str(int(txt))
#     plt.annotate(txt, (word_budget[i], treie_results['attack'][i]),fontsize=20) #color='b',marker='>', lw=2, label=f'TextFooler',alpha=0.5
# for i, txt in enumerate(textbugger_results['query']):
#     txt = str(int(txt))
#     plt.annotate(txt, (word_budget[i], textbugger_results['attack'][i]),fontsize=20)
dwbpFF = plt.plot(word_budget, random_false_results['attack'], color='g',marker='s', lw=2, label=f'DWBP RPos=F/RPunc=F',alpha=0.5,markersize=15)


dwbpTT = plt.plot(word_budget, all_char_false_results['attack'], color='m',marker='P', lw=2, label=f'DWBP RPos=T/RPunc=T',alpha=0.5,markersize=15)

dwbpFT = plt.plot(word_budget, all_false_results['attack'], color='k',marker='X', lw=2, label=f'DWBP RPos=F/RPunc=T',alpha=0.5,markersize=15)



# no_skill = len(all_scores['original_label'][all_scores['original_label']==1]) / len(all_scores['original_label'])
# plt.plot([0, 1], [no_skill, no_skill], color='navy', lw=2, linestyle='--', label='No Skill')
plt.xlabel('N Embeddings/Characters',fontsize=30)
plt.xticks([i for i in range(0,11,1)],fontsize=25)
plt.yticks([i for i in range(10,110,20)],fontsize=20)
plt.yticks(fontsize=25)
plt.ylabel('Atk Succ Rate [%]',fontsize=30)
# plt.title(f'{args.dataset} with Bert-Base-Uncased \n N of Synonyms/Punctuation \n vs \n After Success Rate [%] \n',fontsize=40)
plt.legend(fontsize=18)


# line_columns = [
#                 p1a, p2a,
#                 (p1a, p1b), (p2a, p2b),
#                 (p1a, p1c), (p2a, p2c)
#                 ]
#
# plt.legend(line_columns, ['']*len(line_columns),
#          title='RPos=T RPos=F',
#          ncol=3, numpoints=1, handletextpad=-0.5)


plt.savefig(f'./synonym_embeddings_{name}_{args.dataset}.png',bbox_inches='tight')
plt.savefig(f'./synonym_embeddings_{name}_{args.dataset}.eps',bbox_inches='tight')
