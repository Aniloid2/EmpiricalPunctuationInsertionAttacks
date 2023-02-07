import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import sys
import os
def test_performance(lines):

    lines_of_interest = lines[-11:]
    lines_of_interest = [l.strip() for l in lines_of_interest]
    # lines_of_interest = [l[1:-1] for l in lines_of_interest ]
    # lines_of_interest = [l.replace("'", "") for l in lines_of_interest ]
    lines_of_interest = [l.split(': ') for l in lines_of_interest ]
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

        if 'Attack Success Rate' in l[0]:
            continue



        lines_of_interest_split.append([l[0], l[1]])

    return lines_of_interest_split


import argparse

parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('--dataset_1', default='MR',
                    help='First dataset')
parser.add_argument('--dataset_2', default='MR',
                    help='Second dataset')
parser.add_argument('--model_1', default='bert-base-uncased',
                    help='First dataset')
parser.add_argument('--model_2', default='bert-base-uncased',
                    help='Second dataset')
parser.add_argument('--p', default='0.2',
                    help='strenght of black box attack')
args = parser.parse_args()


folder_data_name_1 = args.dataset_1# 'MNLI'
folder_model_name_1 = args.model_1# 'bert-base-uncased'


folder_data_name_2 = args.dataset_2# 'MNLI'
folder_model_name_2 = args.model_2 #'distilbert-base-uncased'
#

folder_treie_1 = f'./Result/Non_Grammar/{folder_data_name_1}/{folder_model_name_1}'
file_name_treie_1 = f'./{folder_model_name_1}/{folder_data_name_1}/bert-base-uncased-{folder_data_name_1}-deepwordbug_'
file_name_textbugger_1 = f'./{folder_model_name_1}/{folder_data_name_1}/bert-base-uncased-{folder_data_name_1}-punctuation_attack_'

# folder_treie_2 = f'./Result/Non_Grammar/{folder_data_name_2}/{folder_model_name_2}'
# file_name_treie_2 = f'./{folder_model_name_2}/{folder_data_name_2}/bert-base-uncased-{folder_data_name_2}-deepwordbug_'
# file_name_textbugger_2 = f'./{folder_model_name_2}/{folder_data_name_2}/bert-base-uncased-{folder_data_name_2}-punctuation_attack_'

# open folder,  file and read these values

treie_results_1_main = os.listdir(folder_treie_1)
treie_results_1_main.sort()

# treie_results_2 = os.listdir(folder_treie_2)
# treie_results_2.sort()

def after_attack_and_words(folder,results,p):
    after_attack_d = ['a','b','c','d','e','f']
    after_attack_p = ['a','b','c','d','e','f']
    # print ('folder',folder,results)
    for i in range(len(results)):
        name = os.path.join(folder,results[i])
        f = open(name, "r")
        lines = f.readlines()
        cols_treie = test_performance(lines)
        if 'zeroe' in results[i] and f'{p}' in results[i]:
            # print ('resutls i',p,results[i],f'_\'_{p}' in results[i] )
            if f'_,_{p}' in results[i]:
                pos = 2
            elif f'_-_{p}' in results[i]:
                pos = 1
            elif f'_._{p}' in results[i]:
                pos = 3
            elif f'_\'_{p}' in results[i]:
                pos = 0
            elif f'_all_{p}' in results[i]:
                if args.dataset_1 == 'QQP':
                    pos = 5
                else:
                    pos = 4
                # after_attack_d.append('e')
            elif f'_char_{p}' in results[i]:
                if args.dataset_1 == 'QQP':
                    pos = 6
                else:
                    pos = 5
                # after_attack_d.append('f')
            elif f'_?_{p}' in results[i]:
                pos = 4
                after_attack_d.append('l')

            after_attack_d[pos] = [float(a[1]) for a in cols_treie if a[0] == 'After Attack Acc [%]'][0]
            # print ('add att d',after_attack_d,results[i], f'{p}' in results[i] )
        if 'zeroe' in results[i] and f'{p}' in results[i]:
            if f'_,_{p}' in results[i]:
                pos = 2
            elif f'_-_{p}' in results[i]:
                pos = 1
            elif f'_._{p}' in results[i]:
                pos = 3
            elif 'f_\'_{p}' in results[i]:
                pos = 0
            elif f'_all_{p}' in results[i]:
                pos = 4
                # after_attack_p.append('e')
            elif f'_char_{p}' in results[i]:
                pos = 5
                # after_attack_p.append('f')
            elif f'_?_{p}' in results[i]:
                pos = 6
                after_attack_p.append('l')
            after_attack_p[pos] = [float(a[1]) for a in cols_treie if a[0] == 'After Attack Acc [%]'][0]

    return {'attack_d':np.array(after_attack_d),'attack_p':np.array(after_attack_p)}


treie_results_1 = after_attack_and_words(folder_treie_1,treie_results_1_main,p=0.2)
treie_results_2 = after_attack_and_words(folder_treie_1,treie_results_1_main,p=0.5)
treie_results_3 = after_attack_and_words(folder_treie_1,treie_results_1_main,p=0.8)
# print ('treie results',treie_results_1)
# print ('treie results',treie_results_2)
# print ('treie results',treie_results_3)
# treie_results_2 = after_attack_and_words(folder_treie_2,treie_results_2)

def final_list(treie_results):
    fin_list = []
    for i in range(len(treie_results['attack_d'])):
        diff_col = []
        for j in range(len(treie_results['attack_p'])):
            # print (treie_results['attack_d'][i] , treie_results['attack_p'][j])
            diff = treie_results['attack_d'][i] - treie_results['attack_p'][j]
            diff_col.append(diff)
        fin_list.append(diff_col)
    return fin_list

fin_list_1 = final_list(treie_results_1)
fin_list_2 = final_list(treie_results_2)
fin_list_3 = final_list(treie_results_3)

# import pandas as pd
#
# df = pd.DataFrame(np.array(fin_list), columns=['\'','-'])
if args.dataset_1 == 'QQP':
    # x_axis_labels = ['Apostrophe','Hyphen','Comma','Full Stop','Question','Zeroe','Char']
    # y_axis_labels = ['Apostrophe','Hyphen','Comma','Full Stop','Question','Zeroe','Char']
    x_axis_labels = ['Ap','Hy','Co','FS','Qu','ZI','Ch' ]
    y_axis_labels = ['Ap','Hy','Co','FS','Qu','ZI','Ch' ] #['.',',','-','\'']
else:
    # x_axis_labels = ['\'','-',',','.']
    # y_axis_labels = ['\'','-',',','.'] #['.',',','-','\'']
    # x_axis_labels = ['Apostrophe','Hyphen','Comma','Full Stop','ZIroe','Char' ]
    # y_axis_labels = ['Apostrophe','Hyphen','Comma','Full Stop','ZIroe','Char' ] #['.',',','-','\'']
    x_axis_labels = ['Ap','Hy','Co','FS','ZI','Ch' ]
    y_axis_labels = ['Ap','Hy','Co','FS','ZI','Ch' ] #['.',',','-','\'']
import string
import matplotlib.pyplot as plt
# alphabet_string = string.ascii_lowercase
# alphabet_list = list(alphabet_string)
# fig = plt.figure(num=None, figsize=(10, 15), dpi=80, facecolor='w', edgecolor='k')
# fig = plt.figure(constrained_layout=True, figsize=(10, 4))
fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(10, 15),dpi=80, facecolor='w', edgecolor='k')

# fig.suptitle(f'PAA Performance Improvement \nWhen Using Punctuation \nInstead of Letters',fontsize =30)
# fig.suptitle(f'Black-Box PAA vs Zeroe (Intrusion) \nFor Different P Values ',fontsize =30)

# cmap = sns.diverging_palette(220, 20, as_cmap=True)
from matplotlib.colors import BoundaryNorm
a=np.random.randn(2500).reshape((50,50))
print (a.shape)
cmap = plt.get_cmap('RdBu') #PuOr #RdBu #bwr
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize and forcing 0 to be part of the colorbar!
# bounds = np.arange(np.min(a),np.max(a),.5)
bounds = np.arange(-20,20,.5)
idx=np.searchsorted(bounds,0)
bounds=np.insert(bounds,idx,0)
norm = BoundaryNorm(bounds, cmap.N)

if args.dataset_1 == 'QQP':
    corr= np.ones((7,7))
else:
    corr= np.ones((6,6))

matrix = np.tril(corr)

# sns.heatmap(ax= ax1, data= fin_list_1,norm=norm,cmap=cmap,mask=matrix, annot=True,linewidths=2,annot_kws={"size": 13},linecolor='black')
sns.heatmap(ax= ax1, data= fin_list_1,norm=norm,cmap=cmap,mask=matrix, annot=True,linewidths=0 ,annot_kws={"size": 25},linecolor='black')

ax1.set_xticklabels(x_axis_labels, fontsize = 20)
ax1.set_yticklabels(y_axis_labels, fontsize = 20, rotation=0)
folder_model_name_1 = folder_model_name_1.upper()
ax1.set_title(f'{folder_data_name_1}:{folder_model_name_1}', loc='center', fontsize =30)
# ax1.set_ylabel('Worst by')
ax1.set_xlabel('P = 0.2',fontsize = 25, weight='bold')
ax1.xaxis.set_tick_params(labeltop=True,labelbottom = False, bottom=False, left=True,right=False )
ax1.xaxis.set_label_position('top')
folder_model_name_1 = folder_model_name_1.lower()
ax1.figure.axes[-1].tick_params( labelsize=20)
# sns.heatmap(ax= ax2,data=fin_list_2,norm=norm,cmap=cmap,mask=matrix, annot=True,linewidths=2,annot_kws={"size": 13},linecolor='black')
sns.heatmap(ax= ax2,data=fin_list_2,norm=norm,cmap=cmap,mask=matrix, annot=True,linewidths=0 ,annot_kws={"size": 25},linecolor='black')

ax2.set_xticklabels(x_axis_labels, fontsize = 20)
ax2.set_yticklabels(y_axis_labels, fontsize = 20, rotation=0)
# ax2.set_ylabel('Worst by')
ax2.set_xlabel('P = 0.5',fontsize = 25, weight='bold')  #Better by
ax2.xaxis.set_tick_params(labeltop=True,labelbottom = False, bottom=False)
ax2.xaxis.set_label_position('top')
# folder_model_name_2 = folder_model_name_2.upper()
# ax2.set_title(f'{folder_data_name_2}:{folder_model_name_2}', loc='center', fontsize =20)
# folder_model_name_2 = folder_model_name_2.lower()
ax2.figure.axes[-1].tick_params( labelsize=20)


# sns.heatmap(ax= ax3,data=fin_list_3,norm=norm,cmap=cmap,mask=matrix, annot=True,linewidths=2,annot_kws={"size": 13},linecolor='black')
sns.heatmap(ax= ax3,data=fin_list_3,norm=norm,cmap=cmap,mask=matrix, annot=True,linewidths=0,annot_kws={"size": 25},linecolor='black')

ax3.set_xticklabels(x_axis_labels, fontsize = 20)
ax3.set_yticklabels(y_axis_labels, fontsize = 20, rotation=0)
# ax3.set_ylabel('Worst by')
ax3.set_xlabel('P = 0.8',fontsize = 25, weight='bold')#Better by
ax3.xaxis.set_tick_params(labeltop=True,labelbottom = False, bottom=False)
ax3.xaxis.set_label_position('top')
# folder_model_name_2 = folder_model_name_2.upper()
# ax3.set_title(f'{folder_data_name_2}:{folder_model_name_2}', loc='center', fontsize =17)
# folder_model_name_2 = folder_model_name_2.lower()

# Where X axis represents by how much the technique is better (red/positive) or worst (blue/negative) than the Y axis counterpart technique.
ax3.figure.axes[-1].tick_params( labelsize=20)

fig.tight_layout()
fig.savefig(f"./Result/Non_Grammar/{folder_data_name_1}/{folder_model_name_1}_{args.p}_heatmap_{args.dataset_1}.png" ,bbox_inches='tight')
fig.savefig(f"./Result/Non_Grammar/{folder_data_name_1}/{folder_model_name_1}_{args.p}_heatmap_{args.dataset_1}.eps" ,bbox_inches='tight')
