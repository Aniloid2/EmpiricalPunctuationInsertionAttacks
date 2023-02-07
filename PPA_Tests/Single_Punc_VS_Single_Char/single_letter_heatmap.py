import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import sys
import os
import argparse
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

# folder_data_name = 'MR'
# folder_model_name = 'bert-base-uncased'

parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('--dataset_1', default='MR',
                    help='First dataset')
parser.add_argument('--dataset_2', default='MR',
                    help='Second dataset')
parser.add_argument('--model_1', default='lstm',
                    help='First dataset')
parser.add_argument('--model_2', default='bert-base-uncased',
                    help='Second dataset')
args = parser.parse_args()


folder_data_name = args.dataset_1# 'MNLI'
folder_model_name = args.model_1# 'bert-base-uncased'
# folder_data_name_2 = args.dataset_2# 'MNLI'
# folder_model_name_2 = args.model_2 #'distilbert-base-uncased'
#


# folder_data_name = 'MNLI'
# folder_model_name = 'bert-base-uncased'
folder_treie = f'./Result/Non_Grammar/{folder_data_name}/{folder_model_name}'
file_name_treie = f'./{folder_model_name}/{folder_data_name}/bert-base-uncased-{folder_data_name}-deepwordbug_'
file_name_textbugger = f'./{folder_model_name}/{folder_data_name}/bert-base-uncased-{folder_data_name}-punctuation_attack_'
# open folder,  file and read these values

treie_results = os.listdir(folder_treie)
treie_results.sort()

def after_attack_and_words(folder,results):
    after_attack_d = []
    after_attack_p = ['a','b','c','d']
    print ('folder',folder,results)
    for i in range(len(results)):
        name = os.path.join(folder,results[i])
        f = open(name, "r")
        lines = f.readlines()
        cols_treie = test_performance(lines)
        if 'deepwordbug' in results[i]:
            after_attack_d.append([float(a[1]) for a in cols_treie if a[0] == 'After Attack Acc [%]'][0])
        else:
            if '_,_symbol' in results[i]:
                pos = 2
            elif '_-_symbol' in results[i]:
                pos = 1
            elif '_._symbol' in results[i]:
                pos = 3
            elif '_\'_symbol' in results[i]:
                pos = 0
            elif '_?_symbol' in results[i]:
                pos = 4
                after_attack_p.append('e')
            after_attack_p[pos] = [float(a[1]) for a in cols_treie if a[0] == 'After Attack Acc [%]'][0]

    return {'attack_d':np.array(after_attack_d),'attack_p':np.array(after_attack_p)}


treie_results = after_attack_and_words(folder_treie,treie_results)
fin_list = []
for i in range(len(treie_results['attack_d'])):
    diff_col = []
    for j in range(len(treie_results['attack_p'])):
        diff = treie_results['attack_d'][i] - treie_results['attack_p'][j]
        diff_col.append(diff)
    fin_list.append(diff_col)

# import pandas as pd
#
# df = pd.DataFrame(np.array(fin_list), columns=['\'','-'])

if args.dataset_1 == 'QQP':
    x_axis_labels = ['Ap','Hy','Co','FS','Qu']
else:
    x_axis_labels = ['Apostrophe','Hyphen','Comma','Full Stop']
import string
import matplotlib.pyplot as plt
alphabet_string = string.ascii_lowercase
alphabet_list = list(alphabet_string)
fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')


# cmap = sns.diverging_palette(220, 20, as_cmap=True)
from matplotlib.colors import BoundaryNorm
a=np.random.randn(2500).reshape((50,50))
cmap = plt.get_cmap('RdBu') #PuOr #RdBu #bwr
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize and forcing 0 to be part of the colorbar!
# bounds = np.arange(np.min(a),np.max(a),.5)
bounds = np.arange(-5,5,.5)
idx=np.searchsorted(bounds,0)
bounds=np.insert(bounds,idx,0)
norm = BoundaryNorm(bounds, cmap.N)





fig = sns.heatmap(fin_list,norm=norm,cmap=cmap, annot=True,linewidths=2,annot_kws={"size": 20},linecolor='black')
fig.figure.axes[-1].tick_params( labelsize=20)

fig.set_xticklabels(x_axis_labels, fontsize = 25)
fig.set_yticklabels(alphabet_list, fontsize = 20)
folder_model_name = folder_model_name.upper()
# fig.set_title(f'PAA Performance Improvement \nWhen Using Punctuation \nInstead of Letters \n{folder_data_name}:{folder_model_name}', loc='center', fontsize =30)
fig.set_title(f'{folder_data_name}:{folder_model_name}', loc='center', fontsize =30)

folder_model_name = folder_model_name.lower()
fig = fig.get_figure()
fig.tight_layout()
fig.savefig(f"./Result/Non_Grammar/{folder_data_name}/{folder_model_name}_heatmap_{args.dataset_1}.png" ,bbox_inches='tight')
fig.savefig(f"./Result/Non_Grammar/{folder_data_name}/{folder_model_name}_heatmap_{args.dataset_1}.eps" ,bbox_inches='tight')
