 # Import libraries
import matplotlib.pyplot as plt
import numpy as np

import sys
import pandas as pd
import  argparse

parser = argparse.ArgumentParser(description='transfer_test')
parser.add_argument('--type1',default='Result',
                    help='Folder_to_plot1')
parser.add_argument('--type2',default='Result',
                    help='Folder_to_plot2')

args = parser.parse_args()
# Creating dataset
# f = open('./internal_table_output.txt', "r")
# lines = f.readlines()
pd.set_option("display.max_columns", None)
data=pd.read_csv(f'./{args.type1}_table_output_pandas.csv', delimiter = ',',encoding='utf-8')
# data.set_index('id')
if args.type2 != 'None':
    data2=pd.read_csv(f'./{args.type2}_table_output_pandas.csv', delimiter = ',',encoding='utf-8')
    frames = [data,data2]
    data = pd.concat(frames,ignore_index=True)

print (data.tail(10))
# print (data['Method'])


def return_method_for_data(dataset):
    MR_subdata = data.loc[data['Dataset'] == dataset]
    MR_subdata = MR_subdata.astype({'Method': 'string'})
    print ('type method',MR_subdata.Method.dtypes)
    # print ('flatten',MR_subdata.Method.values.flatten())
    vals = list(pd.Series(MR_subdata.Method.values))
    res = []
    for D in vals:
        if 'DWBP' in D:
            res.append(True)
        else:
            res.append(False)
    # print (type(vals))
    # print(vals)
    # print (vals.str.contains("^DWBP")  )
    # .str.contains("^DWBP")

    DWBP = MR_subdata[res]#.loc[data['Method'] == '.*DWBP']
    # print ('dwbp',DWBP)
    DWB = MR_subdata.loc[data['Method'] == 'DWB']
    DWBP_acc = list(DWBP['After Attack Acc [%]'])
    DWB_acc = list(DWB['After Attack Acc [%]'])
    return [DWB_acc,DWBP_acc]
#
# MR_subdata = data.loc[data['Dataset'] == 'MR']
# MR_subdata = MR_subdata.astype({'Method': 'string'})
# print ('type method',MR_subdata.Method.dtypes)
# # print ('flatten',MR_subdata.Method.values.flatten())
# vals = list(pd.Series(MR_subdata.Method.values))
# res = []
# for D in vals:
#     if 'DWBP' in D:
#         res.append(True)
#     else:
#         res.append(False)
#
# DWBP = MR_subdata[res]#.loc[data['Method'] == '.*DWBP']
# DWB = MR_subdata.loc[data['Method'] == 'DWB']
# DWBP_acc = list(DWBP['After Attack Acc [%]'])
# DWB_acc = list(DWB['After Attack Acc [%]'])
#
# data = [DWB_acc,DWBP_acc]

datasets = ['MR','MNLI','SNLI','QNLI','QQP']
data_res = []

bloat = [ data_res + return_method_for_data(i) for i in datasets]

flatten_list = []

for i in bloat:
    for j in i:
        flatten_list.append(j)



data = flatten_list

fig = plt.figure(figsize =(10, 7))

# Creating plot
medianprops = dict(linestyle='-', linewidth=2, color='black')

# bplot = plt.boxplot(data, patch_artist=True,medianprops=medianprops)

bplot = plt.violinplot(data,showmeans=True)

for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = bplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(2)
        vp.set_alpha(1)

# x_axis_labels = ['','DWB\nMR','DWBP\nMR','DWB\nMNLI','DWBP\nMNLI','DWB\nSNLI','DWBP\nSNLI','DWB\nQNLI','DWBP\nQNLI','DWB\nQQP','DWBP\nQQP']
# x_axis_labels = ['','MR','MR','MNLI','MNLI','SNLI','SNLI','QNLI','QNLI','QQP','QQP']

x_axis_labels = ['MR','MNLI','SNLI','QNLI','QQP']


# plt.xticks([i for i in range(11)],x_axis_labels, fontsize = 20)
print ([i+0.5 for i in range(1,11,2)])
plt.xticks([i+0.5 for i in range(1,11,2)],x_axis_labels, fontsize = 20)

# ax1.set_yticklabels(y_axis_labels, fontsize = 20, rotation=0)

plt.xlabel('Attack/Dataset',fontsize=30)
# plt.xticks([i for i in range(0,11,1)],fontsize=25)
# plt.yticks([i for i in range(0,110,20)],fontsize=20)
plt.yticks(fontsize=25)
plt.ylabel('After Atk Acc [%]',fontsize=30)
# plt.title(f'{args.dataset} with Bert-Base-Uncased \n N of Synonyms/Punctuation \n vs \n After Success Rate [%] \n',fontsize=40)
# plt.legend(fontsize=30)
colors = ['lightskyblue', 'lightcoral']*5
for patch, color in zip(bplot['bodies'], colors):
        patch.set_facecolor(color)
from matplotlib.lines import Line2D
cmap = plt.cm.Pastel1
# custom_lines = [Line2D([0], [0], color=cmap(0.2), lw=12),
#                 Line2D([0], [0], color=cmap(0), lw=12)]
custom_lines = [Line2D([0], [0], color='lightskyblue', lw=12),
                Line2D([0], [0], color='lightcoral', lw=12)]


plt.legend(custom_lines, ['DWB', 'DWBP'],fontsize=20, loc='lower right')
[plt.axvline(x, color = 'black', linestyle='--') for x in [0.5,2.5,4.5,6.5,8.5]]


fig.tight_layout()


# show plot
plt.savefig(f'./{args.type1}_{args.type2}.png',bbox_inches='tight')
plt.savefig(f'./{args.type1}_{args.type2}.eps',bbox_inches='tight')

# fig.savefig(f"./Result/Non_Grammar/{folder_data_name_1}/{folder_model_name_1}_{args.p}_heatmap_{args.dataset_1}.png" ,bbox_inches='tight')
# fig.savefig(f"./Result/Non_Grammar/{folder_data_name_1}/{folder_model_name_1}_{args.p}_heatmap_{args.dataset_1}.eps" ,bbox_inches='tight')
