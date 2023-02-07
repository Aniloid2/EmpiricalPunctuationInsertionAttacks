import os
import argparse
import re
parser = argparse.ArgumentParser(description='transfer_test')
parser.add_argument('--folder_name',default='Result',
                    help='Folder_to_plot')


args = parser.parse_args()

# test 1 treie vs insert
# folder = 'Treie'
# tests = ["bert-base-uncased-mr_treie.txt", "roberta-base-mr_treie.txt", "xlnet-base-cased-mr_treie.txt", "bert-base-uncased-mr_insert.txt", "roberta-base-mr_insert.txt", "xlnet-base-cased-mr_insert.txt", "bert-base-uncased-imdb_treie.txt", "roberta-base-imdb_treie.txt", "xlnet-base-cased-imdb_treie.txt", "bert-base-uncased-imdb_insert.txt", "roberta-base-imdb_insert.txt", "xlnet-base-cased-imdb_insert.txt"]

#test 2 treie + char vs textbugger no words
# folder = 'Treie_Char'
# tests = ["roberta-base-mr_textbugger_no_words.txt" ,"bert-base-uncased-mr_textbugger_no_words.txt","xlnet-base-cased-mr_textbugger_no_words.txt","roberta-base-imdb_textbugger_no_words.txt","bert-base-uncased-imdb_textbugger_no_words.txt","xlnet-base-cased-imdb_textbugger_no_words.txt","roberta-base-mr_treie_textbugger_no_words.txt","bert-base-uncased-mr_treie_textbugger_no_words.txt","xlnet-base-cased-mr_treie_textbugger_no_words.txt","roberta-base-imdb_treie_textbugger_no_words.txt","bert-base-uncased-imdb_treie_textbugger_no_words.txt","xlnet-base-cased-imdb_treie_textbugger_no_words.txt",]

# test 3 treie + char + words vs textbugger


if args.folder_name ==  'Result':
    folder = './Result'
    tests = os.listdir(folder)


try:
    tests.remove('tfhub_modules')
except:
    pass


def test_performance(lines):

    lines_of_interest = lines[-11:]
    lines_of_interest = [l.strip() for l in lines_of_interest]
    lines_of_interest_split = []
    for l in lines_of_interest:
        if '[' or ']' in l:
            # l = re.sub(r'[\]\[\',]','',l)
            l = re.sub(r'\[\'','',l)
            l = re.sub(r'\'\]','',l)
            l = re.sub(r'[\',]','',l)
            # l = re.sub(r'\[','',l)
        column_value_split = l.split(': ')
        if '[' or ']' in column_value_split[1]:
            column_value_split[1] = re.sub(r'[\]\[\',]','',column_value_split[1] )
        if "%" in column_value_split[1]:
            column_value_split[1] = column_value_split[1][:-1]
        else:
            pass

        if 'Number Of Successful Attacks' in l:
            continue

        # if 'Perturbed Words' in l:
        #     continue

        if 'Number Changed Words' in l:
            continue

        if 'Words Per Input' in l:
            continue

        # if 'Semantic Sim' in l:
        #     continue

        # if 'Avg Number Queries' in l:
        #     continue

        if 'Number Of Failed Attacks' in l:
            continue

        if 'Number Of Skipped Attacks' in l:
            continue

        if 'Attack Success Rate' in l:
            continue

        print ('cols',column_value_split[0], float(column_value_split[1]))

        # column_value_split = l.split(':')
        lines_of_interest_split.append([column_value_split[0], float(column_value_split[1])])

    return lines_of_interest_split

def order_columns(cols):
    cols = [cols[0],cols[1],cols[2],cols[3],cols[4],cols[6],cols[5],cols[7],cols[8]]#,cols[5],cols[6],cols[9]]
    return cols



def model_name(test):
    name_model = ['Model']
    if 'lstm' in test:
        name_model.append('LSTM')
    elif 'roberta' in test:
        name_model.append('RoBERTa')
    elif 'cnn' in test:
        name_model.append('CNN')
    elif 'distilbert' in test:
        name_model.append('DistilBERT')
    elif 'bert' in test:
        name_model.append('BERT')
    elif 'xlnet' in test:
        name_model.append('XLNet')
    return [name_model]

def dataset_name(test):
    name_dataset = ['Dataset']
    if 'MR' in test:
        name_dataset.append('MR')
    elif 'IMDB' in test:
        name_dataset.append('IMDB')
    elif 'SNLI' in test:
        name_dataset.append('SNLI')
    elif 'MNLI' in  test:
        name_dataset.append('MNLI')
    elif 'AG-News' in test:
        name_dataset.append('AG-News')
    elif 'QNLI' in test:
        name_dataset.append('QNLI')
    elif 'QQP' in test:
        name_dataset.append('QQP')
    return [name_dataset]



def name_method(folder,test):
    name_method = ['Method']

    if folder =='./Result':

        if 'punctuation_attack' in test:
            name_method.append('PAA')
        elif 'textfooler_paa' in test:
            name_method.append('TextFooler/PAA')
        elif 'pso_paa' in test:
            name_method.append('SememePSO/PAA')
        elif 'pso' in test:
            name_method.append('SememePSO')
        elif 'textfooler' in test:
            name_method.append('TextFooler')

    return [name_method]



from collections import OrderedDict


if folder ==  './Result':
    Ordered_table = OrderedDict([
            ('MR',OrderedDict([
                ('CNN',OrderedDict([ ('PAA',[]),('TextFooler',[]),('TextFooler/PAA',[]),('SememePSO',[]),('SememePSO/PAA',[])])),
                ('LSTM',OrderedDict([ ('PAA',[]),('TextFooler',[]),('TextFooler/PAA',[]),('SememePSO',[]),('SememePSO/PAA',[])])),
                ('BERT',OrderedDict([ ('PAA',[]),('TextFooler',[]),('TextFooler/PAA',[]),('SememePSO',[]),('SememePSO/PAA',[])])),
                ('RoBERTa',OrderedDict([ ('PAA',[]),('TextFooler',[]),('TextFooler/PAA',[]),('SememePSO',[]),('SememePSO/PAA',[])])),
                ('XLNet',OrderedDict([ ('PAA',[]),('TextFooler',[]),('TextFooler/PAA',[]),('SememePSO',[]),('SememePSO/PAA',[])])),
                ]),
            ),
            # ('IMDB',OrderedDict([
            #     ('CNN',OrderedDict([ ('PAA',[]),('TextFooler',[]),('PSO',[]),('TextFooler/PAA',[]),('PSO/PAA',[])])),
            #     ('LSTM',OrderedDict([ ('PAA',[]),('TextFooler',[]),('PSO',[]),('TextFooler/PAA',[]),('PSO/PAA',[])])),
            #     ('BERT',OrderedDict([ ('PAA',[]),('TextFooler',[]),('PSO',[]),('TextFooler/PAA',[]),('PSO/PAA',[])])),
            #     ('RoBERTa',OrderedDict([ ('PAA',[]),('TextFooler',[]),('PSO',[]),('TextFooler/PAA',[]),('PSO/PAA',[])])),
            #     ('XLNet',OrderedDict([ ('PAA',[]),('TextFooler',[]),('PSO',[]),('TextFooler/PAA',[]),('PSO/PAA',[])])),
            #     ]),
            # ),
            ('MNLI',OrderedDict([
                ('BERT',OrderedDict([ ('PAA',[]),('TextFooler',[]),('SememePSO',[]),('TextFooler/PAA',[]),('SememePSO/PAA',[])])),
                ('DistilBERT',OrderedDict([ ('PAA',[]),('TextFooler',[]),('SememePSO',[]),('TextFooler/PAA',[]),('SememePSO/PAA',[])])),
                ]),
            ),
            ('SNLI',OrderedDict([
                ('BERT',OrderedDict([ ('PAA',[]),('TextFooler',[]),('SememePSO',[]),('TextFooler/PAA',[]),('SememePSO/PAA',[])])),
                ('DistilBERT',OrderedDict([ ('PAA',[]),('TextFooler',[]),('SememePSO',[]),('TextFooler/PAA',[]),('SememePSO/PAA',[])])),
                ]),
            ),

            ('QNLI',OrderedDict([
                ('BERT',OrderedDict([ ('PAA',[]),('TextFooler',[]),('SememePSO',[]),('TextFooler/PAA',[]),('SememePSO/PAA',[])])),
                ('RoBERTa',OrderedDict([ ('PAA',[]),('TextFooler',[]),('SememePSO',[]),('TextFooler/PAA',[]),('SememePSO/PAA',[])])),
                ('DistilBERT',OrderedDict([ ('PAA',[]),('TextFooler',[]),('SememePSO',[]),('TextFooler/PAA',[]),('SememePSO/PAA',[])])),
                ]),

            ),
            ('QQP',OrderedDict([
                ('BERT',OrderedDict([ ('PAA',[]),('TextFooler',[]),('SememePSO',[]),('TextFooler/PAA',[]),('SememePSO/PAA',[])])),
                ('DistilBERT',OrderedDict([ ('PAA',[]),('TextFooler',[]),('SememePSO',[]),('TextFooler/PAA',[]),('SememePSO/PAA',[])])),
                ('XLNet',OrderedDict([ ('PAA',[]),('TextFooler',[]),('SememePSO',[]),('TextFooler/PAA',[]),('SememePSO/PAA',[])])),
                ])
            ),


            ]
    )




print ('start')
for test in tests:
    name = os.path.join(folder,test)
    f = open(name, "r")
    lines = f.readlines()
    print (name)
    # get preliminary columns by looking at name
    output_table = []
    name_dataset = output_table + dataset_name(test)
    name_model = name_dataset + model_name(test)
    # name_method = name_model + [[test]]

    method = name_model + name_method(folder,test)
    lines_of_interest = method + test_performance(lines)
    lines_of_interest = order_columns(lines_of_interest)
    print (lines_of_interest)
    Ordered_table[lines_of_interest[0][1]][lines_of_interest[1][1]][lines_of_interest[2][1]] = [i[1] for i in lines_of_interest]

    # print ('Test:',test)
    # print (lines_of_interest)



column_names = OrderedDict((i[0],[]) for i in lines_of_interest)
column_names_index = [i[0] for i in lines_of_interest]

for i,dataset in Ordered_table.items():
    for j,model in dataset.items():
        for k,method in model.items():

            for l,P in enumerate(method):

                column_names[column_names_index[l]].append(P)

# calcualte drop in performance

original_acc = column_names['Original Acc[%]']
adv_acc = column_names['After Attack Acc [%]']
drop = []
for i in range(len(adv_acc)):
    diff = original_acc[i] - adv_acc[i]
    drop.append(diff)
column_names['Drop [%]'] = drop
print ('new cols:',column_names)

from tabulate import tabulate




headers= [i for i,j in column_names.items()]
table = tabulate(column_names, tablefmt="plain",headers=headers)
print (table)
with open(f'{args.folder_name}_table_output.txt', 'w') as f:
    f.write(table)

# with open(f'{args.folder_name}_table_output_pandas.txt', 'w') as f:
#     f.write(table)

# import pandas as pd
# pd.set_option("display.max_columns", None)
# print ('column names',column_names)
# DF = pd.DataFrame(column_names,headers=headers)
# DF.to_csv('{args.folder_name}_table_output_pandas.csv')
# print (DF)
