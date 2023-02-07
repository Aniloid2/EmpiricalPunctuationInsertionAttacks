import sys
import re
file = open("bert-base-uncased-MR-punctuation-attack.txt", "r")
current_result = { }
list_samples = []
for (i),x in enumerate(file):
    if 'Number Of Successful Attacks' in x:
        break

    if 'SKIPPED' in x or 'FAILED' in x:
        current_result['pass'] = True
        continue
    elif x == '\n':
        continue
    elif 'Result' in x:
        id = re.findall(r'\d+',x)[0]
        current_result = {'ID':int(id)}
        continue
    elif 'Negative' in x or 'Positive' in x:
        labels = x.split('-->')

        if 'Positive' in labels[0]:
            original_label = 1
            new_label = 0
        else:
            original_label = 0
            new_label = 1
        current_result['original_label'] = original_label
        current_result['new_label'] = new_label
        continue
    else:
        if x != '\n':
            if 'original_sentance' not in current_result.keys() :
                current_result['original_sentance'] = x.strip()
                continue
            else:
                current_result['adv_sentance'] = x.strip()

    if i == 25:
        print (current_result)
    list_samples.append(current_result)

    # sys.exit()

dictionary_frame = {'ID':[],'original_label':[],'new_label':[],'original_sentance':[],'adv_sentance':[]}
for idx,i in enumerate(list_samples):
    if idx == 120:
        break
    dictionary_frame['ID'].append(i['ID'])
    dictionary_frame['original_label'].append(i['original_label'])
    dictionary_frame['new_label'].append(i['new_label'])
    dictionary_frame['original_sentance'].append(i['original_sentance'])
    dictionary_frame['adv_sentance'].append(i['adv_sentance'])

import pandas as pd
pd.set_option("display.max_columns", None)
df = pd.DataFrame(data=dictionary_frame,columns=['ID','original_sentance','adv_sentance','original_label','new_label'])
df.to_csv('bert-mr-punc.csv',index=False)

print (df.head())
