# import requests
#
# word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
#
# response = requests.get(word_site)
# WORDS = response.content.splitlines()
#
# print (WORDS)

import nltk
nltk.download('words')
from nltk.corpus import words

a = words.words()

len_words = {}

for word in a:
    res = any(c.isupper() for c in word)
    if res == True:
        pass
    else:
        len_words[word] = len(word)

histogram_of_word_len = {}

for i,j in len_words.items():
    if j in histogram_of_word_len:
        histogram_of_word_len[j] +=1
    else:
        histogram_of_word_len[j] = 1

# print ('histo',histogram_of_word_len)

histo_acutal_words = {}

for i,j in len_words.items():
    if j in histo_acutal_words:
        histo_acutal_words[j].append(i)
    else:
        histo_acutal_words[j] = [i]

import random

import string
# letters_to_insert = '.'
# letters_to_insert = '@'
# letters_to_insert = '\''
# letters_to_insert = "”!#$%&'()∗+,−./:;<=>?@[\]ˆ‘{|}"
letters_to_insert =string.ascii_lowercase

if letters_to_insert == '\'':
    random.seed(4)
elif letters_to_insert == '.':
    random.seed(3)
elif letters_to_insert == '@':
    random.seed(10)
elif letters_to_insert == "”!#$%&'()∗+,−./:;<=>?@[\]ˆ‘{|}":
    random.seed(5)
elif letters_to_insert == string.ascii_lowercase:
    random.seed(7)
# we use the nltk word list, and remove all words with capital letters to avoid names
final_list = []
for i in range(4,10):
    print ('i',i)
    small_subset = random.choices(histo_acutal_words[i], k=6)
    final_list = final_list + small_subset

print (final_list)

for i,j in enumerate(final_list):
    print (i,j)



def _get_random_letter():
    # letters_to_insert = '\''
    """Helper function that returns a random single letter from the English
    alphabet that could be lowercase or uppercase."""
    return random.choice(letters_to_insert)

attacked_words = []
skip_first_char = False
skip_last_char = False
p = 0.8

import numpy as np
for word in final_list:


    """Returns returns a list containing all possible words with 1 random
    character inserted."""

    # print ('word replaced',[word])

    start_idx = 1 if  skip_first_char else 0
    end_idx = (len(word) - 1) if skip_last_char else len(word)
    end_idx = len(word) - 1




    anchor_word = np.random.uniform(0,1)
    # print ('anchor_word',anchor_word,p)
    letters = ''


    while start_idx < end_idx:
    # for i in range(start_idx, end_idx):
        anchor_letter = np.random.uniform(0,1)
        if anchor_letter <= p:
            candidate_letters =   word[start_idx] + _get_random_letter()
            # print ('candidate letters1',candidate_letters)
            letters+=candidate_letters
            start_idx+=1
        else:
            letters += word[start_idx]
            # print ('candidate letters12',letters)
            start_idx+=1
        # print ('anchor_letter',anchor_letter,p,letters )
    letters+=word[start_idx]
    # print ('anchor_letter',anchor_letter,p,letters )




        # add random symbol
    attacked_words.append(letters)

# print ('attacked words',attacked_words)

for i,j in enumerate(attacked_words):
    print (i,j)
