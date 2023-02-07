# from dataset.dataset_wrapper import AnomalyDataset
import transformers
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_results import FailedAttackResult, SkippedAttackResult
import sys
from textattack.datasets import HuggingFaceDataset

import numpy as np
import torch
import random
from textattack import Attacker
import argparse
from pathlib import Path
import os

from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR,BlackBoxSearch
from textattack.transformations import WordSwapEmbedding
from textattack.search_methods import ParticleSwarmOptimization
from textattack.transformations import WordSwapHowNet
from textattack.constraints.grammaticality import LanguageTool

from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapSpecialSymbolInsertion,
    WordSwapSpecialSymbolSubstitution,
    WordSwapSampleSpecialSymbolInsertion,
    WordSwapSampleSpecialSymbolInsertionExtended,
    WordSwapTokenSpecificPunctuationInsertion,
    WordSwapRandomCharacterSubstitution,
    WordSwapZeroeCharProbInsertion


)



parser = argparse.ArgumentParser(description='anomaly_train')

parser.add_argument('--dataset_load',default=False,
                    help='attacked dataset')
# "textattack/bert-base-uncased-rotten-tomatoes"
parser.add_argument('--global_dataset',default="MR",
                    help='attacked dataset')
parser.add_argument('--recipe',default="punctuation_attack",
                    help='attacked dataset')
parser.add_argument('--punc',default="",
                    help='punctuation to use agianst grammar checker')
parser.add_argument('--internal_type',default="None",
                    help='attack using internal symbols or external symbols')
parser.add_argument('--type',default="Grammar",
                    help='save type')
parser.add_argument('--stopwords_arg',default=False, action='store_true',
                    help='use stopwords or not')
# parser.add_argument('--punc',default='all',
#                     help='for zeroe only, what punc to use')
args = parser.parse_args()

import textattack
import transformers


def attack_setup(model_used,recipe,punctuation):
    if 'cnn' in model_used:
        model = (
            textattack.models.helpers.WordCNNForClassification.from_pretrained(
                model_used
            )
        )
        model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(
            model, model.tokenizer
        )
    elif 'lstm' in model_used:

        model = textattack.models.helpers.LSTMForClassification.from_pretrained(
            model_used
        )
        model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(
            model, model.tokenizer
        )
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_used)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_used)
        # We wrap the model so it can be used by textattack
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)


    # We'll use untargeted classification as the goal function.
    goal_function = UntargetedClassification(model_wrapper)

    stopwords = set(
        ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
    )

    if recipe =='punctuation_attack':
        transformation = WordSwapTokenSpecificPunctuationInsertion(all_char = True,random_one=False,skip_first_char=True, skip_last_char=False,letters_to_insert=punctuation)
         # We'll constrain modification of already modified indices and stopwords
        # constraints = [RepeatModification(), StopwordModification(), UniversalSentenceEncoder(threshold=0.8)]

        if args.stopwords_arg:
            constraints = [RepeatModification(), StopwordModification(stopwords=stopwords), UniversalSentenceEncoder(threshold=0.8)]
        else:
            constraints = [RepeatModification(), UniversalSentenceEncoder(threshold=0.8)]


        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )

        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question1", "question2"], {"question1"}
        )
        constraints.append(input_column_modification)
        search_method =  GreedyWordSwapWIR(wir_method="delete")

    if recipe =='punctuation_attack_random_one_true':
        transformation = WordSwapTokenSpecificPunctuationInsertion(all_char = True,random_one=True,skip_first_char=True, skip_last_char=False,letters_to_insert=punctuation)
         # We'll constrain modification of already modified indices and stopwords
        # constraints = [RepeatModification(), StopwordModification(), UniversalSentenceEncoder(threshold=0.8)]

        if args.stopwords_arg:
            constraints = [RepeatModification(), StopwordModification(stopwords=stopwords), UniversalSentenceEncoder(threshold=0.8)]
        else:
            constraints = [RepeatModification(), UniversalSentenceEncoder(threshold=0.8)]


        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )

        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question1", "question2"], {"question1"}
        )
        constraints.append(input_column_modification)
        search_method =  GreedyWordSwapWIR(wir_method="delete")


    if recipe =='punctuation_attack_grammar_random_one_true':
        transformation = WordSwapTokenSpecificPunctuationInsertion(all_char = True,random_one=True,skip_first_char=True, skip_last_char=False,letters_to_insert=punctuation)
         # We'll constrain modification of already modified indices and stopwords
        # constraints = [RepeatModification(), StopwordModification(), UniversalSentenceEncoder(threshold=0.8)]

        if args.stopwords_arg:
            constraints = [RepeatModification(), StopwordModification(stopwords=stopwords), UniversalSentenceEncoder(threshold=0.8),LanguageTool()]
        else:
            constraints = [RepeatModification(), UniversalSentenceEncoder(threshold=0.8),LanguageTool()]


        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )

        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question1", "question2"], {"question1"}
        )
        constraints.append(input_column_modification)
        search_method =  GreedyWordSwapWIR(wir_method="delete")

    if recipe =='punctuation_attack_grammar':
        transformation = WordSwapTokenSpecificPunctuationInsertion(all_char = True,random_one=False,skip_first_char=True, skip_last_char=False,letters_to_insert=punctuation)
         # We'll constrain modification of already modified indices and stopwords
        # constraints = [RepeatModification(), StopwordModification(), UniversalSentenceEncoder(threshold=0.8)]

        if args.stopwords_arg:
            constraints = [RepeatModification(), StopwordModification(stopwords=stopwords), UniversalSentenceEncoder(threshold=0.8),LanguageTool()]
        else:
            constraints = [RepeatModification(), UniversalSentenceEncoder(threshold=0.8),LanguageTool()]


        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )

        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question1", "question2"], {"question1"}
        )
        constraints.append(input_column_modification)
        search_method =  GreedyWordSwapWIR(wir_method="delete")
    elif recipe == 'deepwordbug':

        transformation = CompositeTransformation(
            [
                # (1) Swap: Swap two adjacent letters in the word.
                WordSwapNeighboringCharacterSwap(),
                # (2) Substitution: Substitute a letter in the word with a random letter.
                WordSwapRandomCharacterSubstitution(),
                # (3) Deletion: Delete a random letter from the word.
                WordSwapRandomCharacterDeletion(),
                # (4) Insertion: Insert a random letter in the word.
                WordSwapRandomCharacterInsertion(),
                # WordSwapSpecialSymbolInsertion(),
            ]
        )


        # Don't modify the same word twice or stopwords
        #
        if args.stopwords_arg:
            constraints = [RepeatModification(), StopwordModification(stopwords=stopwords), UniversalSentenceEncoder(threshold=0.8)]

        else:
            constraints = [RepeatModification(), UniversalSentenceEncoder(threshold=0.8)]

        # In these experiments, we hold the maximum difference
        # on edit distance (ϵ) to a constant 30 for each sample.
        #
        # constraints.append(LevenshteinEditDistance(10)) \
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )

        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question1", "question2"], {"question1"}
        )
        constraints.append(input_column_modification)
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        # search_method = GreedyWordSwapWIR(wir_method="delete")
        search_method = GreedyWordSwapWIR(wir_method="delete")
    elif recipe == 'deepwordbug_grammar':
        transformation = CompositeTransformation(
            [
                # (1) Swap: Swap two adjacent letters in the word.
                WordSwapNeighboringCharacterSwap(),
                # (2) Substitution: Substitute a letter in the word with a random letter.
                WordSwapRandomCharacterSubstitution(),
                # (3) Deletion: Delete a random letter from the word.
                WordSwapRandomCharacterDeletion(),
                # (4) Insertion: Insert a random letter in the word.
                WordSwapRandomCharacterInsertion(),
                # WordSwapSpecialSymbolInsertion(),
            ]
        )


        # Don't modify the same word twice or stopwords
        #
        if args.stopwords_arg:
            constraints = [RepeatModification(), StopwordModification(stopwords=stopwords), UniversalSentenceEncoder(threshold=0.8),LanguageTool()]
        else:
            constraints = [RepeatModification(), UniversalSentenceEncoder(threshold=0.8),LanguageTool()]


        #
        # In these experiments, we hold the maximum difference
        # on edit distance (ϵ) to a constant 30 for each sample.
        #
        # constraints.append(LevenshteinEditDistance(10)) \
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )

        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question1", "question2"], {"question1"}
        )
        constraints.append(input_column_modification)
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        # search_method = GreedyWordSwapWIR(wir_method="delete")
        search_method = GreedyWordSwapWIR(wir_method="delete")

    elif recipe == 'zeroe_grammar':

        # if args.punc == 'all':
        #     local_punc =  "”!#$%&'()∗+,−./:;<=>?@[\]ˆ‘{|}" #!”#$%&()∗+, −./ :; <=>?@[\]ˆ‘{|}
        # elif args.punc == 'char':
        #     local_punc = string.ascii_lowercase
        # else:
        #     local_punc = args.punc
        p = 0.8
        local_punc = punctuation
        transformation = WordSwapZeroeCharProbInsertion(p = p, random_one=True, letters_to_insert=local_punc)  #letters_to_insert="!”#$%&'()∗+, −./ :; <=>?@[\]ˆ‘{|}"

        if args.stopwords_arg:
            constraints = [RepeatModification(), StopwordModification(stopwords=stopwords), UniversalSentenceEncoder(threshold=0.8),LanguageTool()]
        else:
            constraints = [RepeatModification(), UniversalSentenceEncoder(threshold=0.8),LanguageTool()]

        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )

        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question1", "question2"], {"question1"}
        )
        constraints.append(input_column_modification)
        #
        goal_function = UntargetedClassification(model_wrapper)
        search_method = BlackBoxSearch()

    elif recipe == 'zeroe_grammar_all':

        # if args.punc == 'all':
        local_punc =  "”!#$%&'()∗+,−./:;<=>?@[\]ˆ‘{|}" #!”#$%&()∗+, −./ :; <=>?@[\]ˆ‘{|}
        # elif args.punc == 'char':
        #     local_punc = string.ascii_lowercase
        # else:
        #     local_punc = args.punc
        p = 0.8
        transformation = WordSwapZeroeCharProbInsertion(p = p, random_one=True, letters_to_insert=local_punc)  #letters_to_insert="!”#$%&'()∗+, −./ :; <=>?@[\]ˆ‘{|}"

        if args.stopwords_arg:
            constraints = [RepeatModification(), StopwordModification(stopwords=stopwords), UniversalSentenceEncoder(threshold=0.8),LanguageTool()]
        else:
            constraints = [RepeatModification(), UniversalSentenceEncoder(threshold=0.8),LanguageTool()]

        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )

        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question1", "question2"], {"question1"}
        )
        constraints.append(input_column_modification)
        #
        goal_function = UntargetedClassification(model_wrapper)
        search_method = BlackBoxSearch()

    # We'll use the Greedy search method
    # Now, let's make the attack from the 4 components:
    attack = Attack(goal_function, constraints, transformation, search_method)
    # attack.search_method.symbols = ['.',';', ':','_','-', '~','|','`','¬'][:emb_size+1]
    attack.search_method.symbols = punctuation
    attack.search_method.num_words = False
    attack.search_method.symbol_search_function = 'baseline'
    # attack.search_method.symbols = ['.',';'][:emb_size+1]
    return attack




# dataset_list = ['MR','MR','MR','MR','MR','IMDB','IMDB','IMDB','IMDB','IMDB','MNLI','MNLI','SNLI','SNLI','QNLI','QNLI','QNLI','QQP','QQP','QQP']
# models_list = ['lstm-mr','cnn-mr','textattack/bert-base-uncased-rotten-tomatoes','textattack/roberta-base-rotten-tomatoes','textattack/xlnet-base-cased-rotten-tomatoes',
# 'lstm-imdb','cnn-imdb','textattack/bert-base-uncased-imdb','textattack/roberta-base-imdb','textattack/xlnet-base-cased-imdb',
# 'textattack/bert-base-uncased-MNLI','textattack/distilbert-base-uncased-MNLI',
# 'textattack/bert-base-uncased-snli','textattack/distilbert-base-cased-snli',
# 'textattack/bert-base-uncased-QNLI','textattack/distilbert-base-uncased-QNLI',"textattack/roberta-base-QNLI",
# "textattack/bert-base-uncased-QQP","textattack/distilbert-base-cased-QQP","textattack/xlnet-base-cased-QQP"]


# dataset_list = ['IMDB','IMDB','IMDB','IMDB','IMDB','MNLI','MNLI','SNLI','SNLI','QNLI','QNLI','QNLI','QQP','QQP','QQP']
# models_list = ['lstm-imdb','cnn-imdb','textattack/bert-base-uncased-imdb','textattack/roberta-base-imdb','textattack/xlnet-base-cased-imdb',
# 'textattack/bert-base-uncased-MNLI','textattack/distilbert-base-uncased-MNLI',
# 'textattack/bert-base-uncased-snli','textattack/distilbert-base-cased-snli',
# 'textattack/bert-base-uncased-QNLI','textattack/distilbert-base-uncased-QNLI',"textattack/roberta-base-QNLI",
# "textattack/bert-base-uncased-QQP","textattack/distilbert-base-cased-QQP","textattack/xlnet-base-cased-QQP"]


# dataset_list = ['mr','mr','mr','mr','mr' ]
# models_list = ['lstm-mr','cnn-mr','textattack/bert-base-uncased-rotten-tomatoes','textattack/roberta-base-rotten-tomatoes','textattack/xlnet-base-cased-rotten-tomatoes']

# dataset_list = ['mr','mr' ]
# models_list = ['lstm-mr', 'textattack/bert-base-uncased-rotten-tomatoes' ]

# dataset_list = ['MR' ]
# models_list = ['lstm-mr'  ]

# dataset_list = ['SNLI']
# models_list = ['textattack/distilbert-base-cased-snli']

dataset_list = ['MR','MR','MR','MR','MR','MNLI','MNLI','SNLI','SNLI','QNLI','QNLI','QNLI','QQP','QQP','QQP']
models_list = ['lstm-mr','cnn-mr','textattack/bert-base-uncased-rotten-tomatoes','textattack/roberta-base-rotten-tomatoes','textattack/xlnet-base-cased-rotten-tomatoes',
'textattack/bert-base-uncased-MNLI','textattack/distilbert-base-uncased-MNLI',
'textattack/bert-base-uncased-snli','textattack/distilbert-base-cased-snli',
'textattack/bert-base-uncased-QNLI','textattack/distilbert-base-uncased-QNLI',"textattack/roberta-base-QNLI",
"textattack/bert-base-uncased-QQP","textattack/distilbert-base-cased-QQP","textattack/xlnet-base-cased-QQP"]


# dataset_list = ['MR','MR','MR','MR','MR','MNLI','MNLI','SNLI','SNLI' ]
# models_list = ['lstm-mr','cnn-mr','textattack/bert-base-uncased-rotten-tomatoes','textattack/roberta-base-rotten-tomatoes','textattack/xlnet-base-cased-rotten-tomatoes',
# 'textattack/bert-base-uncased-MNLI','textattack/distilbert-base-uncased-MNLI',
# 'textattack/bert-base-uncased-snli','textattack/distilbert-base-cased-snli']


# dataset_list = ['MR','MNLI' ]
# models_list = ['textattack/bert-base-uncased-rotten-tomatoes','textattack/bert-base-uncased-MNLI'  ]

# if args.recipe == 'deepwordbug':
# dataset_list = ['QNLI','QNLI','QNLI','QQP','QQP','QQP' ]
# models_list = [ 'textattack/bert-base-uncased-QNLI','textattack/distilbert-base-uncased-QNLI',"textattack/roberta-base-QNLI",
# "textattack/bert-base-uncased-QQP","textattack/distilbert-base-cased-QQP","textattack/xlnet-base-cased-QQP" ]
# elif args.recipe == 'deepwordbug_grammar':
#     dataset_list = ['MR','MR','MR','MR','MR','MNLI','MNLI','SNLI','SNLI','QNLI','QNLI','QNLI','QQP','QQP','QQP']
#     models_list = ['lstm-mr','cnn-mr','textattack/bert-base-uncased-rotten-tomatoes','textattack/roberta-base-rotten-tomatoes','textattack/xlnet-base-cased-rotten-tomatoes',
#     'textattack/bert-base-uncased-MNLI','textattack/distilbert-base-uncased-MNLI',
#     'textattack/bert-base-uncased-snli','textattack/distilbert-base-cased-snli',
#     'textattack/bert-base-uncased-QNLI','textattack/distilbert-base-uncased-QNLI',"textattack/roberta-base-QNLI",
#     "textattack/bert-base-uncased-QQP","textattack/distilbert-base-cased-QQP","textattack/xlnet-base-cased-QQP"]



# dataset_list = [ 'QNLI','QNLI']
# models_list = [
#  'textattack/distilbert-base-uncased-QNLI'  ]


internal = args.internal_type

# if args.punc == '\'':
#     punctuation_type = '_\'_symbol'
# elif args.punc == '-':
#     punctuation_type = '_-_symbol'
# else:
#     punctuation_type = ''

from pathlib import Path
for d in range(len(dataset_list)):
    dset = dataset_list[d]
    mod = models_list[d]
    if dset == 'MR':
        dataset = HuggingFaceDataset('rotten_tomatoes', None,'test',shuffle=True)
        if internal == 'non_internal':
            punc_type = ['.',',','\"']
            # punc_type = ['\"']
        elif internal == 'internal':
            punc_type = ['\'','-']
    elif dset == 'IMDB':
        dataset = HuggingFaceDataset( 'imdb',None, "test",shuffle=True)
        if internal == 'non_internal':
            punc_type = [',','/','>']
            # punc_type = ['/','>']
        elif internal == 'internal':
            punc_type = ['\'','.']
    elif dset == 'MNLI':
        dataset = HuggingFaceDataset("glue",'mnli',"validation_matched",None,{0: 1, 1: 2, 2: 0},shuffle=True)
        if internal == 'non_internal':
            punc_type = ['.',',',')']
            # punc_type = [ ')']
        elif internal == 'internal':
            punc_type = ['\'','-']
    elif dset == 'SNLI':
        if 'distilbert' in mod:
            dataset = HuggingFaceDataset( 'snli', None, "test", None, None,shuffle=True)
        else:
            dataset = HuggingFaceDataset( 'snli', None, "test", None, {0: 1, 1: 2, 2: 0},shuffle=True)
            dataset.filter_by_labels_([0,1,2])
        if internal == 'non_internal':
            punc_type = ['.',',','\"']
        elif internal == 'internal':
            punc_type = ['\'','-']
            # punc_type = ['\'']
    elif dset == 'QNLI':
        dataset = HuggingFaceDataset('glue', 'qnli', "validation",shuffle=True)
        if internal == 'non_internal':
            punc_type = [',','.','?']
        elif internal == 'internal':
            punc_type = ['\'','-']
    elif dset == 'QQP':
        dataset = HuggingFaceDataset('glue', 'qqp', "validation",shuffle=True)
        if internal == 'non_internal':
            punc_type = ['?',',','\"']
        elif internal == 'internal':
            punc_type = ['\'','-']

    if 'deepwordbug' in args.recipe:
        punc_type = ['D']
    if 'zeroe_grammar_all' in args.recipe:
        punc_type = ['D']


    for P in punc_type:
        development_attack = attack_setup(mod,args.recipe,P)
        num_examples = 500
        if 'distilbert' in mod:
            model_path = 'distilbert-base-uncased'
        elif 'bert-base-uncased' in  mod:
            model_path =  'bert-base-uncased'
        elif 'roberta' in mod:
            model_path =  'roberta-base'
        elif 'xlnet' in mod:
            model_path =  'xlnet-cased-base'
        elif 'lstm' in mod:
            model_path =  'lstm'
        elif 'cnn' in mod:
            model_path =  'cnn'


        if P == '/':
            save_P = 'divisiondash'
        elif P == '>':
            save_P = 'greaterthan'
        elif P == '\"':
            save_P = 'dubblequote'
        else:
            save_P = P


        model_path_final = f"./{args.recipe}/{dset}/{model_path}/Models/"

        model_path_final = f"./{internal}/{args.recipe}/{dset}/{model_path}/Models/"

        result_path_symbol = f"./Result/{args.type}"
        # if 'grammar' in args.recipe:
        #     result_path_symbol = f"./Result/Grammar/"
        # else:
        #     result_path_symbol = f"./Result/Non_Grammar/"

        if args.dataset_load:
            Path(model_path_final).mkdir(parents=True, exist_ok=True)
            name_load = args.dataset_load+str(emb_size)
            transfer_attack_dataset = torch.load(os.path.join(model_path,name_load))
        else:
            Path(model_path_final).mkdir(parents=True, exist_ok=True)
            Path(result_path_symbol).mkdir(parents=True, exist_ok=True)
            # name_save = f'{model_path}-{dset}-{args.recipe}{punctuation_type}.txt'
            if 'deepwordbug' in args.recipe:
                name_save = f'{model_path}-{dset}-{args.recipe}.txt'
            elif 'zeroe_all' in args.recipe:
                name_save = f'{model_path}-{dset}-{args.recipe}.txt'
            else:
                name_save = f'{model_path}-{dset}-{args.recipe}'+'_'+str(save_P)+'_'+'symbol.txt'

            result_path_symbol_final = os.path.join(result_path_symbol,name_save)
            attack_args = textattack.AttackArgs(log_to_txt= f"{result_path_symbol_final}")
            attacker = Attacker(development_attack,dataset,attack_args)
            attacker.attack_args.num_examples = num_examples
            attacker.attack_args.random_seed = 765
            attacker.attack_args.shuffle = True
            attacker.attack_dataset()

            transfer_attack_dataset = attacker.return_results_dataset()

            torch.save(transfer_attack_dataset,os.path.join(model_path_final,name_save))


    # Path(f"./{args.recipe}/{dset}/{model_path}").mkdir(parents=True, exist_ok=True)
    #
    #
    #
    # file1 = open(f"./{args.recipe}/{dset}/{model_path}/{args.recipe}_{str(save_P)}.txt","w")
    #
    #
    # file_outer = open(f"./{model_path}_{dset}_{args.recipe}_{str(save_P)}_symbol.txt","w")


    # attacker.attack_log_manager.results = final_results
    # attacker.attack_log_manager.log_summary()
    # rows = attacker.attack_log_manager.summary_table_rows
    # for row in rows:
    #     file1.write(str(row)+'\n')
    #     file_outer.write(str(row)+'\n')
    # file1.close()
