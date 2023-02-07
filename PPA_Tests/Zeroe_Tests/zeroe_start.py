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
import string
from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

from textattack.constraints.overlap import MaxWordsPerturbed

from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR, BlackBoxSearch
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
    WordSwapZeroeCharProbInsertion,


)



parser = argparse.ArgumentParser(description='anomaly_train')

parser.add_argument('--dataset_load',default=False,
                    help='attacked dataset')
# "textattack/bert-base-uncased-rotten-tomatoes"
parser.add_argument('--dataset_input',default="MR",
                    help='attacked dataset')
parser.add_argument('--model',default="textattack/bert-base-uncased-rotten-tomatoes",
                    help='attacked dataset')
parser.add_argument('--recipe',default="punctuation_attack",
                    help='attacked dataset')
parser.add_argument('--p',default=0.2,type=float,
                    help='strenght of attack')
parser.add_argument('--punc',default="",
                    help='punctuation to use agianst grammar checker')
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

        constraints = [MaxWordsPerturbed(max_percent=0.1),RepeatModification(), StopwordModification(stopwords=stopwords), UniversalSentenceEncoder(threshold=0.8)]


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
        search_method =  GreedyWordSwapWIR(wir_method="delete") # this has to be custom

    elif recipe == 'zeroe':

        if args.punc == 'all':
            local_punc =  "”!#$%&'()∗+,−./:;<=>?@[\]ˆ‘{|}" #!”#$%&()∗+, −./ :; <=>?@[\]ˆ‘{|}
        elif args.punc == 'char':
            local_punc = string.ascii_lowercase
        else:
            local_punc = args.punc

        transformation = WordSwapZeroeCharProbInsertion(p = args.p, random_one=True, letters_to_insert=local_punc)  #letters_to_insert="!”#$%&'()∗+, −./ :; <=>?@[\]ˆ‘{|}"



        # transformation = CompositeTransformation(
        #     [
        #         # (1) Swap: Swap two adjacent letters in the word.
        #         # WordSwapNeighboringCharacterSwap(),
        #         # # (2) Substitution: Substitute a letter in the word with a random letter.
        #         # WordSwapRandomCharacterSubstitution(),
        #         # # (3) Deletion: Delete a random letter from the word.
        #         # WordSwapRandomCharacterDeletion(),
        #         # (4) Insertion: Insert a random letter in the word.
        #         WordSwapRandomCharacterInsertion(random_one=False,letters_to_insert=punctuation,skip_first_char=True, skip_last_char=False),
        #         # WordSwapSpecialSymbolInsertion(),
        #     ]
        # )


        # Don't modify the same word twice or stopwords
        #
        constraints = []
        # constraints = [MaxWordsPerturbed(max_percent=0.1),RepeatModification(), StopwordModification(stopwords=stopwords),UniversalSentenceEncoder(threshold=0.8)]

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

# dataset_list = ['mr' ]
# models_list = ['lstm-mr'  ]

# dataset_list = ['SNLI']
# models_list = ['textattack/distilbert-base-cased-snli']

# dataset_list = ['MR','MR','MR','MR','MR','MNLI','MNLI','SNLI','SNLI','QNLI','QNLI','QNLI','QQP','QQP','QQP']
# models_list = ['lstm-mr','cnn-mr','textattack/bert-base-uncased-rotten-tomatoes','textattack/roberta-base-rotten-tomatoes','textattack/xlnet-base-cased-rotten-tomatoes',
# 'textattack/bert-base-uncased-MNLI','textattack/distilbert-base-uncased-MNLI',
# 'textattack/bert-base-uncased-snli','textattack/distilbert-base-cased-snli',
# 'textattack/bert-base-uncased-QNLI','textattack/distilbert-base-uncased-QNLI',"textattack/roberta-base-QNLI",
# "textattack/bert-base-uncased-QQP","textattack/distilbert-base-cased-QQP","textattack/xlnet-base-cased-QQP"]

dataset_list = [args.dataset_input]
models_list = [ args.model]


# dataset_list = ['MR']
# models_list = [
# 'textattack/bert-base-uncased-rotten-tomatoes']

# dataset_list = ['MNLI']
# models_list = [
# 'textattack/bert-base-uncased-MNLI']

#
# dataset_list = ['MR','MNLI','MNLI','SNLI','SNLI','QNLI','QNLI']#],'QQP','QQP']
# models_list = ['lstm-mr',
# 'textattack/bert-base-uncased-MNLI','textattack/distilbert-base-uncased-MNLI',
# 'textattack/bert-base-uncased-snli','textattack/distilbert-base-cased-snli',
# 'textattack/bert-base-uncased-QNLI','textattack/distilbert-base-uncased-QNLI',
# # "textattack/bert-base-uncased-QQP","textattack/distilbert-base-cased-QQP"
# ]

dataset_list = ['QQP','QQP']
models_list = [
"textattack/bert-base-uncased-QQP","textattack/distilbert-base-cased-QQP"]


if 'punctuation_attack' in args.recipe :
    if args.punc == '\'':
        punctuation_type = '_\'_symbol'
    elif args.punc == '-':
        punctuation_type = '_-_symbol'
    elif args.punc == '.':
        punctuation_type = '_._symbol'
    elif args.punc == ',':
        punctuation_type = '_,_symbol'
    elif args.punc == '?':
        punctuation_type = '_?_symbol'
    else:
        print ('declare a punctuation type')
        sys.exit()
else:
    punctuation_type = f'_{args.punc}_{args.p}_symbol'


from pathlib import Path
for d in range(len(dataset_list)):
    dset = dataset_list[d]
    mod = models_list[d]
    if dset == 'MR':
        dataset = HuggingFaceDataset('rotten_tomatoes', None,'test',shuffle=True)

    elif dset == 'IMDB':
        dataset = HuggingFaceDataset( 'imdb',None, "test",shuffle=True)

    elif dset == 'MNLI':
        dataset = HuggingFaceDataset("glue",'mnli',"validation_matched",None,{0: 1, 1: 2, 2: 0},shuffle=True)

    elif dset == 'SNLI':
        if 'distilbert' in mod:
            dataset = HuggingFaceDataset( 'snli', None, "test", None, None,shuffle=True)
        else:
            dataset = HuggingFaceDataset( 'snli', None, "test", None, {0: 1, 1: 2, 2: 0},shuffle=True)
            dataset.filter_by_labels_([0,1,2])

    elif dset == 'QNLI':
        dataset = HuggingFaceDataset('glue', 'qnli', "validation",shuffle=True)

    elif dset == 'QQP':
        dataset = HuggingFaceDataset('glue', 'qqp', "validation",shuffle=True)


    development_attack = attack_setup(mod,args.recipe,args.punc)
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


    model_path_final = f"./{args.recipe}/{dset}/{model_path}/Models/"
    if 'grammar' in args.recipe:
        result_path_symbol = f"./Result/Grammar/{dset}/{model_path}"
    else:
        result_path_symbol = f"./Result/Non_Grammar/{dset}/{model_path}"
    if args.dataset_load:
        Path(model_path_final).mkdir(parents=True, exist_ok=True)
        name_load = args.dataset_load+str(emb_size)
        transfer_attack_dataset = torch.load(os.path.join(model_path,name_load))
    else:
        Path(model_path_final).mkdir(parents=True, exist_ok=True)
        Path(result_path_symbol).mkdir(parents=True, exist_ok=True)
        name_save = f'{model_path}-{dset}-{args.recipe}{punctuation_type}.txt'
        result_path_symbol_final = os.path.join(result_path_symbol,name_save)
        attack_args = textattack.AttackArgs(log_to_txt= f"{result_path_symbol_final}" )
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
