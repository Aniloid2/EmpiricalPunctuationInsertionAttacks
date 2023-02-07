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
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding


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

)



# from .attack_recipe import AttackRecipe



parser = argparse.ArgumentParser(description='anomaly_train')
parser.add_argument('--specific_name',default=False,
                    help='attacked dataset')
parser.add_argument('--dataset_load',default=False,
                    help='attacked dataset')
parser.add_argument('--dataset_save',default='example',
                    help='attacked dataset')
# parser.add_argument('--model_source',default="textattack/bert-base-uncased-QNLI",
#                     help='attacked dataset')
parser.add_argument('--model_source',default="textattack/bert-base-uncased-rotten-tomatoes",
                    help='attacked dataset')
# "textattack/bert-base-uncased-rotten-tomatoes"
parser.add_argument('--global_dataset',default="MR",
                    help='attacked dataset')
parser.add_argument('--recipe',default="punctuation_attack",
                    help='attacked dataset')
parser.add_argument('--output_dir',default="adv_training",
                    help='where to store model')
parser.add_argument('--internal_type',default="non_internal",
                    help='attack using internal symbols or external symbols')
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


    if recipe =='punctuation_attack':
        transformation = WordSwapRandomCharacterInsertion(random_one=False,skip_first_char=True, skip_last_char=False,letters_to_insert=P)
         # We'll constrain modification of already modified indices and stopwords
        # constraints = [RepeatModification(), StopwordModification(), UniversalSentenceEncoder(threshold=0.8)]

        constraints = [ UniversalSentenceEncoder(threshold=0.8)]


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


    # We'll use the Greedy search method
    search_method =  GreedyWordSwapWIR(wir_method="delete")
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


# dataset_list = ['QNLI','QNLI','QQP','QQP','QQP']
# models_list = [
#
# 'textattack/distilbert-base-uncased-QNLI',"textattack/roberta-base-QNLI",
# "textattack/bert-base-uncased-QQP","textattack/distilbert-base-cased-QQP","textattack/xlnet-base-cased-QQP"]

dataset_list = [ 'MNLI']
models_list = [ 'textattack/distilbert-base-uncased-MNLI']

internal = args.internal_type

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

        model_path_final = f"./{internal}/{args.recipe}/{dset}/{model_path}/Models/"
        # model_path_symbol = f"./{args.recipe}/{dset}/{model_path}/"
        if args.dataset_load:
            Path(model_path_final).mkdir(parents=True, exist_ok=True)
            name_load = args.dataset_load+str(emb_size)
            transfer_attack_dataset = torch.load(os.path.join(model_path,name_load))

        else:
            attacker = Attacker(development_attack,dataset)
            attacker.attack_args.num_examples = num_examples
            attacker.attack_args.random_seed = 765
            attacker.attack_args.shuffle = True
            attacker.attack_dataset()
            transfer_attack_dataset = attacker.return_results_dataset()
            Path(model_path_final).mkdir(parents=True, exist_ok=True)
            name_save = args.dataset_save+'_'+str(save_P)+'_'+'symbol'
            torch.save(transfer_attack_dataset,os.path.join(model_path_final, name_save))


        Path(f"./internal/{args.recipe}/{dset}/{model_path}").mkdir(parents=True, exist_ok=True)



        file1 = open(f"./{internal}/{args.recipe}/{dset}/{model_path}/{args.recipe}_{str(save_P)}.txt","w")


        file_outer = open(f"./{internal}/{model_path}_{dset}_{args.recipe}_{str(save_P)}_symbol.txt","w")


        # attacker.attack_log_manager.results = final_results
        attacker.attack_log_manager.log_summary()
        rows = attacker.attack_log_manager.summary_table_rows
        for row in rows:
            file1.write(str(row)+'\n')
            file_outer.write(str(row)+'\n')
        file1.close()
