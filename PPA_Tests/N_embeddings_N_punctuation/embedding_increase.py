
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
import os
# https://huggingface.co/textattack

# def set_seed(random_seed):
#     random.seed(random_seed)
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
# set_seed(765)

parser = argparse.ArgumentParser(description='transfer_test')
parser.add_argument('--specific_name',default=False,
                    help='attacked dataset')
parser.add_argument('--dataset_load',default=False,
                    help='attacked dataset')

parser.add_argument('--dataset_save',default='example.pt',
                    help='attacked dataset')
parser.add_argument('--model_source',default="textattack/bert-base-uncased-rotten-tomatoes",
                    help='attacked dataset')

parser.add_argument('--global_dataset',default="rotten_tomatoes",
                    help='attacked dataset')
parser.add_argument('--recipe',default="punctuation_attack",
                    help='attacked dataset')
parser.add_argument('--punctuation_list',default=".,\'-\"[]:()",
                    help='attacked dataset')



args = parser.parse_args()

from textattack import Attack
from textattack.search_methods import GreedySearch,GreedyWordSwapWIR

from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.pre_transformation import RepeatModification
from textattack.constraints.pre_transformation import StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.semantics import WordEmbeddingDistance


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
    WordSwapTokenSpecificPunctuationInsertion,
    WordSwapSampleSpecialSymbolInsertionExtended,

)





if args.global_dataset =='mnli':
    dataset = HuggingFaceDataset("glue", "mnli", "validation_matched", None, {0: 1, 1: 2, 2: 0},shuffle=True)
elif args.global_dataset =='qnli':
    dataset = HuggingFaceDataset("glue", args.global_dataset, "validation",shuffle=True)

elif  args.global_dataset =='rotten_tomatoes':
    dataset = HuggingFaceDataset(args.global_dataset, None, "test",shuffle=True)

def attack_setup(model_used,recipe,emb_size,total_list_char):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_used)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_used)
    # We wrap the model so it can be used by textattack
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)


    # We'll use untargeted classification as the goal function.
    goal_function = UntargetedClassification(model_wrapper)
    # constraints = [RepeatModification(), StopwordModification()]#,LanguageTool()] # if grammer checker use language tool
    stopwords = set(
        ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
    )
    if recipe =='punctuation_attack':
        letters_to_insert = total_list_char[:emb_size+1]
        transformation = WordSwapTokenSpecificPunctuationInsertion(all_char = True, random_one=True,skip_first_char=True, skip_last_char=False,letters_to_insert=letters_to_insert)
        # We'll constrain modification of already modified indices and stopwords
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords),PartOfSpeech(allow_verb_noun_swap=True),UniversalSentenceEncoder(threshold=0.8)]
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

    elif recipe == 'textfooler':
        transformation = WordSwapEmbedding(max_candidates=(emb_size+1))

        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords),WordEmbeddingDistance(min_cos_sim=0.5),PartOfSpeech(allow_verb_noun_swap=True),UniversalSentenceEncoder( threshold=0.840845057,  metric="angular", compare_against_original=False, window_size=15, skip_text_shorter_than_window=True )]

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
    attack.search_method.symbol_search_function = 'baseline'
    # attack.search_method.symbols = ['.',';', ':','_','-', '~','|','`','¬'][:emb_size+1]
    attack.search_method.symbols = ['¬','`','|','~','-','_',':',';','.'][:emb_size+1]
    attack.search_method.num_words = False
    # attack.search_method.symbols = ['.',';'][:emb_size+1]
    return attack

from pathlib import Path
for emb_size in range(11):

    development_attack = attack_setup(args.model_source,args.recipe ,emb_size,args.punctuation_list)

    if 'bert-base-uncased' in  args.model_source:
        model_path = 'bert-base-uncased'

    num_examples =500

    failed = 0
    skipped = 0
    correct = 0
    correct_transfer = 0
    model_path = f"./{args.recipe}/{args.global_dataset}/{model_path}/Models/"
    if args.dataset_load:
        Path(model_path).mkdir(parents=True, exist_ok=True)
        name_load = args.dataset_load+str(emb_size)
        transfer_attack_dataset = torch.load(os.path.join(model_path,name_load))

    else:
        attacker = Attacker(development_attack,dataset)
        attacker.attack_args.num_examples = num_examples
        attacker.attack_args.random_seed = 765
        attacker.attack_args.shuffle = True
        attacker.attack_dataset()
        transfer_attack_dataset = attacker.return_results_dataset()
        Path(model_path).mkdir(parents=True, exist_ok=True)
        name_save = args.dataset_save+str(emb_size)
        torch.save(transfer_attack_dataset,os.path.join(model_path, name_save))

    if 'bert-base-uncased' in  args.model_source:
        model_path =  'bert-base-uncased'

    Path(f"./{args.recipe}/{args.global_dataset}/{model_path}").mkdir(parents=True, exist_ok=True)

    file1 = open(f"./{args.recipe}/{args.global_dataset}/{model_path}/{args.recipe}_embedding_size_{str(emb_size+1)}.txt","w")



    # attacker.attack_log_manager.results = final_results
    attacker.attack_log_manager.log_summary()
    rows = attacker.attack_log_manager.summary_table_rows
    for row in rows:
        file1.write(str(row)+'\n')
    file1.close()
