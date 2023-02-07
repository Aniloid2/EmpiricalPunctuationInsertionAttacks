python embedding_increase.py --specific_name 'testing' --punctuation_list .,\'-\"[]:\(\)  --dataset_save './bert_treie.pt' --model_source "textattack/bert-base-uncased-rotten-tomatoes" --global_dataset "rotten_tomatoes" --recipe "punctuation_attack" ;
python embedding_increase.py --specific_name 'testing' --punctuation_list .,\'-\"[]:\(\)  --dataset_save './bert_treie.pt' --model_source "textattack/bert-base-uncased-rotten-tomatoes" --global_dataset "rotten_tomatoes" --recipe "textfooler" ;
#
#
python embedding_increase.py --specific_name 'testing' --punctuation_list .,\'-\(\)?\"\;\! --dataset_save './bert_treie.pt' --model_source "textattack/bert-base-uncased-MNLI"  --global_dataset "mnli" --recipe "punctuation_attack" ;
python embedding_increase.py --specific_name 'testing' --punctuation_list .,\'-\(\)?\"\;\! --dataset_save './bert_treie.pt' --model_source "textattack/bert-base-uncased-MNLI"  --global_dataset "mnli" --recipe "textfooler" ;
