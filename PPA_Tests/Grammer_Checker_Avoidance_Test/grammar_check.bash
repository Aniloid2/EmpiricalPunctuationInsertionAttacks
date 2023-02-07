
# python grammar_checker.py --recipe deepwordbug --internal_type 'internal' --type 'Non_Grammar_Internal'   --stopwords_arg ;
# python grammar_checker.py --recipe punctuation_attack --internal_type 'internal'  --type 'Non_Grammar_Internal' --stopwords_arg ;
# python grammar_checker.py --recipe punctuation_attack --internal_type 'non_internal'  --type 'Non_Grammar_Non_Internal' --stopwords_arg ;
#
# python grammar_checker.py --recipe punctuation_attack_random_one_true --internal_type 'internal'  --type 'Non_Grammar_Internal_Random_Pos' --stopwords_arg ; # this use to be grammar
# python grammar_checker.py --recipe punctuation_attack_random_one_true --internal_type 'non_internal'  --type 'Non_Grammar_Non_Internal_Random_Pos' --stopwords_arg ;
#
#
#
# python grammar_checker.py --recipe deepwordbug_grammar --internal_type 'internal' --type 'Grammar_Internal' --stopwords_arg ;
# python grammar_checker.py --recipe punctuation_attack_grammar --internal_type 'internal'  --type 'Grammar_Internal' --stopwords_arg ;
# python grammar_checker.py --recipe punctuation_attack_grammar --internal_type 'non_internal'  --type 'Grammar_Non_Internal' --stopwords_arg ;
# #
# python grammar_checker.py --recipe punctuation_attack_grammar_random_one_true --internal_type 'internal'  --type 'Grammar_Internal_Random_Pos' --stopwords_arg ;
# python grammar_checker.py --recipe punctuation_attack_grammar_random_one_true --internal_type 'non_internal'  --type 'Grammar_Non_Internal_Random_Pos' --stopwords_arg ;
# #
#
# python grammar_checker.py --recipe deepwordbug --internal_type 'internal' --type 'Non_Grammar_Internal_Random_Pos' --stopwords_arg ;
# python grammar_checker.py --recipe deepwordbug_grammar --internal_type 'internal' --type 'Grammar_Internal_Random_Pos' --stopwords_arg ;


# python grammar_checker.py --recipe punctuation_attack_random_one_true --internal_type 'internal'  --type 'Non_Grammar_Internal_Random_Pos' --stopwords_arg ;
# python grammar_checker.py --recipe punctuation_attack --internal_type 'internal'  --type 'Non_Grammar_Internal' --stopwords_arg ;
# python grammar_checker.py --recipe deepwordbug --internal_type 'internal' --type 'Non_Grammar_Internal'   --stopwords_arg ;

# python grammar_checker.py --recipe deepwordbug_grammar --internal_type 'internal' --type 'Grammar_Internal' --stopwords_arg ;



python grammar_checker.py --recipe zeroe_grammar --internal_type 'internal' --type 'Grammar_Internal_Zeroe' --stopwords_arg ;
python grammar_checker.py --recipe zeroe_grammar_all --internal_type 'internal' --type 'Grammar_Internal_Zeroe_All' --stopwords_arg ;
python grammar_checker.py --recipe zeroe_grammar --internal_type 'non_internal' --type 'Grammar_Non_Internal_Zeroe' --stopwords_arg ;
