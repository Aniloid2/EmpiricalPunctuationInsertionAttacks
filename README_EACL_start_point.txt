Running PAA

unzip PAA_code_test_ACL
unzip TextAttack

Conda
If conda isn't installed:
get it from here:
https://www.anaconda.com/products/individual-d
if conda is installed:
create env:
conda create --name PAA python=3.8.5 pip=20.2.4
source activate PAA

THE FOLLOWING TWO COMMANDS SHOULD INSTALL ALL DEPENDENCIES AND TEXTATTACK FROM THIS REPO USING THE SETUP FILE 
(MAKE SURE YOU INSTALL THE VERSION OF TEXTATTACK IN THIS REPO)
pip install --use-feature=2020-resolver -r local_run.txt
pip install -e ./
 
 

#### all dependencies should be installed by now ####

# note, treie == CATE == PAA == EmpiricalPunctuationInsertionAttacks == EPIA (we changed the name last minute, we will update the name in the future) 


SENTIMENT mr  
in Classification_Tests run:
python classification_test.py --recipe 'punctuation_attack';
this will use the ' and - punctuation symbol to attack MR on bert
python classification_test.py --recipe 'pso_paa';
will use paa as a multi-level attack together with SememePSO



To plot the mr results in Classification_tests run:
python plot_classification_table.py


For mnli entailment on bert
in Entailment_Test
python entailment.py --recipe 'punctuation_attack';
python plot_entialment_test.py

For question answering on bert
in Question_Answering_Test
python question_answering.py --recipe 'punctuation_attack';
python plot_question_answering.py