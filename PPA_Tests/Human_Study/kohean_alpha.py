import numpy as np
import pandas as pd
import sys




def calculate_k_a(M1,M2,column_names):

    M1 = [str(i) for i in M1]
    M2 = [str(i) for i in M2]
    column_names_T = [[i] for i in column_names]


    max = {}



    for i in column_names:
        for j in column_names_T:
            max[i+j[0]] = 0

    N = len(M1)
    for i in range(N):
        max[M1[i]+M2[i]] +=1
        max[M2[i]+M1[i]] +=1
    N=N*2

    max_as_matrix = []
    for i in column_names:
        row = []
        for j in column_names_T:
            row.append(max[i+j[0]])
        max_as_matrix.append(row)

    coincidence_matrx = np.array(max_as_matrix)
    trace = np.trace(coincidence_matrx)
    row_sum = np.sum(coincidence_matrx,axis=1)
    row_sum_extended = [ i*(i-1) for i in row_sum]
    row_sum_extended = sum(row_sum_extended)
    try:
        K_a = ((N-1)*trace - row_sum_extended )/(N*(N-1) - row_sum_extended)
    except Exception as e:
        print (e, 'K_a cant be calculated')
        return None
    return K_a

orig_gram_ad =[5,4,4,4,4,4,4,4,4,5,5,5,4,4,4,4,4,5,3,4,4,5,4,5,4,5,4,5,5,4,4,4,5,4,3,5,5,5,5,5,5,5,4,5,4,3,1,4,4,5,5,4,4,4,2,4,5,2,4,5,5,4,4,3,5,4,4,3,5,2,4,2,4,5,4,4,4,4,3,3,4,5,4,5,4,4,5,4,4,4,4,3,4,4,5,4,4,5,4,5]
orig_gram_pu =[5,5,5,4,5,4,5,4,5,5,5,5,5,5,5,5,5,4,3,5,4,5,5,5,5,4,4,5,5,5,5,5,4,4,3,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,5,5,5,4,5,5,4,5,5,4,5,5,4,5,5,5,4,5,5,4,3,4,4,3,5,5,5,5,5,5,5,4]


column_names = ['1','2','3','4','5']

K_a_original = calculate_k_a(orig_gram_ad,orig_gram_pu,column_names)
print ('original',K_a_original)
mean_orig_ad = np.mean(np.array(orig_gram_ad))
mean_orig_pu = np.mean(np.array(orig_gram_pu))
print ('pu mean',mean_orig_pu,'ad mean',mean_orig_ad,'both mean',np.mean(np.array([mean_orig_pu,mean_orig_ad])))



adv_gram_ad = [4,3,4,4,4,3,4,4,3,4,3,3,3,4,2,4,3,4,3,4,3,4,3,4,3,3,4,4,4,4,4,4,3,4,3,4,4,2,4,4,4,4,4,4,3,3,1,2,4,3,3,4,2,3,1,4,4,2,1,3,4,4,4,3,4,4,4,2,4,2,2,2,4,3,3,3,3,3,2,1,2,3,3,4,3,3,3,3,2,2,4,2,3,4,2,3,1,3,4,3]
adv_gram_pu = [4,3,4,3,4,1,2,4,3,4,3,1,2,3,1,3,3,3,2,3,1,4,2,4,3,3,3,4,4,4,4,5,2,3,2,4,2,3,4,3,4,3,4,4,3,4,4,1,4,2,2,4,2,3,2,3,3,4,4,3,4,4,4,4,3,4,3,1,4,3,3,4,4,3,2,2,3,3,2,3,3,4,4,4,4,3,3,3,1,2,3,3,4,4,3,4,2,3,3,3]

column_names = ['1','2','3','4','5']
K_a_adv = calculate_k_a(adv_gram_ad,adv_gram_pu,column_names)
print ('adv',K_a_adv)
mean_adv_ad = np.mean(np.array(adv_gram_ad))
mean_adv_pu = np.mean(np.array(adv_gram_pu))
print ('pu mean',mean_adv_pu,'ad mean',mean_adv_ad,'both mean', np.mean(np.array([mean_adv_pu,mean_adv_ad])))
 # classification
pu_class = [0,1,1,1,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,1,0,0,1,1,1,1,1,0,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,1]
ad_class = [0,1,1,1,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,1,0,0,1,1,1,1,1,0,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,1,1,1]
column_names = ['0','1']
K_a_class = calculate_k_a(ad_class,pu_class,column_names)
print ('classication K_a',K_a_class)


ad_sem_sim = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
pu_sem_sim = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
column_names = ['0','0.5','1']
K_a_sem_sim = calculate_k_a(ad_sem_sim,pu_sem_sim,column_names)
print ('sem sim K_a',K_a_sem_sim,'if nan = 1')
