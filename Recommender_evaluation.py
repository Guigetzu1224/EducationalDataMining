# CODE 2
import scipy.stats as st
import random
from random import sample
import numpy as np
import pandas as pd
import statistics
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import math
from tqdm import tqdm
import functions
import re
import matplotlib
import matplotlib.pyplot as plt
from operator import itemgetter
import sim2 as s
#-----
# DECLARING CONSTANTS
#-----
MIN_CLASS = 7
TANH_FACTOR = 0.9
NO_VOTES=10
TOP_N =5
MAJOR_MIN = 50 
UNDER_SAMPLE = 50
POWER = 1.6
#----
core = list(pd.read_csv('s_core.csv',names='A')['A'])
data_sparse = pd.read_csv('data_sparse.csv')
Data = pd.read_csv('Data.csv')
stdnt_perform = pd.read_csv('stdnt_perform.csv')
data_sparse = data_sparse.set_index(data_sparse.columns[0])
Data = Data.set_index('SID')
stdnt_perform = stdnt_perform.set_index(stdnt_perform.columns[0])
non_core = []
years = [[],[],[],[],[]]
bad_stdnt = []

def get_major(sid):
    return Data.loc[sid,'GRA_MajorAtGraduation'].values[0]

majors = []
major_pair = {}
lst_stdnts = data_sparse.index
for sids in lst_stdnts:
    if get_major(sids) not in major_pair:
        major_pair[get_major(sids)]=[]
    major_pair[get_major(sids)].append((sids,stdnt_perform.loc[sids].values[0]))


list_of_majors = list(Data['GRA_MajorAtGraduation'].value_counts().index)
average_distances = {}
for mjr_temp in list_of_majors:
    average_distances[mjr_temp]={}
distance = 1 - cosine_similarity(data_sparse)
k=0
results ={}
grades = {}
# Loop through the similarities let them vote 
for row in tqdm(distance):
    row = np.delete(row,k) 
    stdnt = lst_stdnts[k]
    ac_mjr = get_major(stdnt)
    majors.append(ac_mjr)
    stdnts = lst_stdnts.copy()
    stdnts = stdnts[stdnts!=stdnt]
    grds = []
    '''
    for stdnt in stdnts:
        grds.append(stdnt_perform.loc[stdnt][0])
    mjr_list = []
    temp_sims = row.copy()
    for temp_stdnt in stdnts:
        mjr_list.append(get_major(temp_stdnt))
    dframe = pd.DataFrame({'SID':stdnts,'Major':mjr_list,'Grades':grds,'Distance':row})
    majors_dframe = dframe['Major'].unique()
    s_pfrm = {}
    for major_dframe in majors_dframe:
        temp_dframe = dframe[dframe['Major'] == major_dframe]
        temp_dframe = temp_dframe.sort_values(by=['Distance'],ascending=True)
        closest_n = sum(temp_dframe['Distance'][:top_n_ppl]/top_n_ppl)
        avg_grade = sum(temp_dframe['Grades'][:top_n_ppl])/top_n_ppl
        if major_dframe not in s_pfrm:
            s_pfrm[major_dframe]=0
        s_pfrm[major_dframe]=(closest_n,avg_grade)
    sorted_final_votes = sorted(s_pfrm.items(),key=lambda x:x[1][0])
    results[stdnt] = (sorted_final_votes,get_major(stdnt),stdnt_perform.loc[stdnt][0])
    for k,grade in enumerate(sorted_final_votes):
        if str(k) not in grades:
            grades[str(k)] = []
        grades[str(k)].append(grade[1][1])
    for key,item in grades.items():
        print(f"{key}:{sum(item)/len(item)}",end=',')
    print("\n")
    k+=1
    '''
    most_similar = np.argsort(row)
    row = row[most_similar][:NO_VOTES]
    stdnts = lst_stdnts[most_similar][:NO_VOTES]
    s_pfrm = {}
    final_votes = {}
    for i in stdnts:
        s_pfrm[i] = stdnt_perform.loc[i][0]
    for dist,t_stdnt in zip(row,stdnts):
        ac_majr = get_major(t_stdnt)
        if ac_majr not in final_votes:
            final_votes[ac_majr]=[]
        final_votes[ac_majr].append(dist*(1-np.tanh(s_pfrm[t_stdnt]*TANH_FACTOR)))
#        final_votes[ac_majr].append(dist)    
    for key,value in final_votes.items():
        final_votes[key]=np.sum(value)/(len(value)**POWER)
    for key,value in final_votes.items():
        if key not in average_distances[get_major(stdnt)]:
            average_distances[get_major(stdnt)][key] = [0,0]
        average_distances[get_major(stdnt)][key][0] += value
        average_distances[get_major(stdnt)][key][1] +=1
    sorted_final_votes = dict(sorted(final_votes.items(),key=itemgetter(1)))
    k+=1
    results[stdnt]=(sorted_final_votes,get_major(stdnt),stdnt_perform.loc[stdnt][0])

#Standard Generate A Confusion Matrix 
c_c = []
c_w = []
w_c = []
w_w = []
confusion_matrix = pd.DataFrame(np.zeros((len(list_of_majors),len(list_of_majors))),index=list_of_majors,columns=list_of_majors,dtype='int')
s_right = []
s_wrong = []
majors = list(set(majors))
# Get the confusion matrix
for key,tpl in results.items():
    if tpl[1] in list(tpl[0].keys())[:TOP_N]:
        confusion_matrix.loc[tpl[1],tpl[1]]+=1
        s_right.append(tpl[2])
        if tpl[2]>0:
            c_c.append(tpl[2])
        if tpl[2]<0:
            c_w.append(tpl[2])
    else:
        confusion_matrix.loc[tpl[1],list(tpl[0].keys())[0]] += 1
        s_wrong.append(tpl[2])
        if tpl[2]>0:
            w_c.append(tpl[2])
        else:
            w_w.append(tpl[2])

k=0
tf = [0,0]
print('-'*20)
for index,row in confusion_matrix.iterrows():
    precision = row[k]/np.sum(row)
    try:
        recall = confusion_matrix[index][k]/np.sum(confusion_matrix[index])
    except:
        recall = 0
    try:
        precision = row[k]/np.sum(row)
    except:
        precision =0 
    try:
        if precision == 0 or recall == 0:
            f = 0
        else:
            f = 2*(precision*recall)/(precision+recall)
    except:
        f = 0
    print(f"{index} | Precision: {precision:.3f} | Recall: {recall:.3f} | Support: {np.sum(row)}")
    k+=1
    tf[0]+=f
    tf[1]+=1
print(tf)
tf=tf[0]/tf[1]
c_c,c_w,w_c,w_w = np.array(c_c),np.array(c_w),np.array(w_c),np.array(w_w)
pst = len(c_c)+len(c_w)+len(w_w)+len(w_c)
x = np.array(confusion_matrix)
print('-'*20)
print(f"Accuracy:{x.trace()/x.sum():.3f}\nRecommended & Did well Count: {len(c_c)} | Predition Mean Performance:{c_c.mean():.3f} | Std. Dev:{c_c.std():.3f} \n Wasn't Recommended & did Good Count: {len(w_c)} | Prediction Mean Performance: {w_c.mean():.3f} | Std. Dev: {w_c.std():.3f}")
print('-'*20)
print(f"Recommended & Did badly Count: {len(c_w)} |Prediction Mean Performance:{c_w.mean():.3f} | Std. Dev: {c_w.std():.3f}\nWasn't recommended & did badly Count: {len(w_w)} | Prediction Mean Performance: {w_w.mean():.3f} | Std. Dev:{w_w.std():.3f}")

#Pai$ed Z Test
s_right = np.array(s_right)
s_wrong = np.array(s_wrong)
s_r_mean = s_right.mean()
s_w_mean = s_wrong.mean()
s_r_std = s_right.std()
s_w_std = s_wrong.std()
s_r_n = len(s_right)
s_w_n = len(s_wrong)

Z_value = (s_r_mean-s_w_mean)/((s_w_std**2)+(s_r_std**2))**0.5
print(f"F Score = {tf:.3f}")
print(f"Accuracy: {x.trace()/x.sum():.3f}")
print(f"Z Value = {Z_value}")
print(f"G-to-B  {len(c_c)/(len(c_c)+len(c_w)):.3f}")
print(f"NR-G-to-B {len(w_c)/(len(w_c)+len(w_w)):.3f}")    
print(average_distances)
