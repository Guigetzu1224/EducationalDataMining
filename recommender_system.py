import numpy as np
import pandas as pd
import statistics
from sklearn.metrics.pairwise import cosine_similarity
import math
from tqdm import tqdm
import functions
import re
import matplotlib
import matplotlib.pyplot as plt
from operator import itemgetter

#matplotlib.use('TkAgg')  # Or any other X11 back-end
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# -----CONSTANTS-----:
CRS_CD_INDX = 1 #Index of REG_CourseCrn in the dataframe
MAJ_GRD_INDX = 4 #Index of Major at graduation
DEG_TYPE = 5 #Index of the degree type at graduation
GRADE_INDX = 7
GRADE_PRE_INDX = 6
YEARS_TO_SKIP = 4
MIN_GRADE_SUPPORT = 10
VOTER_COUNT = 100
TEST_SIZE = 0.2
FAIL_FLOAT=0.33
MIN_CLASS = 7
NO_VOTES=10
TOP_N = 3
MAJOR_MIN = 50 
irrel_courses = ['REG_Programcode','REG_CourseCrn','REG_crsSchool','REG_REG_credHr',
                                'REG_banTerm','REG_term','CRS_crsCampus','CRS_schdtyp','FID','CRS_contact_hrs',
                                'CRS_XLSTGRP','CRS_PrimarySect','CRS_enrolltally','STU_ulevel','STU_DegreeSeek',
                                'STU_credstua','STU_G_transfer','GRA_Grad_Term','GRA_Major2atGraduation','GRA_MinorAtGraduation',
                                'GRA_Minor2AtGraduation','GRA_Conc2atGraduation','GRA_ConcAtGraduation','OTCM_FinalGradeC','OTCM_Crs_Graded']

# ------LOADING THE DATA------
Data = pd.read_csv('final-datamart-6-7-19.csv')
#Data = pd.read_csv('cs-sample-6-7-19.csv') # Smaller data set for inital testing
Data = Data.replace(' ',0) # Null values were simply not entered
#Drop non rose hill
Data = Data[Data['REG_crsSchool']=='Fordham College/Rose Hill' ]   
core = list(pd.read_csv('s_core.csv',names='A')['A'])
# ------DROPPING IRRELEVANT COURSES AND STUDENTS WITH GRADUATION PRE2014------------------
Data = Data.astype({Data.columns[21]:'int64'})
for j in tqdm(range(YEARS_TO_SKIP)):
    Compar = int("201"+str(j+1)+"20")
    Data = Data[(Data.iloc[:,21]%Compar).astype(bool)]
Data = Data.drop(irrel_courses,axis=1)
# COLUMNS ARE ['SID', 'REG_Numbercode', 'REG_classSize', 'CRS_coursetitle', 'GRA_MajorAtGraduation', 'GRA_degreeatGraduation', 'OTCM_FinalGradeN']
# ----------------- GENERATING SPARSE MATRIX -----------------------
Classes,Students,arr_to_drop= set(),set(),set()
temp_data = Data.drop_duplicates('SID')
value_counts = temp_data['GRA_MajorAtGraduation'].value_counts()
for i in tqdm(range(len(value_counts))):
    if int(list(value_counts)[i]) < MAJOR_MIN:
        stop = i-1
        break
bad_classes = value_counts.index[i+1:]

Data = Data.astype({Data.columns[GRADE_PRE_INDX]:'float64'})
Data = Data[(Data.iloc[:,GRADE_PRE_INDX]).astype(bool)] 
# Change it such that if it is a 0, then its a fail 
for cls in bad_classes:
    idxs = Data.iloc[:,MAJ_GRD_INDX]==cls
    idx = [not i for i in idxs]
    Data = Data[idx]
for i in tqdm(range(0,len(Data))):
    if Data.iloc[i,MAJ_GRD_INDX] in bad_classes:
        arr_to_drop.add(i)
    Classes.add(str(Data.iloc[i,1])+str(Data.iloc[i,3]))
    Students.add(Data.iloc[i,0])
counts = {}
classes = list(Classes)
students = list(Students)
data_sparse = pd.DataFrame(0, index = students, columns=Classes,dtype='float64')
# ---- CHANGING THE STUDENT VECTOR POINTS TO GRADES INSTEAD OF ZEROS --------
for  i in tqdm(range(len(Data))):
    Class = str(Data.iloc[i,1])+str(Data.iloc[i,3])
    Stdnt = Data.iloc[i,0]
    if float(Data.iloc[i,6] == 0) :
        data_sparse.loc[Stdnt,Class] = FAIL_FLOAT
    else:
        data_sparse.loc[Stdnt,Class] = float(Data.iloc[i,6])
grade_support = []
bad_class = []
for i in tqdm(range(len(data_sparse.columns))):
    al_grd = data_sparse.iloc[:,i]
    valid_grades = []
    grade_support = al_grd.astype(bool).sum(axis=0)
non_core = []
years = [[],[],[],[],[]]
for i in data_sparse.columns:
    los = int(i[0])
    if los < 5:
        years[los-1].append(i)
    else:
        years[4].append(i)
    if i in core:
        pass
    else:
        non_core.append(i)


to_drop = years[4].copy()
to_drop.extend(years[3])
to_drop.extend(non_core)
to_drop = set(to_drop)
data_sparse = data_sparse.drop(to_drop,axis=1)
to_remove = []
data_sparse = data_sparse[(data_sparse.sum(axis=1).astype(bool))] # Any non classes 
bad_stdnt = []

for index,row in tqdm(data_sparse.iterrows()):
    class_taken = 0
    for point in row:
        if point > 0:
            class_taken +=1
    if class_taken < MIN_CLASS:
        bad_stdnt.append(index)
data_sparse = data_sparse.drop(bad_stdnt)
#Generate Standardized Performance Vector 
GPA = []
GPA_stdnt = []
for index,row in tqdm(data_sparse.iterrows()):
    GPA.append(np.mean(row))
    GPA_stdnt.append(index)
GPA, mGPA, sGPA = np.array(GPA)
mGPA,sGPA = np.mean(GPA),np.std(GPA)
nGPA = (GPA-mGPA)/sGPA
stdnt_perform = pd.DataFrame({'A':nGPA},index=GPA_stdnt) # Standardized student performance 
Data = Data.drop_duplicates('SID').set_index('SID')

def get_major(sid):
    return Data.loc[sid][MAJ_GRD_INDX-1]

list_of_majors = list(Data['GRA_MajorAtGraduation'].value_counts().index)

distance = 1 - cosine_similarity(data_sparse)

k=0
lst_stdnts = data_sparse.index
results ={}
# Loop through the similarities let them vote 
for row in tqdm(distance):
    row = row[row>1e-15] 
    stdnt = lst_stdnts[k]
    ac_mjr = get_major(stdnt)
    stdnts = lst_stdnts.copy()
    stdnts = stdnts[stdnts!=stdnt]
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
        final_votes[ac_majr].append(dist*np.tanh(1-s_pfrm[t_stdnt]))
    for key,value in final_votes.items():
        final_votes[key]=sum(value)
    sorted_final_votes = dict(sorted(final_votes.items(),key=itemgetter(1)))
    k+=1
    results[stdnt]=(sorted_final_votes,get_major(stdnt),stdnt_perform.loc[stdnt][0])

c_c = []
c_w = []
w_c = []
w_w = []
confusion_matrix = pd.DataFrame(np.zeros((len(list_of_majors),len(list_of_majors))),index=list_of_majors,columns=list_of_majors,dtype='int')
# Get the confusion matrix
for key,tpl in results.items():
    if tpl[1] in list(tpl[0].keys())[:TOP_N]:
        confusion_matrix.loc[tpl[1],tpl[1]]+=1
        if tpl[2]>0:
            c_c.append(tpl[2])
        if tpl[2]<0:
            c_w.append(tpl[2])
    else:
        confusion_matrix.loc[tpl[1],list(tpl[0].keys())[0]] += 1
        if tpl[2]>0:
        else:
            w_w.append(tpl[2])

c_c,c_w,w_c,w_w = np.array(c_c),np.array(c_w),np.array(w_c),np.array(w_w)
pst = len(c_c)+len(c_w)+len(w_w)+len(w_c)

x = np.array(confusion_matrix)
print(f"Accuracy:{x.trace()/x.sum()}\nGood/Good Count: {len(c_c)} | Predition Mean Performance:{c_c.mean()} | Std. Dev:{c_c.std()} \nWrong/Good Count: {len(w_c)} | Prediction Mean Performance: {w_c.mean()} | Std. Dev: {w_c.std()}")
print(f"Good/Wrong Count: {len(c_w)} |Prediction Mean Performance:{c_w.mean()} | Std. Dev: {c_w.std()}\nWrong/Wrong Count: {len(w_w)} | Prediction Mean Performance: {w_w.mean()} | Std. Dev:{w_w.std()}")

