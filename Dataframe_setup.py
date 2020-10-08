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
import sim2 as s
#matplotlib.use('TkAgg')  # Or any other X11 back-end
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# -----CONSTANTS-----:
# The following is a list of constants relating to the parameters used in the Dataset, as well as the parameters listed in the paper
CRS_CD_INDX = 1 #Index of REG_CourseCrn in the dataframe
MAJ_GRD_INDX = 4 #Index of Major at graduation
DEG_TYPE = 5 #Index of the degree type at graduation
GRADE_INDX = 7
GRADE_PRE_INDX = 6
YEARS_TO_SKIP = 4
MIN_GRADE_SUPPORT = 10
TEST_SIZE = 0.2
FAIL_FLOAT=0.33
MIN_CLASS =5
NO_VOTES=10
TOP_N =3
MAJOR_MIN = 0
CUT_OFF = 0.90
#-----
irrel_courses = ['REG_Programcode','REG_CourseCrn','REG_crsSchool','REG_REG_credHr',
                                'REG_banTerm','REG_term','CRS_crsCampus','CRS_schdtyp','FID','CRS_contact_hrs',
                                'CRS_XLSTGRP','CRS_PrimarySect','CRS_enrolltally','STU_ulevel','STU_DegreeSeek',
                                'STU_credstua','STU_G_transfer','GRA_Grad_Term','GRA_Major2atGraduation','GRA_MinorAtGraduation',
                                'GRA_Minor2AtGraduation','GRA_Conc2atGraduation','GRA_ConcAtGraduation','OTCM_FinalGradeC','OTCM_Crs_Graded']

# ------LOADING THE DATA------
Data = pd.read_csv('final-datamart-6-7-19.csv')
Data = Data.replace(' ',0) # Null values were simply not entered, so we replace them with 0. 
Data = Data[Data['REG_crsSchool']=='Fordham College/Rose Hill' ]  #Drop non rose hill 
core = list(pd.read_csv('s_core.csv',names='Core_Courses')['Core_Courses'])

# ------DROPPING IRRELEVANT COURSES AND STUDENTS WITH GRADUATION PRE2014------------------
Data = Data.astype({Data.columns[21]:'float32'})
for j in tqdm(range(YEARS_TO_SKIP)):
    Compar = int("201"+str(j+1)+"20")
    Data = Data[(Data.iloc[:,21]%Compar).astype(bool)]
Data = Data.drop(irrel_courses,axis=1)

# COLUMNS ARE ['SID', 'REG_Numbercode', 'REG_classSize', 'CRS_coursetitle', 'GRA_MajorAtGraduation', 'GRA_degreeatGraduation', 'OTCM_FinalGradeN'] CURRENTLY PRESENT IN DATAFRAME

# DROPPING MAJORS THAT ARNT RELEVANT
non_ex_classes = ['Organizational Leadership-Westchester','NATURAL SCI/INTERDISC','German Studies','Business','Accounting','Information Systems','MEDIA STUDIES','COMPU SYS/MGMT APP','BUSINESS ADMIN','Individualized Major','FINANCE','Applied Accounting and Finance','MARKETING']
for cls in non_ex_classes:
    idxs = Data.iloc[:,MAJ_GRD_INDX]==cls
    idx = [not i for i in idxs]
    Data = Data[idx]

# ----------------- GENERATING SPARSE MATRIX -----------------------
Classes,Students,arr_to_drop= set(),set(),set()
# From here, we are going to find out which students we are going to keep based on the majors they are taking
temp_data = Data.copy()
temp_data = temp_data.drop_duplicates('SID')
value_counts = temp_data['GRA_MajorAtGraduation'].value_counts()
total_students = sum(list(value_counts))
break_point = int(total_students*CUT_OFF)
net_current = 0
for i in tqdm(range(len(value_counts))):
    net_current += int(list(value_counts)[i])
    if net_current > break_point:
        stop = i-1
        break

bad_classes = value_counts.index[stop+1:]
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
    if int(str(Data.iloc[i,1])[0]) > 2:
        continue
    Classes.add(str(Data.iloc[i,1])+str(Data.iloc[i,3]))
    Students.add(Data.iloc[i,0])
counts = {}
classes = list(set(Classes))
students = list(set(Students))
# Here we take hte set of students and set of classes, and generate a large sparse matrix
data_sparse = pd.DataFrame(0, index = students, columns=Classes,dtype='float64')
# ---- CHANGING THE STUDENT VECTOR POINTS TO GRADES INSTEAD OF ZEROS --------
for  i in tqdm(range(len(Data))):
    Class = str(Data.iloc[i,1])+str(Data.iloc[i,3])
    if Class not in data_sparse.columns:
        continue
    Stdnt = Data.iloc[i,0]
    if float(Data.iloc[i,6] == 0) :
        data_sparse.loc[Stdnt,Class] = FAIL_FLOAT
    else:
        data_sparse.loc[Stdnt,Class] = Data.iloc[i,6]
grade_support = []
bad_class = []
for i in tqdm(range(len(data_sparse.columns))):
    valid_grades = []
    grade_support = data_sparse.iloc[:,i].astype(bool).sum(axis=0)
print(data_sparse.shape)

bad_stdnt = []
Data = Data.set_index('SID')
for sid in data_sparse.index:
    try:
        majors = Data.loc[sid]['GRA_MajorAtGraduation'].value_counts()
        if len(majors) > 1:
            bad_stdnt.append(sid)
    except:
        pass

data_sparse = data_sparse[(data_sparse.sum(axis=1).astype(bool))] # Any non classes 
data_sparse = data_sparse.drop(bad_stdnt)
core = list(pd.read_csv('s_core.csv',names='A')['A'])
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

# SETTING TH EYEARS TO DROP AND NON CORE
to_drop = years[4].copy()
to_drop.extend(years[3])
to_drop.extend(years[2])
#to_drop.extend(years[1])
#to_drop.extend(years[0])
to_drop.extend(non_core)
to_drop = set(to_drop)
data_sparse = data_sparse.drop(to_drop,axis=1)
bad_stdnt = []
for index,row in tqdm(data_sparse.iterrows()):
    class_taken = row.astype(bool).sum()
    if class_taken < MIN_CLASS:
        bad_stdnt.append(index)
data_sparse = data_sparse.drop(bad_stdnt)

Data = Data.drop(bad_stdnt)
GPA = []
GPA_stdnt = []
for index,row in tqdm(data_sparse.iterrows()):
    average = [0,0]
    for item in row:
        if float(item) > 0:
            average[0] += item
            average[1] += 1
    GPA.append(float(average[0])/float(average[1]))
    GPA_stdnt.append(index)
GPA= np.array(GPA)
mGPA,sGPA = np.mean(GPA),np.std(GPA)
#nGPA = (GPA-mGPA)/sGPA
nGPA = GPA
stdnt_perform = pd.DataFrame({'A':nGPA},index=GPA_stdnt) # Standardized student performance 
print(data_sparse.shape)
print(stdnt_perform[stdnt_perform['A']>=0])
print(len(stdnt_perform))
#making the csv's
stdnt_perform.to_csv('stdnt_perform.csv')
data_sparse.to_csv('data_sparse.csv')
Data.to_csv('Data.csv')

