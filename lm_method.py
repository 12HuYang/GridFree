from sklearn import linear_model
import numpy as np
import csv
import matplotlib.pyplot as plt

# filename='spore balls-outputdata by PC3.csv'

def lm_method(lenlist,widlist,originarea,all=False):

# lwlist=[]
# area=[]
# with open(filename) as f:
#     readcsv=csv.reader(f,delimiter=',')
#     for row in readcsv:
#         # print(row)
#         if row[0]=='Index':
#             continue
#         a=float(row[2])
#         l=float(row[3])
#         w=float(row[4])
#         d=(l**2+w**2)**0.5
#         s=l+w
#         templ=[l,w,d,s]
#         # print(templ)
#         lwlist.append(templ)
#         area.append(a)
    lwlist=[]
    area=[]
    for i in range(len(lenlist)):
        l=float(lenlist[i])
        w=float(widlist[i])
        d=(l**2+w**2)**0.5
        s=l+w
        templ=[l,w,d,s]
        lwlist.append(templ)
        area.append(float(originarea[i]))

    table=np.array(lwlist)
    area=np.array(area)
    print(table.shape,area.shape)
    print(table)
    M=np.mean(table.T,axis=1)
    print('M',M,'M shape',M.shape)
    C=table-M
    # print('C',C)
    V=np.corrcoef(C.T)
    std_table=C/np.std(table.T,axis=1)
    tablestd=np.std(table.T,axis=1)
    print('Standarddeviation',std_table)
    eigvalues,eigvectors=np.linalg.eig(V)
    print('eigvalues',eigvalues)
    print('eigvectors',eigvectors)

    pcabands=np.zeros(table.shape)
    for i in range(table.shape[1]):
        pcn=eigvectors[:,i]
        pcnband=np.dot(std_table,pcn)
        pcabands[:,i]=pcabands[:,i]+pcnband

    print('pcabands',pcabands)
    regr=linear_model.LinearRegression()
    regr.fit(pcabands,area)
    print('coef',regr.coef_,regr.coef_.shape,pcabands.shape,'intercept',regr.intercept_)
    residual=area-np.matmul(pcabands,regr.coef_)
    print(residual,residual.shape)

    # plt.scatter(area,residual)
    # plt.show()
    if all==False:
        return residual,area
    else:
        return residual,area,tablestd,eigvectors,regr.coef_



