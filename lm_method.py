from sklearn import linear_model
import numpy as np
#import csv
#import matplotlib.pyplot as plt
#from numpy import savetxt

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

    dlist=[]
    for i in range(len(lenlist)):
        l=float(lenlist[i])
        w=float(widlist[i])
        d=(l**2+w**2)**0.5
        s=l+w
        # templ=[l,w,d,s]
        # lwlist.append(templ)
        area.append(float(originarea[i]))
        dlist.append(d)

    dlist=np.array(dlist)
    # table=np.array(lwlist)
    # table=np.array(dlist)
    area=np.array(area)
    # print(table.shape,area.shape)
    # print(table)
    # M=np.mean(table.T,axis=1)
    # print('M',M,'M shape',M.shape)
    # C=table-M
    # # print('C',C)
    # V=np.corrcoef(C.T)
    # std_table=C/np.std(table.T,axis=1)
    # tablestd=np.std(table.T,axis=1)
    # print('Standarddeviation',std_table)
    # eigvalues,eigvectors=np.linalg.eig(V)
    # print('eigvalues',eigvalues)
    # print('eigvectors',eigvectors)
    #
    # pcabands=np.zeros(table.shape)
    # for i in range(table.shape[1]):
    #     pcn=eigvectors[:,i]
    #     pcnband=np.dot(std_table,pcn)
    #     pcabands[:,i]=pcabands[:,i]+pcnband
    # # savetxt('python-pcas.csv',pcabands,delimiter=',')
    # print('pcabands',pcabands)
    regr=linear_model.LinearRegression()
    # temp=pcabands[:,0]
    # regr.fit(pcabands,area)
    # regr.fit(area.reshape(-1,1),temp)
    regr.fit(area.reshape(-1,1),dlist)
    # print('coef',regr.coef_,regr.coef_.shape,pcabands.shape,'intercept',regr.intercept_)
    print('coef',regr.coef_,regr.coef_.shape,'intercept',regr.intercept_)
    # residual=area-np.matmul(pcabands,regr.coef_)
    # residual=pcabands[:,0]-np.matmul(area.reshape(-1,1),regr.coef_)-regr.intercept_
    residual=-(dlist-np.matmul(area.reshape(-1,1),regr.coef_)-regr.intercept_)
    print(residual,residual.shape)
    print('residual sum',np.sum(residual))

    # plt.scatter(area,residual)
    # plt.show()
    # name='python-res.csv'
    # savetxt(name,residual,delimiter=',')

    if all==False:
        return dlist,area
    else:
        # return residual,area,M,tablestd,eigvectors,regr.coef_,regr.intercept_
        return dlist,area,regr.coef_,regr.intercept_

# def lm_method_fit(lenlist,widlist,originarea,M,tablestd,eigvectors,coef,intercept):
def lm_method_fit(lenlist,widlist,originarea,coef,intercept):
    lwlist=[]
    area=[]

    dlist=[]
    for i in range(len(lenlist)):
        l=float(lenlist[i])
        w=float(widlist[i])
        d=(l**2+w**2)**0.5
        s=l+w
        templ=[l,w,d,s]
        lwlist.append(templ)
        area.append(float(originarea[i]))
        dlist.append(d)

    dlist=np.array(dlist)
    # table=np.array(lwlist)
    area=np.array(area)
    # print(table.shape,area.shape)
    # print(table)
    # # M=np.mean(table.T,axis=1)
    # print('M',M,'M shape',M.shape)
    # C=table-M
    # # print('C',C)
    # # V=np.corrcoef(C.T)
    # std_table=C/tablestd
    # # tablestd=np.std(table.T,axis=1)
    # print('Standarddeviation',std_table)
    # # eigvalues,eigvectors=np.linalg.eig(V)
    # # print('eigvalues',eigvalues)
    # print('eigvectors',eigvectors)
    #
    # pcabands=np.zeros(table.shape)
    # for i in range(table.shape[1]):
    #     pcn=eigvectors[:,i]
    #     pcnband=np.dot(std_table,pcn)
    #     pcabands[:,i]=pcabands[:,i]+pcnband
    # # savetxt('python-pcas.csv',pcabands,delimiter=',')
    # print('pcabands',pcabands)
    # regr=linear_model.LinearRegression()
    # temp=pcabands[:,0]
    # regr.fit(pcabands,area)
    # regr.fit(area.reshape(-1,1),temp)
    # regr.fit(area.reshape(-1,1),dlist)
    # print('coef',regr.coef_,regr.coef_.shape,pcabands.shape,'intercept',regr.intercept_)
    # residual=area-np.matmul(pcabands,regr.coef_)
    # residual=pcabands[:,0]-np.matmul(area.reshape(-1,1),coef)-intercept
    residual=-(dlist-np.matmul(area.reshape(-1,1),coef)-intercept)
    print(residual,residual.shape)
    return dlist,area

    # plt.scatter(area,residual)
    # plt.show()
    # name='python-res.csv'
    # savetxt(name,residual,delimiter=',')

    # if all==False:
    #     return residual,area
    # else:
    #     return residual,area,M,tablestd,eigvectors,regr.coef_


def getpcs(pc1,lenlist,widlist,x):
    lwlist=[]
    for i in range(len(lenlist)):
        l=float(lenlist[i])
        w=float(widlist[i])
        d=(l**2+w**2)**0.5
        s=l+w
        templ=[l,w,d,s]
        lwlist.append(templ)
    table=np.array(lwlist)
    M=np.mean(table.T,axis=1)
    C=table-M
    std_table=C/np.std(table.T,axis=1)
    xind=lwlist.index(x)
    vector=std_table[xind,:]
    pc=np.dot(vector,pc1)
    return pc


