import numpy as np

def plotindice(mat_a,mat_b):
    import matplotlib.pyplot as plt
    for i in range(161,167):
        plt.subplot(i)
        r1=mat_a[:,i-161]
        r2=mat_b[:,i-161]
        plt.scatter(r1,r2,color='tab:red',label='R')
        plt.xlabel('RGB 255 indice'+str(i-161+1))
        if i==161:
            plt.ylabel('RGB float')
    plt.show()
    # plt.savefig('indice_float_int_1.png')
    for i in range(161,167):
        plt.subplot(i)
        r1=mat_a[:,i-161+6]
        r2=mat_b[:,i-161+6]
        plt.scatter(r1,r2,color='tab:red',label='R')
        plt.xlabel('RGB 255 indice'+str(i-161+1+6))
        if i==161:
            plt.ylabel('RGB float')
    plt.show()
    # plt.savefig('indice_float_int_2.png')



def colorindcal(mat,isfloat=False):
    Red=mat[:,0]
    Green=mat[:,1]
    Blue=mat[:,2]

    secondsmall_R=np.partition(Red,1)[1]
    print(secondsmall_R)

    Red=Red+1
    Green=Green+1
    Blue=Blue+1

    PAT_R=Red/(Red+Green)
    PAT_G=Green/(Green+Blue)
    PAT_B=Blue/(Blue+Red)
    print('Red',Red,'Red+Green',Red+Green)
    print('PAT_R',PAT_R.max(),PAT_R.min(),PAT_R)
    print('PAT_G',PAT_G.max(),PAT_G.min(),PAT_G)
    print('PAT_B',PAT_B.max(),PAT_B.min(),PAT_B)

    ROO_R=Red/Green
    ROO_G=Green/Blue
    ROO_B=Blue/Red

    DIF_R=2*Red-Green-Blue
    DIF_G=2*Green-Blue-Red
    DIF_B=2*Blue-Red-Green

    GLD_R=Red/(np.multiply(np.power(Blue,0.618),np.power(Green,0.382)))
    GLD_G=Green/(np.multiply(np.power(Blue,0.618),np.power(Red,0.382)))
    GLD_B=Blue/(np.multiply(np.power(Green,0.618),np.power(Red,0.382)))

    global colorindex_vector,colorindex_vector_float
    if isfloat==True:
        fillbands(colorindex_vector_float,0,PAT_R)
        fillbands(colorindex_vector_float,1,PAT_G)
        fillbands(colorindex_vector_float,2,PAT_B)
        fillbands(colorindex_vector_float,3,ROO_R)
        fillbands(colorindex_vector_float,4,ROO_G)
        fillbands(colorindex_vector_float,5,ROO_B)
        fillbands(colorindex_vector_float,6,DIF_R)
        fillbands(colorindex_vector_float,7,DIF_G)
        fillbands(colorindex_vector_float,8,DIF_B)
        fillbands(colorindex_vector_float,9,GLD_R)
        fillbands(colorindex_vector_float,10,GLD_G)
        fillbands(colorindex_vector_float,11,GLD_B)
    else:
        fillbands(colorindex_vector,0,PAT_R)
        fillbands(colorindex_vector,1,PAT_G)
        fillbands(colorindex_vector,2,PAT_B)
        fillbands(colorindex_vector,3,ROO_R)
        fillbands(colorindex_vector,4,ROO_G)
        fillbands(colorindex_vector,5,ROO_B)
        fillbands(colorindex_vector,6,DIF_R)
        fillbands(colorindex_vector,7,DIF_G)
        fillbands(colorindex_vector,8,DIF_B)
        fillbands(colorindex_vector,9,GLD_R)
        fillbands(colorindex_vector,10,GLD_G)
        fillbands(colorindex_vector,11,GLD_B)



def fillbands(vector,vecind,band):
    vector[:,vecind]=vector[:,vecind]+band
    return

def plotcorr(mat_a,mat_b):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt

    plt.subplot(131)

    r1=mat_a[:,0]
    r2=mat_b[:,0]

    plt.scatter(r1,r2,color='tab:red',label='R')
    plt.xlabel('RGB R 255')
    plt.ylabel('RGB float')

    plt.subplot(132)

    r1=mat_a[:,1]
    r2=mat_b[:,1]

    plt.scatter(r1,r2,color='tab:green',label='G')
    plt.xlabel('RGB G 255')

    plt.subplot(133)

    r1=mat_a[:,2]
    r2=mat_b[:,2]

    plt.scatter(r1,r2,color='tab:blue',label='B')
    plt.xlabel('RGB B 255')

    plt.savefig('RGB_float_int.png')

    # plt.show()

    # plt.show()
    #plt.savefig('3dplot_PC.png')

randrgb=np.random.randint(1,255,size=(1000,3))

randrgbfloat=randrgb/255

print(randrgb,randrgbfloat)
plotcorr(randrgb,randrgbfloat)

colorindex_vector=np.zeros((1000,12))
colorindex_vector_float=np.zeros((1000,12))

colorindcal(randrgb)
colorindcal(randrgbfloat,True)
plotindice(colorindex_vector,colorindex_vector_float)

np.savetxt('rgb_int.csv',randrgb,delimiter=',')
np.savetxt('rgb_float.csv',randrgbfloat,delimiter=',')
np.savetxt('indice_int.csv',colorindex_vector,delimiter=',')
np.savetxt('indice_float.csv',colorindex_vector_float,delimiter=',')

#integer pcs

rgb_M=np.mean(randrgb.T,axis=1)
colorindex_M=np.mean(colorindex_vector.T,axis=1)
print('rgb_M',rgb_M,'colorindex_M',colorindex_M)
rgb_C=randrgb-rgb_M
colorindex_C=colorindex_vector-colorindex_M
rgb_V=np.corrcoef(rgb_C.T)
color_V=np.corrcoef(colorindex_C.T)
rgb_std=rgb_C/np.std(randrgb.T,axis=1)
color_std=colorindex_C/np.std(colorindex_vector.T,axis=1)
rgb_eigval,rgb_eigvec=np.linalg.eig(rgb_V)
color_eigval,color_eigvec=np.linalg.eig(color_V)
print('rgb_eigvec',rgb_eigvec)
print('color_eigvec',color_eigvec)
featurechannel=12
pcabands=np.zeros((colorindex_vector.shape[0],featurechannel))
rgbbands=np.zeros((colorindex_vector.shape[0],3))
for i in range(3):
    pcn=rgb_eigvec[:,i]
    pcnbands=np.dot(rgb_std,pcn)
    pcvar=np.var(pcnbands)
    print('rgb pc',i+1,'var=',pcvar)
    # pcabands[:,i]=pcabands[:,i]+pcnbands
    rgbbands[:,i]=rgbbands[:,i]+pcnbands
# plot3d(pcabands)
np.savetxt('rgbint_pc.csv',rgbbands,delimiter=',')
#indexbands=np.zeros((colorindex_vector.shape[0],3))
for i in range(0,featurechannel):
    pcn=color_eigvec[:,i]
    pcnbands=np.dot(color_std,pcn)
    pcvar=np.var(pcnbands)
    print('color index pc',i+1,'var=',pcvar)
    pcabands[:,i]=pcabands[:,i]+pcnbands
    # if i<5:
    #     indexbands[:,i-2]=indexbands[:,i-2]+pcnbands
np.savetxt('colorindint_pc.csv',pcabands,delimiter=',')


rgb_M=np.mean(randrgbfloat.T,axis=1)
colorindex_M=np.mean(colorindex_vector_float.T,axis=1)
print('rgb_M',rgb_M,'colorindex_M',colorindex_M)
rgb_C=randrgbfloat-rgb_M
colorindex_C=colorindex_vector_float-colorindex_M
rgb_V=np.corrcoef(rgb_C.T)
color_V=np.corrcoef(colorindex_C.T)
rgb_std=rgb_C/np.std(randrgbfloat.T,axis=1)
color_std=colorindex_C/np.std(colorindex_vector_float.T,axis=1)
rgb_eigval,rgb_eigvec=np.linalg.eig(rgb_V)
color_eigval,color_eigvec=np.linalg.eig(color_V)
print('rgb_eigvec',rgb_eigvec)
print('color_eigvec',color_eigvec)
featurechannel=12
pcabands=np.zeros((colorindex_vector_float.shape[0],featurechannel))
rgbbands=np.zeros((colorindex_vector_float.shape[0],3))
for i in range(3):
    pcn=rgb_eigvec[:,i]
    pcnbands=np.dot(rgb_std,pcn)
    pcvar=np.var(pcnbands)
    print('rgb pc',i+1,'var=',pcvar)
    # pcabands[:,i]=pcabands[:,i]+pcnbands
    rgbbands[:,i]=rgbbands[:,i]+pcnbands
# plot3d(pcabands)
np.savetxt('rgbfloat_pc.csv',rgbbands,delimiter=',')
#indexbands=np.zeros((colorindex_vector.shape[0],3))
for i in range(0,featurechannel):
    pcn=color_eigvec[:,i]
    pcnbands=np.dot(color_std,pcn)
    pcvar=np.var(pcnbands)
    print('color index pc',i+1,'var=',pcvar)
    pcabands[:,i]=pcabands[:,i]+pcnbands
    # if i<5:
    #     indexbands[:,i-2]=indexbands[:,i-2]+pcnbands
np.savetxt('colorindfloat_pc.csv',pcabands,delimiter=',')