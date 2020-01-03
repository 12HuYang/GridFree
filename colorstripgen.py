import numpy as np
import matplotlib.pyplot as pyplt

colorstrip=np.zeros((15,35*10,3),'uint8')
for j in range(35):
    colorstrip[:,0*35+j]=[255,0,0]
for j in range(35):
    colorstrip[:,1*35+j]=[255,127,0]
for j in range(35):
    colorstrip[:,2*35+j]=[255,255,0]
for j in range(35):
    colorstrip[:,3*35+j]=[127,255,0]
for j in range(35):
    colorstrip[:,4*35+j]=[0,255,255]
for j in range(35):
    colorstrip[:,5*35+j]=[0,127,255]
for j in range(35):
    colorstrip[:,6*35+j]=[0,0,255]
for j in range(35):
    colorstrip[:,7*35+j]=[127,0,255]
for j in range(35):
    colorstrip[:,8*35+j]=[75,0,130]
for j in range(35):
    colorstrip[:,9*35+j]=[255,0,255]
pyplt.imsave('colortable.png',colorstrip)


