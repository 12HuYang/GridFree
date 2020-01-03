from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import numpy as np

x,y=np.indices((80,80))
print('x,y',x,y)
x1,y1,x2,y2=28,28,30,30
r1,r2=5,5
mask_circle1=(x-x1)**2+(y-y1)**2 < r1**2
mask_circle2=(x-x2)**2+(y-y2)**2 < r2**2
mask_circle3=(x-26)**2+(y-26)**2 < 5**2
image=np.logical_or(mask_circle1,mask_circle2)
image=np.logical_or(image,mask_circle3)

distance=ndi.distance_transform_edt(image)
local_max=peak_local_max(distance,labels=image,footprint=np.ones((3,3)),indices=False)
markers=ndi.label(local_max)[0]
labels=watershed(-distance,markers,mask=image)
print('labels',labels)