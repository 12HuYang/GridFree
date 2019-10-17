# A pixel-level label tools to process high-throughput images at pixel level.
## Implement Python version >= 3.6.5
<!--![screenshot](https://raw.githubusercontent.com/12HuYang/FreeCADITS/master/Training_intro.png)-->
![screenshot](https://raw.githubusercontent.com/12HuYang/GridFree/master/compare.png)
![screenshot](https://raw.githubusercontent.com/12HuYang/GridFree/master/normaldistribution.png)
<!--#### RUN ```pip3 install plantlabeller``` to install from terminal, if you cannot run ```pip3```, use ```python3 -m pip install plantlabeller```.
#### Linux user may need to use ```sudo pip3 install plantlabeller``` or ```sudo python3 -m pip install plantlabeller```.-->
#### UNIX OS 
* execute ```python3 -m pip install -r requirements.txt``` to install required packages
* execute ```./run``` on a terminal to run the software
#### Windows OS 
* install Python 3.6.5 or later, download link: https://www.python.org/downloads/release/python-365/
* run ```cmd``` go to the path you downloaded GridFree
* execute ```py -m pip install -r requirements.txt``` to install required packages
* execute ```py -m pip install opencv-python``` for opencv package
* execute ```py -m pip install scikit-image``` for package installation
* execute ```py -m pip install sklearn``` for package installation
* Windows 32bit OS execute ```py -m pip install rasterio‑1.0.24+gdal24‑cp36‑cp36m‑win32.whl```
* Windows 64 bit OS execute ```py -m pip install rasterio‑1.0.24+gdal24‑cp36‑cp36m‑win_amd64.whl```
* execute ```py tkinterGUI.py``` to run the software
#### dup1OUTPUT.tif 
dup1OUTPUT.tif is a sample filed image, download it to GridFree folder:
https://drive.google.com/file/d/1hZzEpsqDWq7yrXRgDWwbDmQCY2iGni3Z/view?usp=sharing


<!---#### ***GDAL instllation instruction:***
1. RUN ```pip3 install GDAL==2.4.2```
   - if failed with error: > gdal-config not found
   - go to step 2.
2. RUN ```brew install gdal```, go to step 1
   - if failed with "gcc" go to step 3
3. RUN ```brew reinstall gcc```, go to step 1--->
#### If crashed
Modify matplotlibrc file ADD: ```backend: TkAgg```
