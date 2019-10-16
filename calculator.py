import numpy as np
from tkinter import messagebox


def init(RGB,Infrared,Height,equation,nodataposition):
    express="".join(equation)
    RGBBand1=RGB.bands[0]
    RGBBand2=RGB.bands[1]
    RGBBand3=RGB.bands[2]
    InfraredBand1=Infrared.bands[0]
    InfraredBand2=Infrared.bands[1]
    InfraredBand3=Infrared.bands[2]
    if Height is not None:
        HeightBand=Height.bands[0]
    try:
        res=eval(express)
        #res[nodataposition]=1e-6
        print(res)
        return res
    except TypeError:
        messagebox.showerror('Invalid Equation',message='Equation is invalid. Please check your equation.')
        return None

