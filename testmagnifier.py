import tkinter
from PIL import ImageGrab,ImageTk
import time
import ctypes
root=tkinter.Tk()
screenWidth=root.winfo_screenwidth()
screenHeight=root.winfo_screenheight()
root.geometry(str(screenWidth)+'x'+str(screenHeight)+'+0+0')
root.overrideredirect(True)

root.resizable(False,False)

canvas=tkinter.Canvas(root,bg='white',width=screenWidth,height=screenHeight)
image=ImageTk.PhotoImage(ImageGrab.grab())
canvas.create_image(screenWidth//2,screenHeight//2,image=image)

def onMouseRightClick(event):
    global root
    root.destroy()
#canvas.bind('<Button-3>',onMouseRightClick)

radius=20
def onMouseMove(event):
    global canvas
    ctypes
    x=event.x
    y=event.y
    print(x,y)
    subIm=ImageGrab.grab((x-radius,y-radius,x+radius,y+radius))
    subIm=subIm.resize((radius*3,radius*3))
    subIm=ImageTk.PhotoImage(subIm)
    #mag=tkinter.Canvas(root,bg='white',width=radius*3,height=radius*3)
    #mag.pack()
    canvas.create_image(x-70,y-70,image=subIm)
    canvas.update()
    time.sleep(0.5)

canvas.bind('<Motion>',onMouseMove)

canvas.pack(fill=tkinter.BOTH,expand=tkinter.YES)

root.mainloop()