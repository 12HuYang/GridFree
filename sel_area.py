# https://stackoverflow.com/questions/55636313/selecting-an-area-of-an-image-with-a-mouse-and-recording-the-dimensions-of-the-s

import tkinter as tk
from PIL import Image, ImageTk,ImageDraw
import numpy as np


class MousePositionTracker(tk.Frame):
    """ Tkinter Canvas mouse position widget. """

    def __init__(self, canvas):
        self.canvas = canvas
        self.canv_width = self.canvas.cget('width')
        self.canv_height = self.canvas.cget('height')
        self.reset()

        # Create canvas cross-hair lines.
        xhair_opts = dict(dash=(3, 2), fill='red', state=tk.HIDDEN)
        self.lines = (self.canvas.create_line(0, 0, 0, self.canv_height, **xhair_opts),
                      self.canvas.create_line(0, 0, self.canv_width,  0, **xhair_opts))

    def cur_selection(self):
        return (self.start, self.end)

    def begin(self, event):
        self.hide()
        self.start = (event.x, event.y)  # Remember position (no drawing).

    def update(self, event):
        self.end = (event.x, event.y)
        self._update(event)
        self._command(self.start, (event.x, event.y))  # User callback.

    def _update(self, event):
        # Update cross-hair lines.
        self.canvas.coords(self.lines[0], event.x, 0, event.x, self.canv_height)
        self.canvas.coords(self.lines[1], 0, event.y, self.canv_width, event.y)
        self.show()

    def reset(self):
        self.start = self.end = None

    def hide(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.HIDDEN)
        self.canvas.itemconfigure(self.lines[1], state=tk.HIDDEN)

    def show(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.NORMAL)
        self.canvas.itemconfigure(self.lines[1], state=tk.NORMAL)

    def autodraw(self, command=lambda *args: None):
        """Setup automatic drawing; supports command option"""
        self.reset()
        self._command = command
        self.canvas.bind("<Button-1>", self.begin)
        self.canvas.bind("<B1-Motion>", self.update)
        self.canvas.bind("<ButtonRelease-1>", self.quit)

    def quit(self, event):
        self.hide()  # Hide cross-hairs.
        self.reset()

    def disable(self):
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")



class SelectionObject:
    """ Widget to display a rectangular area on given canvas defined by two points
        representing its diagonal.
    """
    def __init__(self, canvas, select_opts):
        # Create attributes needed to display selection.
        self.canvas = canvas
        self.select_opts1 = select_opts
        self.width = self.canvas.cget('width')
        self.height = self.canvas.cget('height')
        # self.npimg=np.zeros((self.canvas.cget('height'),self.canvas.cget('width')))
        # self.img=img

        # Options for areas outside rectanglar selection.
        select_opts1 = self.select_opts1.copy()  # Avoid modifying passed argument.
        select_opts1.update(state=tk.HIDDEN)  # Hide initially.
        # Separate options for area inside rectanglar selection.
        select_opts2 = dict(dash=(2, 2), fill='', outline='black', state=tk.HIDDEN)

        # Initial extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = 0, 0,  1, 1
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        self.rects = (
            # Area *outside* selection (inner) rectangle.
            self.canvas.create_rectangle(omin_x, omin_y,  omax_x, imin_y, **select_opts1),
            # self.canvas.create_rectangle(omin_x, imin_y,  imin_x, imax_y, **select_opts1),
            # self.canvas.create_rectangle(imax_x, imin_y,  omax_x, imax_y, **select_opts1),
            # self.canvas.create_rectangle(omin_x, imax_y,  omax_x, omax_y, **select_opts1),
            # Inner rectangle.
            # self.canvas.create_rectangle(imin_x, imin_y,  imax_x, imax_y, **select_opts2)
            self.canvas.create_oval(imin_x, imin_y,  imax_x, imax_y, **select_opts2)
        )

        # self.startpx=(0,0)
        # self.endpx=(0,0)
        print('end init selection area')


    def update(self, start, end):
        # Current extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = self._get_coords(start, end)
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        # Update coords of all rectangles based on these extrema.
        self.canvas.coords(self.rects[0], omin_x, omin_y,  omax_x, imin_y),
        # self.canvas.coords(self.rects[1], omin_x, imin_y,  imin_x, imax_y),
        # self.canvas.coords(self.rects[2], imax_x, imin_y,  omax_x, imax_y),
        # self.canvas.coords(self.rects[3], omin_x, imax_y,  omax_x, omax_y),
        self.canvas.coords(self.rects[1], imin_x, imin_y,  imax_x, imax_y),


        for rect in self.rects:  # Make sure all are now visible.
            self.canvas.itemconfigure(rect, state=tk.NORMAL)


    def _get_coords(self, start, end):
        """ Determine coords of a polygon defined by the start and
            end points one of the diagonals of a rectangular area.
        """
        return (min((start[0], end[0])), min((start[1], end[1])),
                max((start[0], end[0])), max((start[1], end[1])))

    def hide(self):
        for rect in self.rects:
            self.canvas.itemconfigure(rect, state=tk.HIDDEN)

    def delete(self,rects):
        for rect in rects:
            self.canvas.delete(rect)


class Application(tk.Frame):

    # Default selection object options.
    SELECT_OPTS = dict(dash=(2, 2), stipple='gray25',
                          outline='')

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # path = "convimg.png"
        # img = ImageTk.PhotoImage(Image.open(path))
        # self.canvas = tk.Canvas(root, width=img.width(), height=img.height(),
        #                         borderwidth=0, highlightthickness=0)
        # self.canvas.pack(expand=True)
        #
        # self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
        # self.canvas.img = img  # Keep reference.
        # self.canvas=canvas
        self.canvas=parent
        npimg=np.zeros((int(self.canvas.cget('height')),int(self.canvas.cget('width'))))
        self.img=Image.fromarray(npimg)

        # Create selection object to show current selection boundaries.
        self.selection_obj = SelectionObject(self.canvas, self.SELECT_OPTS)
        self.posn_tracker = MousePositionTracker(self.canvas)


    def end(self,rects):
        self.posn_tracker.disable()
        self.selection_obj.delete(rects)
        self.canvas.update()

    def start(self):
        # Callback function to update it given two points of its diagonal.
        def on_drag(start, end, **kwarg):  # Must accept these arguments.
            self.selection_obj.update(start, end)


                # Create mouse position tracker that uses the function.
        self.posn_tracker.autodraw(command=on_drag)  # Enable callbacks.
        return self.selection_obj.rects


    def getinfo(self,rect):
        # sizes=self.canvas.coords(rect)
        # draw=ImageDraw.Draw(self.img)
        # draw.ellipse([(sizes[0],sizes[1]),(sizes[2],sizes[3])],fill='red')
        # self.img.save('selection_oval.tiff')
        return self.canvas.coords(rect)




# if __name__ == '__main__':
#
#     WIDTH, HEIGHT = 900, 900
#     BACKGROUND = 'grey'
#     TITLE = 'Image Cropper'
#
#     root = tk.Tk()
#     root.title(TITLE)
#     root.geometry('%sx%s' % (WIDTH, HEIGHT))
#     root.configure(background=BACKGROUND)
#
#     app = Application(root, background=BACKGROUND)
#     app.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.TRUE)
#     app.mainloop()


