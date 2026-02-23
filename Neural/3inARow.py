from neural import *
from tkinter import *

scale = 10
game = 0

wn = Tk()
wn.title = "Winow"

lbl = Label("Game" + game)
btn = Button(wn, text="Save", width=25)

lbl.grid(row = 0, column = 4*scale)

wn.mainloop()