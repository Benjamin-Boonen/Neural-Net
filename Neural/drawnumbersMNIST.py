from tkinter import *
from neural import *
import numpy as np
scale = 18

win = Tk()
win.title("MNIST Test")

canvas = Canvas(win, width=28*scale, height=28*scale, bg="white")

grid = np.zeros(784)

def gridrender(grid=grid):
    for e in range(len(grid)):
        collumn = e % 28
        row = e // 28
        color = (grid[e]*100)
        color = "grey"+str(int(100-color))
        #print(f"square {e} has value {color}")
        square = canvas.create_rectangle(collumn*scale, row*scale, (collumn+1)*scale, (row+1)*scale, fill=color)

canvas.pack()
gridrender()

def rando(grid=grid):
    for e in range(len(grid)):
        grid[e] = np.random.rand()
    gridrender()

mousedown = False
def key(event):
    print("pressed")
    repr(event.char)

def click_down(event):
    global mousedown
    print("clicked at", event.x, event.y)
    mousedown = True

def click_up(event):
    global mousedown
    print("released at", event.x, event.y)
    mousedown = False

def moved(event):
    global mousedown
    print("moved to:", event.x, event.y, mousedown)


#canvas.bind("<Key>", key)
canvas.bind("<ButtonPress-1>", click_down)
canvas.bind("<ButtonRelease-1>", click_up)
canvas.bind('<Motion>', moved)


rand_button = Button(win, text="randomise", command=rando)
rand_button.pack()
win.mainloop()
