from tkinter import *
from neural import *
import numpy as np
scale = 18

win = Tk()
win.title("MNIST Test")

canvas = Canvas(win, width=28*scale, height=28*scale, bg="white")

grid = np.zeros(784)

def gridrender(grid=grid, canvas=canvas):
    canvas.delete("all")
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
drawradius = 300
sensitivity = 0.8

def key(event):
    #print("pressed")
    repr(event.char)

def click_down(event):
    global mousedown
    #print("clicked at", event.x, event.y)
    mousedown = True

def click_up(event):
    global mousedown
    #print("released at", event.x, event.y)
    mousedown = False

def squares_in_radius(x, y, radius=drawradius):
    ind = []
    for e in range(len(grid)):
        x_square = (e % 28)*scale + scale//2
        y_square = (e // 28)*scale + scale//2

        if (x_square-x)**2 + (y_square-y)**2 <= drawradius:
            ind.append(e)
    return ind

def color_squares(squares, grid=grid, sensitivity=sensitivity):
    for s in squares:
        grid[s] = max(min(1, grid[s] + sensitivity), 0)
        #print(grid[s], "color")

def moved(event):
    global mousedown
    #print("moved to:", event.x, event.y, mousedown)
    if mousedown:
        sir = squares_in_radius(event.x, event.y)
        color_squares(sir)
        gridrender()


#canvas.bind("<Key>", key)
canvas.bind("<ButtonPress-1>", click_down)
canvas.bind("<ButtonRelease-1>", click_up)
canvas.bind('<Motion>', moved)


rand_button = Button(win, text="randomise", command=rando)
rand_button.pack()
win.mainloop()
