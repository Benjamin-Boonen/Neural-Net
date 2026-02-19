from tkinter import *
from neural import *
import numpy as np
scale = 20

win = Tk()
win.title("MNIST Test")

canvas = Canvas(win, width=28*scale, height=28*scale, bg="white")

grid = np.zeros(784)

model = "mnist_1m.nn"
n = load_network(f'networks/{model}')
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
guess = 0
def rando(grid=grid):
    for e in range(len(grid)):
        grid[e] = np.random.rand()
    gridrender()

def clear():
    for e in range(len(grid)):
        grid[e] = 0
    gridrender()

mousedown = False
drawradius = 500
sensitivity = 1
text_var = StringVar()
text_var.set(f"Guess: {guess}")
est = Label(win, textvariable=text_var, height=3, width=30, bg="white", font=("Helvetica", 16, "bold"))
est.pack()

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

def color_squares_in_radius(x, y, radius=drawradius):
    for e in range(len(grid)):
        x_square = (e % 28)*scale + scale//2
        y_square = (e // 28)*scale + scale//2

        if (x_square-x)**2 + (y_square-y)**2 <= drawradius:
            d = np.sqrt((x_square-x)**2 + (y_square-y)**2)
            amt = -((d/drawradius)**2)+1
            color_square(e, amt=amt)

def color_square(square, grid=grid, amt=1):
    grid[square] = max(min(1, grid[square] + amt), 0)
    #print(grid[s], "color")

def moved(event):
    global mousedown
    global guess
    #print("moved to:", event.x, event.y, mousedown)
    if mousedown:
        color_squares_in_radius(event.x, event.y)
        gridrender()
        guess = f_propagation(n, grid)
        guess = guess.tolist().index(np.max(guess))
        text_var.set(f"Guess: {guess}")

#canvas.bind("<Key>", key)
canvas.bind("<ButtonPress-1>", click_down)
canvas.bind("<ButtonRelease-1>", click_up)
canvas.bind('<Motion>', moved)


rand_button = Button(win, text="randomise", command=rando)
rand_button.pack()
clear_button = Button(win, text="clear", command=clear)
clear_button.pack()

win.mainloop()