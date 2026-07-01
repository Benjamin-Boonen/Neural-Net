from neural import *
from tkinter import * 
from time import sleep

scl = 5 
players = ["GREEN", "YELLOW"]
taken = []
taken0 = []
taken1 = []
takenNr = []
turn = False
wins = 0
games = 0
n = Network(shape = [9, 4, 5, 4, 9], is_random = True, activation=SIGMOID)

wn = Tk()
cv = Canvas(wn, width=100*scl, height=100*scl, bg="BLACK")

#Create lines
cv.create_rectangle(30*scl, 0, 35*scl, 100*scl, fill="RED")
cv.create_rectangle(65*scl, 0, 70*scl, 100*scl, fill="RED")
cv.create_rectangle(0, 30*scl, 100*scl, 35*scl, fill="RED", outline="RED")
cv.create_rectangle(0, 65*scl, 100*scl, 70*scl, fill="RED", outline="RED")

#On left click
def callback(event):
    global turn, wins

    if event.x < 35*scl:
        cx = 0
    elif event.x > 65*scl:
        cx = 2
    else:
        cx = 1

    if event.y < 35*scl:
        cy = 0
    elif event.y > 65*scl:
        cy = 2
    else:
        cy = 1

    cord = (cx, cy)
    legal = cord not in taken
    
    if legal and not turn:
        getRekt(cord, players[turn])
        cv.update()

        taken.append(cord)  
        taken0.append(cord)
        takenNr.append(cord[1]*3 + cord[0])

        before = games
        checkWin()
        if games != before: return
        if len(taken) == 9:
            reset()
            return
        turn = not turn

    elif legal and turn:      #if/elif   Switch for PvA/PvP
        feed = buildFeed()
        output = f_propagation(n, feed)
        
        remaining_indices = [i for i in range(len(output)) if i not in takenNr]

        if remaining_indices:
            ind = max(remaining_indices, key=lambda i: output[i])

        target = (ind % 3, ind // 3)
        exVl = cord[1]*3 + cord[0]

        b_propagation(n, feed, exVl, learning_rate=0.1, function=SIGMOID)
        print(ind, "|", exVl)
        #getRekt(target, players[turn])       #Switch for PvA/PvP
        getRekt(cord, players[turn])        #Switch for PvA/PvP
        
        taken.append(cord)        #target/cord      Switch for PvA/PvP
        taken1.append(cord)
        takenNr.append(cord[1]*3 + cord[0])         #ind/cord[1]*3 + cord[0     Switch for PvA/PvP

        before = games
        checkWin()
        if games != before: return
        if len(taken) == 9:
            reset()
            return
        turn = not turn

def reset():
    global lbl, turn, games
    sleep(0.5)
    cv.delete("all")
    cv.create_rectangle(30*scl, 0, 35*scl, 100*scl, fill="RED")
    cv.create_rectangle(65*scl, 0, 70*scl, 100*scl, fill="RED")
    cv.create_rectangle(0, 30*scl, 100*scl, 35*scl, fill="RED", outline="RED")
    cv.create_rectangle(0, 65*scl, 100*scl, 70*scl, fill="RED", outline="RED")
    lblW.config(text=wins)
    lblG.config(text=games)
    taken.clear()
    taken0.clear()
    taken1.clear()
    takenNr.clear()
    turn = False
    games += 1

    print("----------Reset----------")
    print("Game", games)
    print("Output   |   Expected Value")

def getRekt(cord, clr):
    x1 = cord[0]*35*scl
    x2 = x1 + 30*scl
    y1 = cord[1]*35*scl
    y2 = y1 + 30*scl
    cv.create_rectangle(x1, y1, x2, y2, fill=clr)

def buildFeed():
    board = [0, 0, 0,
             0, 0, 0,
             0, 0, 0]

    for move in taken0:
        index = move[1] * 3 + move[0]
        board[index] = -1

    for move in taken1:
        index = move[1] * 3 + move[0]
        board[index] = 1

    return board

def check_winner(moves):
    wins = [
        [(0,0),(1,0),(2,0)],  # rows
        [(0,1),(1,1),(2,1)],
        [(0,2),(1,2),(2,2)],
        [(0,0),(0,1),(0,2)],  # columns
        [(1,0),(1,1),(1,2)],
        [(2,0),(2,1),(2,2)],
        [(0,0),(1,1),(2,2)],  # diagonals
        [(2,0),(1,1),(0,2)],
    ]
    return any(all(sq in moves for sq in line) for line in wins)

def checkWin():
    global wins
    if check_winner(taken0):
        wins += 1
        reset()
    elif check_winner(taken1):
        reset()

def loadNet():
    global n, games
    model = "3iaR_1.nn"
    games = 0       #set to model number -1
    n = load_network(f'networks/{model}')
    print("loaded", model)
    reset()

def saveNet():
    save_network(n, f"networks/3iaR_{games}.nn")
    print("Saved 3iaR_", games, ".nn")

lblW = Label(wn, text=wins)
lblG = Label(wn, text=games)
saveBtn = Button(wn, text="Save network", activebackground="blue", activeforeground="white", disabledforeground="gray", command=saveNet)
loadBtn = Button(wn, text="Load network", activebackground="blue", activeforeground="white", disabledforeground="gray", command=loadNet)

lblW.pack()
lblG.pack()
saveBtn.pack()
loadBtn.pack()
cv.pack()
cv.bind("<Button-1>", callback)
reset()
wn.mainloop()