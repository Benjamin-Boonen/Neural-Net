from neural import *
from tkinter import *
from time import sleep

scl = 5
players = ["GREEN", "YELLOW"]
taken = []
taken0 = []
taken1 = []
takenNr = []
games = 0
n0 = Network(shape = [9, 4, 5, 4, 9], is_random = True, activation=SIGMOID)
n1 = Network(shape = [9, 4, 5, 4, 9], is_random = True, activation=SIGMOID)
winner = None

wn = Tk()
cv = Canvas(wn, width=100*scl, height=100*scl, bg="BLACK")

def drawBoard():
    cv.create_rectangle(30*scl, 0, 35*scl, 100*scl, fill="RED")
    cv.create_rectangle(65*scl, 0, 70*scl, 100*scl, fill="RED")
    cv.create_rectangle(0, 30*scl, 100*scl, 35*scl, fill="RED", outline="RED")
    cv.create_rectangle(0, 65*scl, 100*scl, 70*scl, fill="RED", outline="RED")

drawBoard()

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
    lines = [
        [(0,0),(1,0),(2,0)],  # rows
        [(0,1),(1,1),(2,1)],
        [(0,2),(1,2),(2,2)],
        [(0,0),(0,1),(0,2)],  # columns
        [(1,0),(1,1),(1,2)],
        [(2,0),(2,1),(2,2)],
        [(0,0),(1,1),(2,2)],  # diagonals
        [(2,0),(1,1),(0,2)],
    ]
    return any(all(sq in moves for sq in line) for line in lines)

def reset():
    global games, winner
    generation()
    sleep(0.5)
    cv.delete("all")
    drawBoard()
    lblG.config(text=games)
    taken.clear()
    taken0.clear()
    taken1.clear()
    takenNr.clear()
    winner = None
    games += 1
    print("----------Reset----------")
    print("Game", games)

def aiMove(network, player_taken, player_nr):
    feed = buildFeed()
    output = f_propagation(network, feed)
    remaining_indices = [i for i in range(len(output)) if i not in takenNr]
    if not remaining_indices:
        return None
    ind = max(remaining_indices, key=lambda i: output[i])
    cord = (ind % 3, ind // 3)
    getRekt(cord, players[player_nr])
    taken.append(cord)
    player_taken.append(cord)
    takenNr.append(ind)
    cv.update()
    return cord

def gameLoop():
    global winner, loser
    # AI 0 move
    move = aiMove(n0, taken0, 0)
    if move is None or check_winner(taken0) or len(taken) == 9:
        if check_winner(taken0):
            winner = n0
        reset()
        wn.after(100, gameLoop)
        return
    sleep(0.3)

    # AI 1 move
    move = aiMove(n1, taken1, 1)
    if move is None or check_winner(taken1) or len(taken) == 9:
        if check_winner(taken1):
            winner = n1
        reset()
        wn.after(100, gameLoop)
        return
    sleep(0.3)

    wn.after(100, gameLoop)

def loadNet():
    global n0, n1, games
    n0 = load_network('networks/3iaR_n0.nn')
    n1 = load_network('networks/3iaR_n1.nn')
    games = 0
    print("loaded networks")
    reset()

def saveNet():
    save_network(n0, f"networks/3iaR_n0_{games}.nn")
    save_network(n1, f"networks/3iaR_n1_{games}.nn")
    print("Saved at game", games)

def copy_network(src, dst):
    for i in range(len(src.layers)):
        if not src.layers[i].is_outp():
            dst.layers[i].set_weights(src.layers[i].get_weights().copy())
            dst.layers[i].set_biases(src.layers[i].get_biases().copy())
            
def generation():
    global winner
    if winner is None:
        n0.radiate(factor=0.1)
        n1.radiate(factor=0.1)
    if winner is n0:
        copy_network(n0, n1)
        n1.radiate(factor=0.2)
    else:
        copy_network(n1, n0)
        n0.radiate(factor=0.2)
    

lblG = Label(wn, text=games)
saveBtn = Button(wn, text="Save networks", activebackground="blue", activeforeground="white", disabledforeground="gray", command=saveNet)
loadBtn = Button(wn, text="Load networks", activebackground="blue", activeforeground="white", disabledforeground="gray", command=loadNet)

lblG.pack()
saveBtn.pack()
loadBtn.pack()
cv.pack()
wn.after(500, gameLoop)
wn.mainloop()