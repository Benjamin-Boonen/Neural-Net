#Fix diagonal for yellow (player 1) up right to bot left
#Make game number increase
from neural import *
from tkinter import *
from time import sleep
scl = 5 
players = ["GREEN", "YELLOW"]
taken = []
taken0 = []
taken1 = []
turn = False
wins = 0

wn = Tk()
cv = Canvas(wn, width=100*scl, height=100*scl, bg="BLACK")

lbl = Label(wn, text=wins)

#Create lines
cv.create_rectangle(30*scl, 0, 35*scl, 100*scl, fill="RED")
cv.create_rectangle(65*scl, 0, 70*scl, 100*scl, fill="RED")
cv.create_rectangle(0, 30*scl, 100*scl, 35*scl, fill="RED")
cv.create_rectangle(0, 65*scl, 100*scl, 70*scl, fill="RED")

#On left click
def callback(event):
    global turn
    global wins
    cord = [1, 1]
    legal = True

    if event.x < 30*scl:
        cord[0] = 0
        x1 = 0*scl
        x2 = 30*scl
    elif event.x > 70*scl:
        cord[0] = 2
        x1 = 70*scl
        x2 = 100*scl
    else:
        cord[0] = 1
        x1 = 35*scl
        x2 = 65*scl

    if event.y < 30*scl:
        cord[1] = 0
        y1 = 0*scl
        y2 = 30*scl
    elif event.y > 70*scl:
        cord[1] = 2
        y1 = 70*scl
        y2 = 100*scl
    else:
        cord[1] = 1
        y1 = 35*scl
        y2 = 65*scl
    
    for i in range(len(taken)):
        if cord == taken[i]:
            legal = False
    
    if legal:
        cv.create_rectangle(x1, y1, x2, y2, fill = players[turn])
        cv.update()

        taken.append(cord)
        if turn:
            taken1.append(cord)
        else:
            taken0.append(cord)

        #Quick overview, once 3 squares have been taken, it starts checking for wins. It will then take a square, 
        #and for every other taken square it starts checking if they are on the same x or y. If so, it adds 1 to
        #the checking list for that axis. The list starts at because it doesn't check itself. If any value in the checking
        #reaches 3, that's a win ig?
        if turn:
            for i in taken1:
                checkingX = [1, 1, 1]
                checkingY = [1, 1, 1]
                checkingD = [0, 0, 
                             0, 0]
                if i == [1, 1]:
                    for j in taken1:
                        if j == [0, 0]:
                            checkingD[0] = 1
                        elif j == [2, 0]:
                            checkingD[1] = 1
                        elif j == [0, 2]:
                            checkingD[2] = 1
                        elif j == [2, 2]:
                            checkingD[3] = 1
                        if checkingD[0] and checkingD[3] == 1:
                            reset()
                            break
                        elif checkingD[1] and checkingD[2] == 1:
                            reset()
                            break
                else:
                    for j in taken1:
                        if i != j:
                            if i[0] == j[0]:
                                checkingX[i[0]] += 1

                            elif i[1] == j[1]:
                                checkingY[i[1]] += 1
                    for j in range(2):
                        if checkingX[j] == 3 or checkingY[j] == 3:
                            reset()
                            break
                    
        else:
            for i in taken0:
                checkingX = [1, 1, 1]
                checkingY = [1, 1, 1]
                checkingD = [0, 0, 
                             0, 0]
                if i == [1, 1]:
                    for j in taken0:
                        if j == [0, 0]:
                            checkingD[0] = 1
                        elif j == [2, 0]:
                            checkingD[1] = 1
                        elif j == [0, 2]:
                            checkingD[2] = 1
                        elif j == [2, 2]:
                            checkingD[3] = 1
                        if checkingD[0] and checkingD[3] == 1:
                            wins += 1
                            reset()
                            break
                        elif checkingD[1] and checkingD[2] == 1:
                            wins += 1
                            reset()
                            break
                else:
                    for j in taken0:
                        if i != j:
                            if i[0] == j[0]:
                                checkingX[i[0]] += 1

                            elif i[1] == j[1]:
                                checkingY[i[1]] += 1
                    for j in range(2):
                        if checkingX[j] == 3 or checkingY[j] == 3:
                            wins += 1
                            reset()
                            break

        turn  = not turn
        if len(taken) == 9:
            reset()

def reset():
    global lbl

    print("Reseting")
    cv.delete("all")
    cv.create_rectangle(30*scl, 0, 35*scl, 100*scl, fill="RED")
    cv.create_rectangle(65*scl, 0, 70*scl, 100*scl, fill="RED")
    cv.create_rectangle(0, 30*scl, 100*scl, 35*scl, fill="RED")
    cv.create_rectangle(0, 65*scl, 100*scl, 70*scl, fill="RED")
    lbl.config(text=wins)
    taken.clear()
    taken0.clear()
    taken1.clear()

lbl.pack()
cv.pack()
cv.bind("<Button-1>", callback)
wn.mainloop()