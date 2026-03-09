from tkinter import *
import numpy as np
import random as rd

WIDTH = 1000
HEIGHT = 1000

root = Tk()
root.title("Neural Lander")

canvas = Canvas(master=root, bg="midnight blue", width=WIDTH, height=HEIGHT)

canvas.pack()
G = 10
class Lander:
    def __init__(self, canv, mass):
        self.canvas = canv
        self.position = np.array([float(WIDTH/2), float(100)])
        self.angle = 0

        self.x_size_body = 100
        self.y_size_body = 50

        self.v = np.array([0.0, 0.0])
        self.a = np.array([0.0, 0.0])
        self.f = np.array([0.0, 0.0])
        self.forces = []
        self.m = mass

        self.L = 0
        self.t = 0
        self.I = self.m*(self.x_size_body**2 + self.y_size_body**2)*(1/12)

    def get(self):
        return self.position, self.angle

    def render(self):
        diag = np.sqrt(self.x_size_body**2 + self.y_size_body**2)
        angle_betw = np.atan(self.y_size_body/self.x_size_body)
        p1 = [diag * np.cos(angle_betw + self.angle) + self.position[0], diag * np.sin(angle_betw + self.angle) + self.position[1]]
        p2 = [diag * np.cos(-angle_betw + self.angle) + self.position[0], diag * np.sin(-angle_betw + self.angle) + self.position[1]]
        p3 = [-diag * np.cos(angle_betw + self.angle) + self.position[0], -diag * np.sin(angle_betw + self.angle) + self.position[1]]
        p4 = [-diag * np.cos(-angle_betw + self.angle) + self.position[0], -diag * np.sin(-angle_betw + self.angle) + self.position[1]]

        body = self.canvas.create_polygon(p1[0], p1[1],
                                          p2[0], p2[1],
                                          p3[0], p3[1],
                                          p4[0], p4[1],
                                fill="grey60")
    
    def update(self, step=1.0):
        self.calc_force()
        if not([0, G] in self.forces):
            self.add_force([0, G])
        self.a = np.array(self.f)
        self.v += self.a * step
        self.position += self.v * step

        self.L += self.t*step*(1/self.I)
        self.angle += self.L * step
        self.t = 0

    def calc_force(self):
        self.f = np.array([0.0, 0.0])
        for f in self.forces:
            self.f += np.array(f)
    
    def add_force(self, n):
        if (type(n) == np.array or type(n) == list) and (len(n)==2):
            self.forces.append(n)
        else:
            raise TypeError("Force is not vector 2 (List or Array).")

    def add_torque(self, r, f):
        if not(type(f) == np.array or type(f) == list) and (len(f)==2):
            raise TypeError("Force is not vector 2 (List or Array).")
        if not(type(r) == np.array or type(r) == list) and (len(r)==2):
            raise TypeError("Moment location is not vector 2 (List or Array).")

        r_rel = np.array(r) - self.position
        unit_rel = r_rel/(np.sqrt(np.sum(np.square(r_rel))))
        unit_tan = (-unit_rel[1], unit_rel[0])
        angle_betw_tan_and_f = np.dot(np.array(f), unit_tan)/(np.sqrt(np.sum(np.square(f))))
        self.t = np.cross(r_rel, f)*np.sin(angle_betw_tan_and_f)

class Flaggie:
    def __init__(self, position, canv: Canvas):
        self.position = position
        self.pole_thickness = 20
        self.pole_length = 50
        self.canv = canv

    def render(self):
        pole = self.canv

class Floor:
    def __init__(self, canv: Canvas):
        self.color = "grey20"
        self.canv = canv
        platform_width = 200
        points = np.random.random(size = 10)
        self.height = 700
        r = 100
        dist = np.ones(10) * self.height + r * points
        place = rd.randint(0, 9)
        x_ = [x for x in range(0, WIDTH-platform_width, int((WIDTH-platform_width)/10))]

        for i in range(len(x_)):
            if i >= place:
                x_[i] += platform_width + 50
        self.platform_xcoordinates = [int(((x_[place-1] + x_[place])/2) - platform_width // 2),
                                 int(((x_[place-1] + x_[place])/2) + platform_width // 2)]
        
        x_.insert(place, self.platform_xcoordinates[1])
        x_.insert(place, self.platform_xcoordinates[0])
        dist = dist.tolist()
        dist.insert(place, self.height); dist.insert(place, self.height)
        self.points = [x_[i//2] if i%2==0 else dist[(i-1)//2] for i in range(24)]
        self.points.append(WIDTH); self.points.append(HEIGHT); self.points.append(0); self.points.append(HEIGHT)
        self.object = self.canv.create_polygon(self.points, fill="green")
        balls = self.canv.create_oval(self.platform_xcoordinates[0]-10, self.height-10, self.platform_xcoordinates[0]+10, self.height+10, fill="red")
        balls = self.canv.create_oval(self.platform_xcoordinates[1]-10, self.height-10, self.platform_xcoordinates[1]+10, self.height+10, fill="red")

    def render(self):
        self.object = self.canv.create_polygon(self.points, fill="green")
        balls = self.canv.create_oval(self.platform_xcoordinates[0]-10, self.height-10, self.platform_xcoordinates[0]+10, self.height+10, fill="red")
        balls = self.canv.create_oval(self.platform_xcoordinates[1]-10, self.height-10, self.platform_xcoordinates[1]+10, self.height+10, fill="red")

yanny = Lander(canvas, 1)
yanny.position -= np.array([150, 0])
yanny.render()

laurel = Lander(canvas, 1)
laurel.position += np.array([150, 0])
laurel.render()
laurel.add_torque([laurel.position[0], laurel.position[1]-25], [40, 0])

floor = Floor(canv=canvas)

def update_frame():
    canvas.delete("all")
    floor.render()
    yanny.update(step=0.1)
    yanny.render()
    laurel.update(step=0.1)
    laurel.render()
    root.after(16, update_frame)

root.after(10, update_frame)
root.mainloop()