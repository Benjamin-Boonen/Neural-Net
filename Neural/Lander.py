from tkinter import *
import numpy as np

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
        diag = np.sqrt(self.x_size_body**2 + self.y_size_body)
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

class Floor:
    def __init__(self, canv: Canvas):
        self.polygon = None
        self.color = "grey20"
        points_left = np.random.random(size = 5)
        points_right = np.random.random(size = 5)
        height = 100
        dist_left = np.ones(5) * 700 + height * points_left
        dits_right = np.ones(5) * 700 + height * points_right
        self.object = canv.create_polygon()####### !!!!!!!!

yanny = Lander(canvas, 1)
yanny.position -= np.array([150, 0])
yanny.render()

laurel = Lander(canvas, 1)
laurel.position += np.array([150, 0])
laurel.render()
laurel.add_torque([laurel.position[0], laurel.position[1]-25], [40, 0])

def update_frame():
    canvas.delete("all")
    yanny.update(step=0.1)
    yanny.render()
    laurel.update(step=0.1)
    laurel.render()
    root.after(16, update_frame)

root.after(10, update_frame)
root.mainloop()