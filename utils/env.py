
type = 2

if type == 1:
    WIDTH = 15
    HEIGHT = 15
    x_start = (2.0, 2.0, 0)  # Starting node (x, y, yaw)
    x_goal = (10.0, 2.0)  # Goal node
if type == 2:
    WIDTH = 20
    HEIGHT = 20
    x_start = (2.0, 2.0, 0)  # Starting node (x, y, yaw)
    x_goal = (15.0, 3.0)  # Goal node

class Env:
    def __init__(self):
        self.x_range = (0, WIDTH)
        self.y_range = (0, HEIGHT)
        self.obs_boundary = self.obs_boundary(WIDTH, HEIGHT)
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary(width, height):  # circle
        w = width
        h = height
        linewidth = 0.5
        obs_boundary = [
            [0, 0, linewidth, h],
            [0, h, w, linewidth],
            [linewidth, 0, w, linewidth],
            [w, linewidth, linewidth, h]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        # obs_rectangle = [
        #     [14, 12, 8, 2],
        #     [18, 22, 8, 3],
        #     [26, 7, 2, 12],
        #     [32, 14, 10, 2]
        # ]
        obs_rectangle = []
        return obs_rectangle
    @staticmethod
    def obs_circle():
        if type == 1:
            obs_cir = [
                [10, 10, 2],
                [10, 15, 2],
                [10, 20, 2],
                [10, 25, 2],
                [20, 3, 2],
                [20, 8, 2],
                [20, 13, 2],
                [20, 18, 2],
                [30, 5, 2],
                [30, 10, 2]
            ]
        if type == 2:
            obs_cir = []
        return obs_cir