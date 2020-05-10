import numpy as np
import time
import sys
from Game_of_Life import Game_of_Life

filename = sys.argv[1]

grid = np.genfromtxt("../data/" + filename)

x,y = grid.shape

game = Game_of_Life(x,y)
game.initiate_grid(grid=grid)

time.sleep(5)
for i in range(2000):
    print(i)
    game.iterate()
    time.sleep(0.05)