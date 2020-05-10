from time import sleep
from random import randint
import pygame
import numpy as np
import time

class Game_of_Life():
    def __init__(self, x_max, y_max):
        pygame.init()
        
        self.x_max = x_max
        self.y_max = y_max
        
        self.x_windowsize = 9*x_max
        self.y_windowsize = 9*y_max

        self.screen = pygame.display.set_mode((self.x_windowsize, self.y_windowsize), 0, 24) #New 24-bit screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 40)
    
    def initiate_grid(self, grid=None, seed=None):
        if grid is None:
            if not seed is None:
                np.random.seed(seed)
            self.start_grid = np.random.randint(0, 2, (self.x_max, self.y_max))
        else:
            self.start_grid = grid
        self.grid = self.start_grid.copy()
        
        self.color_wheel = 0
        self.alive_color = pygame.Color(0,0,0)
        self.alive_color.hsva = [self.color_wheel, 100, 100]

        for x in range(self.x_max):
            for y in range(self.y_max):
                alive = self.grid[x][y]
                cell_color = self.alive_color if alive else (0,0,0)
                self.draw_block(x, y, cell_color)
        pygame.display.flip()
        
    def iterate(self, iterations=1):
        for i in range(iterations):
            self.grid = self.evolve(self.grid)
            cell_number = 0
            for x in range(self.x_max):
                for y in range(self.y_max):
                    alive = self.grid[x][y]
                    cell_number += 1
                    cell_color = self.alive_color if alive else (0,0,0)
                    self.draw_block(x, y, cell_color)
            # self.screen.blit(self.update_fps(), (10, 0))
            # self.clock.tick(240)
            pygame.display.flip()
            self.color_wheel = (self.color_wheel + 2) % 360
            self.alive_color.hsva = (self.color_wheel, 100, 100)
            cell_number = 0
        
        
    def draw_block(self, x, y, alive_color):
        block_size = 9
        x *= block_size
        y *= block_size
        center_point = ((x + (block_size // 2)), (y + (block_size // 2)))
        pygame.draw.circle(self.screen, alive_color, center_point, block_size // 2,0)

    def evolve_cell(self, alive, neighbours):
        return neighbours == 3 or (alive and neighbours == 2)


    def evolve(self, grid):
        new_grid = np.zeros((self.x_max, self.y_max))
        for r in range(self.x_max):
            for c in range(self.y_max):
                cell = grid[r][c]
                neighbours = self.count_neighbours(grid, (r, c))
                new_grid[r][c] = 1 if self.evolve_cell(cell, neighbours) else 0
        return new_grid

    def count_neighbours(self, grid, position):
        x,y = position
        neighbour_cells = np.array([(x - 1, y - 1), (x - 1, y + 0), (x - 1, y + 1),
                                    (x + 0, y - 1),                 (x + 0, y + 1),
                                    (x + 1, y - 1), (x + 1, y + 0), (x + 1, y + 1)])
        count = 0
        for x,y in neighbour_cells:
            # if (0 <= x < xlen) and (0 <= y < ylen): # Why is this slower???
            if x >= 0 and y >= 0:
                try:
                    count += grid[x][y]
                except:
                    pass
        # Why is this slower????:
        # in_bounds_indices = (neighbour_cells[:,0] >= 0) * (neighbour_cells[:,1] >= 0) * (neighbour_cells[:,0] < xlen) * (neighbour_cells[:,1] < ylen)
        # neighbour_cells = neighbour_cells[in_bounds_indices, :]
        # count = np.sum(grid[neighbour_cells[:,0], neighbour_cells[:,1]])
        # assert(count == count2)
        return count
    
    def pause(self, dur=5):
        time.sleep(dur)

    def update_fps(self):
        fps = str(int(self.clock.get_fps()))
        fps_text = self.font.render(fps, False, pygame.Color("coral"))
        return fps_text
