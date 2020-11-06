import sys

import pygame as pg
from pygame.locals import *

from Boton import Boton
from Cursor import Cursor
from maze import Maze, Cell

pg.init()

s_width = 800
s_height = 600
win = pg.display.set_mode((s_width, s_height), 0, 32)
pg.display.set_caption('TB!')
play_width = 200  # 200 / 4 = 50 width per block
play_height = 200  # 200 / 4 = 50 height per block
block_size = 50

top_left_x = (s_width - play_width) // 2
top_left_y = 100

# Cells de 2 walls
es = pg.image.load("CELLS/2_Walls/ES.png")
ne = pg.image.load("CELLS/2_Walls/NE.png")
ns = pg.image.load("CELLS/2_Walls/NS.png")
nw = pg.image.load("CELLS/2_Walls/NW.png")
we = pg.image.load("CELLS/2_Walls/WE.png")
ws = pg.image.load("CELLS/2_Walls/WE.png")

# Cells de 3 walls
nes = pg.image.load("CELLS/3_Walls/NES.png")
nwe = pg.image.load("CELLS/3_Walls/NWE.png")
nws = pg.image.load("CELLS/3_Walls/NWS.png")
wes = pg.image.load("CELLS/3_Walls/WES.png")


def make_maze(nx, ny):
    # Maze entry position
    ix, iy = 0, 0

    maze = Maze(nx, ny, ix, iy)
    maze.make_maze()
    return maze


def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)


def draw_cell(surface, cell, x, y):
    #                (N, S, E, W)
    if cell.walls == (0, 1, 1, 0):
        pg.blit(es, (x, y))
    if cell.walls == (1, 0, 1, 0):
        pg.blit(ne, (x, y))
    if cell.walls == (1, 1, 0, 0):
        pg.blit(ns, (x, y))
    if cell.walls == (1, 0, 0, 1):
        pg.blit(nw, (x, y))
    if cell.walls == (0, 0, 1, 1):
        pg.blit(we, (x, y))
    if cell.walls == (0, 1, 0, 1):
        pg.blit(ws, (x, y))
    if cell.walls == (1, 1, 1, 0):
        pg.blit(nes, (x, y))
    if cell.walls == (1, 0, 1, 1):
        pg.blit(nwe, (x, y))
    if cell.walls == (1, 1, 0, 1):
        pg.blit(nws, (x, y))
    if cell.walls == (0, 1, 1, 1):
        pg.blit(wes, (x, y))

    pg.display.update()


def draw_maze(surface, maze):
    for i in range(len(maze.ny)):
        y = 30 + 30 * i
        for j in range(len(maze.nx)):
            x = 30 + 30 * j
            draw_cell(surface, maze[i][j], x, y)
    pg.display.update()


def main():
    run = True
    victory = True
    # Maze dimensions (ncols, nrows)
    maze = make_maze(10, 5)

    while run:
        win.fill((250, 173, 91))
        draw_maze(win, maze)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                pg.quit()
                sys.exit()
