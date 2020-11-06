import sys

import pygame as pg

from mouse import Mouse

pg.init()

# Cargar Imagenes
upcpng = pg.image.load("IMG/UPC.png")
mousepng = pg.image.load("IMG/mouse.png")
cheesepng = pg.image.load("IMG/cheese.png")

s_width = 766
s_height = 610
win = pg.display.set_mode((s_width, s_height), 0, 32)
pg.display.set_icon(upcpng)
pg.display.set_caption('Inteligencia Artificial - Trabajo 2')

# Tamaño de cada Casilla
size_cell = 50

# Posicion del Laberinto
posLabX = 10
posLabY = 100

# Posicion del Raton
posRatX = posLabX + 10
posRatY = posLabY + 10
# Posicion del Raton en Casilla
mouseX = 0
mouseY = 0

# Posicion del Queso
posCheX = posRatX
posCheY = posRatY
# Posicion del queso en Casilla
cheeseX = 9
cheeseY = 9

# Matriz de 7x7 para Laberinto (40 movimientos para llegar al queso)
energy = 40
maze_array = [[1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
              [0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
              [1, 1, 0, 1, 0, 1, 0, 0, 0, 1],
              [1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 0, 1, 1]]


def draw_text(text, size, pos_x, pos_y, surface):
    font = pg.font.SysFont('comicsans', size)
    label = font.render(text, 1, (255, 255, 255))
    surface.blit(label, (pos_x, pos_y))


def draw_mouse(surface, mouse):
    surface.blit(mousepng, (posRatX + size_cell * mouse.posX, posRatY + size_cell * mouse.posY))

    pg.display.update()


def draw_cheese(surface, x, y):
    surface.blit(cheesepng, (posCheX + size_cell * x, posCheY + size_cell * y))
    pg.display.update()


def draw_maze(surface):
    # Dibujar Fondo del Laberinto
    pg.draw.rect(surface, (0, 0, 0),
                 (posLabX, posLabY, len(maze_array) * size_cell,
                  len(maze_array) * size_cell))
    # Dibujar Laberinto
    for i in range(len(maze_array)):
        y = size_cell * i
        for j in range(len(maze_array)):
            x = size_cell * j
            if maze_array[i][j]:
                pg.draw.rect(surface, (255, 255, 255),
                             (x + posLabX, y + posLabY, size_cell, size_cell))
                pg.draw.rect(surface, (0, 0, 0),
                             (x + posLabX, y + posLabY, size_cell, size_cell), 1)
    pg.display.update()


def update_surface(surface, mouse):
    surface.fill((228, 35, 34))
    draw_text('Trabajo 2', 80, 10, 10, surface)
    draw_text('Grupo 4', 50, 20, 60, surface)
    draw_text('Hunger', 50, 530, 100, surface)
    draw_text(str(mouse.energy), 50, 530, 130, surface)
    draw_maze(surface)
    draw_cheese(surface, cheeseX, cheeseY)
    draw_mouse(surface, mouse)
    pg.display.update()


run = True
victory = True
raton = Mouse('Juan', mouseX, mouseY, energy)
update_surface(win, raton)
while run:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
            pg.quit()
            sys.exit()
        if raton.posX == cheeseX and raton.posY == cheeseY:
            raton.energy = 999
            raton.posX = 8
            raton.posY = 9
            update_surface(win, raton)
            draw_text('Nice Cheese!', 58, 510, 560, win)
            pg.display.flip()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_LEFT:
                raton.move('LEFT')
                update_surface(win, raton)
            if event.key == pg.K_UP:
                raton.move('UP')
                update_surface(win, raton)
            if event.key == pg.K_RIGHT:
                raton.move('RIGHT')
                update_surface(win, raton)
            if event.key == pg.K_DOWN:
                raton.move('DOWN')
                update_surface(win, raton)
pg.quit()
