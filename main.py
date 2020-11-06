import sys

import pygame as pg

pg.init()

s_width = 800
s_height = 600
win = pg.display.set_mode((s_width, s_height), 0, 32)
pg.display.set_caption('Inteligencia Artificial - Trabajo 2')

# Tamaño de cada Casilla
size_cell = 50

# Posicion del Laberinto
posLabX = 200
posLabY = 150

# Posicion del Raton
posRatX = 210
posRatY = 160

# Posicion del Queso
posCheX = 200
posCheY = 150
# Posicion del queso en Casilla
cheseX = 6
cheseY = 6

# Cargar Imagenes
mouse = pg.image.load("IMG/mouse.png")
chese = pg.image.load("IMG/cheese.png")

# Matriz de 7x7 para Laberinto
maze_array = [[1, 0, 1, 1, 1, 1, 0],
              [1, 1, 1, 0, 0, 1, 0],
              [0, 0, 0, 0, 1, 1, 0],
              [1, 1, 1, 1, 1, 0, 0],
              [1, 0, 0, 0, 0, 0, 1],
              [1, 0, 1, 1, 1, 1, 1],
              [1, 1, 1, 0, 1, 1, 1]
              ]

def draw_text(text, size, pos_x, pos_y, surface):
    font = pg.font.SysFont('comicsans', size)
    label = font.render(text, 1, (255, 255, 255))
    surface.blit(label, (pos_x, pos_y))

def draw_mouse(surface, x, y):
    surface.blit(mouse, (posRatX + size_cell * x, posRatY + size_cell * y))

    pg.display.update()

def draw_chesse(surface, x, y):
    surface.blit(chese, (posCheX + size_cell * x, posCheY + size_cell * y))
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
    pg.display.update()

def update_surface(surface, mouseX, mouseY):
    draw_text('Trabajo 2', 100, 200, 50, surface)
    draw_maze(surface)
    draw_chesse(surface, cheseX, cheseY)
    draw_mouse(surface, mouseX, mouseY)
    pg.display.update()

run = True
victory = True
# Maze dimensions (ncols, nrows)
win.fill((150, 10, 100))
# Posicion inicial de la Rata
mouseX = 0
mouseY = 0
update_surface(win, mouseX, mouseY)
while run:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
            pg.quit()
            sys.exit()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_LEFT:
                mouseX = mouseX - 1
                update_surface(win, mouseX, mouseY)
            if event.key == pg.K_UP:
                mouseY = mouseY - 1
                update_surface(win, mouseX, mouseY)
            if event.key == pg.K_RIGHT:
                mouseX = mouseX + 1
                update_surface(win, mouseX, mouseY)
            if event.key == pg.K_DOWN:
                mouseY = mouseY + 1
                update_surface(win, mouseX, mouseY)
pg.quit()
