from __future__ import print_function

import datetime
import random
import time

import numpy as np
import pygame as pg
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense
from keras.models import Sequential

####################################################################################################################
# CONSTANTES
####################################################################################################################

pg.init()

# Cargar Imagenes
upcpng = pg.image.load("IMG/UPC.png")
ratonpng = pg.image.load("IMG/mouse.png")
quesopng = pg.image.load("IMG/cheese.png")

# Configuracion de la Ventana
s_width = 1080
s_height = 766
win = pg.display.set_mode((s_width, s_height), 0, 32)
pg.display.set_icon(upcpng)
pg.display.set_caption('Inteligencia Artificial - Trabajo 2')

# Tamaño de cada Casilla
size_cell = 50

# Posicion del Laberinto
posLabX = 10
posLabY = 100

# Posicion del Inicial de Raton y Queso
posRatX = posLabX + 10
posRatY = posLabY + 10

matriz = np.array([
    [1., 0., 1., 1., 1., 1., 1.],
    [1., 1., 1., 0., 0., 1., 0.],
    [0., 0., 0., 1., 1., 1., 0.],
    [1., 1., 1., 1., 0., 0., 1.],
    [1., 0., 0., 0., 1., 1., 1.],
    [1., 0., 1., 1., 1., 1., 1.],
    [1., 1., 1., 0., 1., 1., 1.]
])

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Movimientos posibles
act_dic = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

# Como s puede mover la rata
num_act = len(act_dic)
# Factor para explorar
exp_fat = 0.1


####################################################################################################################
# CLASES
####################################################################################################################
# Clase para Definir lo que es el Laberinto, el Raton y el Queso; y sus respectivas posiciones
class Laberinto(object):
    def __init__(self, matriz, rat=(0, 0)):
        self._matriz = np.array(matriz)
        nfilas, ncolumnas = self._matriz.shape
        self.target = (nfilas - 1, ncolumnas - 1)  # En que casilla esta el queso
        self.casillas_vacias = [(r, c) for r in range(nfilas) for c in range(ncolumnas) if self._matriz[r, c] == 1.0]
        self.casillas_vacias.remove(self.target)
        if self._matriz[self.target] == 0.0:
            raise Exception("Laberinto invalido: El quezo esta bloqueado!")
        if not rat in self.casillas_vacias:
            raise Exception("Posicion de Rata Invalido: Debe aparecer libre!")
        self.reset(rat)

    def reset(self, rat):
        self.rat = rat
        self.matriz = np.copy(self._matriz)
        nfilas, ncolumnas = self.matriz.shape
        row, col = rat
        self.state = (row, col, 'start')
        self.min_recompensa = -0.5 * self.matriz.size
        self.total_recompensa = 0
        self.visited = set()

    def update_state(self, accion):
        nfilas, ncolumnas = self.matriz.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state
        if self.matriz[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # Marca casilla visitada
        acciones_permitidas = self.acciones_permitidas()
        if not acciones_permitidas:
            nmode = 'blocked'
        elif accion in acciones_permitidas:
            nmode = 'valid'
            if accion == LEFT:
                ncol -= 1
            elif accion == UP:
                nrow -= 1
            if accion == RIGHT:
                ncol += 1
            elif accion == DOWN:
                nrow += 1
        else:  # Accion Invalida
            mode = 'invalid'
        # Nuevo estado
        self.state = (nrow, ncol, nmode)

    def recompensa(self):
        rat_row, rat_col, mode = self.state
        nfilas, ncolumnas = self.matriz.shape
        if rat_row == nfilas - 1 and rat_col == ncolumnas - 1:
            return 1.0
        if mode == 'blocked':
            return self.recompensa - 1
        if (rat_row, rat_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, accion):
        self.update_state(accion)
        recompensa = self.recompensa()
        self.total_recompensa += recompensa
        status = self.estado_juego()
        estado_entorno = self.observe()
        return estado_entorno, recompensa, status

    def observe(self):
        canvas = np.copy(self.matriz)
        estado_entorno = canvas.reshape((1, -1))
        return estado_entorno

    def estado_juego(self):
        if self.total_recompensa < self.min_recompensa:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nfilas, ncolumnas = self.matriz.shape
        if rat_row == nfilas - 1 and rat_col == ncolumnas - 1:
            return 'win'
        return 'not_over'

    def acciones_permitidas(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        acciones = [0, 1, 2, 3]
        nfilas, ncolumnas = self.matriz.shape
        if row == 0:
            acciones.remove(1)
        elif row == nfilas - 1:
            acciones.remove(3)
        if col == 0:
            acciones.remove(0)
        elif col == ncolumnas - 1:
            acciones.remove(2)
        if row > 0 and self.matriz[row - 1, col] == 0.0:
            acciones.remove(1)
        if row < nfilas - 1 and self.matriz[row + 1, col] == 0.0:
            acciones.remove(3)
        if col > 0 and self.matriz[row, col - 1] == 0.0:
            acciones.remove(0)
        if col < ncolumnas - 1 and self.matriz[row, col + 1] == 0.0:
            acciones.remove(2)
        return acciones


# Clase para Definir lo que Experimenta le Raton
class Memoria(object):
    def __init__(self, modeloNeuronal, max_memory=100, discount=0.95):
        self.modeloNeuronal = modeloNeuronal
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_acciones = modeloNeuronal.output_shape[-1]

    # Recordar movimientos anteriores
    def remember(self, intento):
        self.memory.append(intento)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # Predice siguiente movimiento segun entorno
    def predict(self, estado_entorno):
        return self.modeloNeuronal.predict(estado_entorno)[0]

    # Retorna toda la data almacenada
    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_acciones))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            estado_entorno, accion, recompensa, estado_entorno_next, fin_juego = self.memory[j]
            inputs[i] = estado_entorno
            targets[i] = self.predict(estado_entorno)
            Q_sa = np.max(self.predict(estado_entorno_next))
            if fin_juego:
                targets[i, accion] = recompensa
            else:
                targets[i, accion] = recompensa + self.discount * Q_sa
        return inputs, targets


####################################################################################################################
# METODOS
####################################################################################################################
# Dibujar el texto
def draw_text(text, size, pos_x, pos_y, surface):
    font = pg.font.SysFont('comicsans', size)
    label = font.render(text, True, (255, 255, 255))
    surface.blit(label, (pos_x, pos_y))


# Dibujar el raton
def draw_raton(surface, lab):
    posX, posY = lab.rat
    surface.blit(ratonpng, (posRatX + size_cell * posX, posRatY + size_cell * posY))
    pg.display.update()


# Dibujar el queso
def draw_queso(surface, lab):
    posX, posY = lab.target
    surface.blit(quesopng, (posRatX + size_cell * posX, posRatY + size_cell * posY))
    pg.display.update()


# Dibujar el Laberinto
def draw_matriz(surface, lab):
    matriz_array = lab.matriz
    # Dibujar Fondo del Laberinto
    pg.draw.rect(surface, (0, 0, 0),
                 (posLabX, posLabY, len(matriz_array) * size_cell,
                  len(matriz_array) * size_cell))
    # Dibujar Laberinto
    for i in range(len(matriz_array)):
        y = size_cell * i
        for j in range(len(matriz_array)):
            x = size_cell * j
            if matriz_array[i][j] > 0:
                pg.draw.rect(surface, (255, 255, 255),
                             (x + posLabX, y + posLabY, size_cell, size_cell))
                pg.draw.rect(surface, (0, 0, 0),
                             (x + posLabX, y + posLabY, size_cell, size_cell), 1)
    pg.display.update()


# Dibujar lo incluido en Ventana Pygame
def update_ventana(surface, lab):
    surface.fill((228, 35, 34))
    draw_text('Trabajo 2', 80, 10, 10, surface)
    draw_text('Grupo 4', 50, 20, 60, surface)
    draw_text('EPOCHs', 50, 360, 60, surface)
    draw_matriz(surface, lab)
    draw_queso(surface, lab)
    draw_raton(surface, lab)
    pg.display.flip()


# Cambiar milisegundos a diferentes unidades
def formato_tiempo(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f segs" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f mins" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f horas" % (h,)


# Modelo de Red Neuronal
def generar_modelo(matriz, lr=0.001):
    modeloNeuronal = Sequential()
    modeloNeuronal.add(Dense(matriz.size, input_shape=(matriz.size,)))
    modeloNeuronal.add(PReLU())
    modeloNeuronal.add(Dense(matriz.size))
    modeloNeuronal.add(PReLU())
    modeloNeuronal.add(Dense(num_act))
    modeloNeuronal.compile(optimizer='adam', loss='mse')
    return modeloNeuronal


####################################################################################################################
# Comandos para la ejecucion
####################################################################################################################

# Se creea el modelo para redes neuronales y empieza el entrenamiento
modeloNeuronal = generar_modelo(matriz)
run = True
# Loop del Juego
while run:
    n_epoch = 200
    max_memory = 1000
    data_size = 50
    # Inicializar Entorno
    lab = Laberinto(matriz)
    update_ventana(win, lab)
    # Inicializar Experiencia
    memoria = Memoria(modeloNeuronal, max_memory=max_memory)
    historia_victorias = []  # history of win/lose game
    hsize = lab.matriz.size // 2  # history window size
    win_rate = 0.0
    # Una pasada del modeloNeuronalo por la red neuronal
    for epoch in range(n_epoch):
        start_time = datetime.datetime.now()
        loss = 0.0
        rat_cell = random.choice(lab.casillas_vacias)
        lab.reset(rat_cell)
        fin_juego = False
        # Se setea el estado del entorno inicial
        estado_entorno = lab.observe()
        n_intentos = 0
        # Un training dentro del epoch
        while not fin_juego:
            acciones_permitidas = lab.acciones_permitidas()
            if not acciones_permitidas: break
            prev_estado_entorno = estado_entorno
            # Get next accion
            if np.random.rand() < exp_fat:
                accion = random.choice(acciones_permitidas)
            else:
                accion = np.argmax(memoria.predict(prev_estado_entorno))
            # Aplica accion, se da premio y se crea nuevo entorno
            estado_entorno, recompensa, estado_juego = lab.act(accion)
            if estado_juego == 'win':
                historia_victorias.append(1)
                fin_juego = True
            elif estado_juego == 'lose':
                historia_victorias.append(0)
                fin_juego = True
            else:
                fin_juego = False
            # Almacena experiencia
            intento = [prev_estado_entorno, accion, recompensa, estado_entorno, fin_juego]
            memoria.remember(intento)
            n_intentos += 1
            # Entrenamiento con red neuronal
            inputs, targets = memoria.get_data(data_size=data_size)
            h = modeloNeuronal.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = modeloNeuronal.evaluate(inputs, targets, verbose=0)
        if len(historia_victorias) > hsize:
            win_rate = sum(historia_victorias[-hsize:]) / hsize
        dt = datetime.datetime.now() - start_time
        t = formato_tiempo(dt.total_seconds())
        time.sleep(1)
        #################################################### Mi PC no me permite mostrarlo graficamente, solo en consola
        template = "Epoch: {:03d}/{:d} | Perdida: {:.4f} | #Intentos: {:d} | SumaVictorias: {:d} | Vict/Inte: {:.3f} | TiempoxEpoch: {}"
        draw_text(template.format(epoch, n_epoch - 1, loss, n_intentos, sum(historia_victorias), win_rate, t), 20, 360,
                  100 + 15 * epoch, win)
        pg.display.flip()
        time.sleep(1)
        print(template.format(epoch, n_epoch - 1, loss, n_intentos, sum(historia_victorias), win_rate, t))
        # Si se llega a encontrar 100% de Victorias
        if win_rate > 0.9: exp_fat = 0.05
        if sum(historia_victorias[-hsize:]) == hsize:
            print("Se llego a 100%% victoria en: %d" % (epoch,))
            pg.quit()
            break
pg.quit()
