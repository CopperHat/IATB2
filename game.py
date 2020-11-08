from __future__ import print_function
import datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import PReLU

import pygame as pg


####################################################################################################################
# CONSTANTES
####################################################################################################################
pg.init()

# Cargar Imagenes
upcpng = pg.image.load("IMG/UPC.png")
mousepng = pg.image.load("IMG/mouse.png")
cheesepng = pg.image.load("IMG/cheese.png")

# Configuracion de la Ventana
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

# Posicion del Inicial de Raton y Queso
posRatX = posLabX + 10
posRatY = posLabY + 10

maze = np.array([
    [1., 0., 1., 1., 1., 1., 1.],
    [1., 1., 1., 0., 0., 1., 0.],
    [0., 0., 0., 1., 1., 1., 0.],
    [1., 1., 1., 1., 0., 0., 1.],
    [1., 0., 0., 0., 1., 1., 1.],
    [1., 0., 1., 1., 1., 1., 1.],
    [1., 1., 1., 0., 1., 1., 1.]
])

# Posibles colores para pintar camino de la rata
visited_mark = 0.8  # Casillas visitadas
rat_mark = 0.5  # Donde esta la rata
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Acciones
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Factor para explorar
exp_fat = 0.1


####################################################################################################################
# CLASES
####################################################################################################################
# Clase para Definir lo que es el Laberinto, el Raton y el Queso; y sus respectivas posiciones
class Qmaze(object):
    def __init__(self, maze, rat=(0, 0)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows - 1, ncols - 1)  # En que casilla esta el queso
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Laberinto invalido: El quezo esta bloqueado!")
        if not rat in self.free_cells:
            raise Exception("Posicion de Rata Invalido: Debe aparecer libre!")
        self.reset(rat)

    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state
        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # Marca casilla visitada
        valid_actions = self.valid_actions()
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:  # Accion Invalida
            mode = 'invalid'
        # Nuevo estado
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # Limpia marcas visuales
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        # Marcar camino de la rata
        row, col, valid = self.state
        canvas[row, col] = rat_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:
            return 'win'
        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)
        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)
        if row > 0 and self.maze[row - 1, col] == 0.0:
            actions.remove(1)
        if row < nrows - 1 and self.maze[row + 1, col] == 0.0:
            actions.remove(3)
        if col > 0 and self.maze[row, col - 1] == 0.0:
            actions.remove(0)
        if col < ncols - 1 and self.maze[row, col + 1] == 0.0:
            actions.remove(2)
        return actions


# Clase para Definir lo que Experimenta le Raton
class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    # Recordar movimientos anteriores
    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # Predice siguiente movimiento segun entorno
    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    # Retorna toda la data almacenada
    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            targets[i] = self.predict(envstate)
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
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
def draw_mouse(surface, qmaze):
    posX, posY = qmaze.rat
    surface.blit(mousepng, (posRatX + size_cell * posX, posRatY + size_cell * posY))
    pg.display.update()


# Dibujar el queso
def draw_cheese(surface, qmaze):
    posX, posY = qmaze.target
    surface.blit(cheesepng, (posRatX + size_cell * posX, posRatY + size_cell * posY))
    pg.display.update()


# Dibujar el Laberinto
def draw_maze(surface, qmaze):
    maze_array = qmaze.maze
    # Dibujar Fondo del Laberinto
    pg.draw.rect(surface, (0, 0, 0),
                 (posLabX, posLabY, len(maze_array) * size_cell,
                  len(maze_array) * size_cell))
    # Dibujar Laberinto
    for i in range(len(maze_array)):
        y = size_cell * i
        for j in range(len(maze_array)):
            x = size_cell * j
            if maze_array[i][j] > 0:
                pg.draw.rect(surface, (255, 255, 255),
                             (x + posLabX, y + posLabY, size_cell, size_cell))
                pg.draw.rect(surface, (0, 0, 0),
                             (x + posLabX, y + posLabY, size_cell, size_cell), 1)
    pg.display.update()


# Dibujar lo incluido en Ventana Pygame
def update_surface(surface, qmaze):
    surface.fill((228, 35, 34))
    draw_text('Trabajo 2', 80, 10, 10, surface)
    draw_text('Grupo 4', 50, 20, 60, surface)
    draw_text('Pruebas', 50, 360, 100, surface)
    draw_maze(surface, qmaze)
    draw_cheese(surface, qmaze)
    draw_mouse(surface, qmaze)
    pg.display.flip()


# Inicio del Juego
def play_game(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # Siguiente accion
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # Aplica accion, se da premio y se crea nuevo entorno
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


# Entrenamiento del IA
def qtrain(model, maze, surface, **opt):
    global exp_fat
    n_epoch = opt.get('n_epoch', 200)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    start_time = datetime.datetime.now()
    # Inicializar Entorno
    qmaze = Qmaze(maze)
    # Inicializar Experiencia
    experience = Experience(model, max_memory=max_memory)
    win_history = []  # history of win/lose game
    hsize = qmaze.maze.size // 2  # history window size
    win_rate = 0.0
    # Una pasada del modelo por la red neuronal
    for epoch in range(n_epoch):
        loss = 0.0
        rat_cell = random.choice(qmaze.free_cells)
        qmaze.reset(rat_cell)
        game_over = False
        # Se setea el estado del entorno inicial
        envstate = qmaze.observe()
        n_episodes = 0
        # Un training dentro del epoch
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action
            if np.random.rand() < exp_fat:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))
            # Aplica accion, se da premio y se crea nuevo entorno
            envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False
            # Almacena experiencia
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1
            # Entrenamiento con red neuronal
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)
        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        #################################################### Mi PC no me permite mostrarlo graficamente, solo en consola
        template = "Entrenamiento: {:03d}/{:d} | Perdida: {:.4f} | #Intentos: {:d} | VictoriasTotales: {:d} | %VictoriasTotales: {:.3f} | TiempoTotal: {}"
        # draw_text(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t), 5, 360, 120 + epoch*10, surface)
        # pg.display.flip()
        print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t))
        # Si se llega a encontrar 100% de Victorias
        if win_rate > 0.9: exp_fat = 0.05
        if sum(win_history[-hsize:]) == hsize:
            print("Se llego a 100%% victoria en: %d" % (epoch,))
            break


# Cambiar milisegundos a diferentes unidades
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f segundos" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutos" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


# Modelo de Red Neuronal
def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model


####################################################################################################################
# Comandos para la ejecucion
####################################################################################################################
# Se crea la clase qmaze
qmaze = Qmaze(maze)
# Se dibuja la ventana
update_surface(win, qmaze)

# Se creea el modelo para redes neuronales y empieza el entrenamiento
model = build_model(maze)
qtrain(model, maze, win, epochs=200, max_memory=8 * maze.size, data_size=32)

for event in pg.event.get():
    if event.type == pg.QUIT:
        pg.quit()
pg.quit()
