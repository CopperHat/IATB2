import sys


class Mouse:
    def __init__(self, name, posX, posY, energy):
        self.name = name
        self.posX = posX
        self.posY = posY
        self.energy = energy

    def move(self, direction):
        if direction == 'UP':  # Se mueve hacia arriba
            self.posY = self.posY - 1
            self.energy = self.energy - 1
        elif direction == 'DOWN':  # Se mueve hacia abajo
            self.posY = self.posY + 1
            self.energy = self.energy - 1
        elif direction == 'LEFT':  # Se mueve hacia la izquierda
            self.posX = self.posX - 1
            self.energy = self.energy - 1
        elif direction == 'RIGHT':  # Se mueve hacia la derecha
            self.posX = self.posX + 1
            self.energy = self.energy - 1