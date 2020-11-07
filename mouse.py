import sys


class Mouse:
    def __init__(self, name, posX, posY, energy):
        self.name = name
        self.posX = posX
        self.posY = posY
        self.energy = energy

    def move(self, array, direction):
        if direction == 'UP' and self.posY != 0:  # Se mueve hacia arriba
            if array[self.posY - 1][self.posX]:
                self.posY = self.posY - 1
                self.energy = self.energy - 1
        elif direction == 'DOWN' and self.posY != len(array) - 1:  # Se mueve hacia abajo
            if array[self.posY + 1][self.posX]:
                self.posY = self.posY + 1
                self.energy = self.energy - 1
        elif direction == 'LEFT' and self.posX != 0:  # Se mueve hacia la izquierda
            if array[self.posY][self.posX-1]:
                self.posX = self.posX - 1
                self.energy = self.energy - 1
        elif direction == 'RIGHT' and self.posX != len(array) - 1:  # Se mueve hacia la derecha
            if array[self.posY][self.posX+1]:
                self.posX = self.posX + 1
                self.energy = self.energy - 1
