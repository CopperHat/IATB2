import pygame


class Boton(pygame.sprite.Sprite):
    def __init__(self, imagen1, x=30, y=30, *groups):
        super().__init__(*groups)
        self.imagen_normal = imagen1
        self.rect = self.imagen_normal.get_rect()
        self.rect.left, self.rect.top = (x, y)

    def update(self, pantalla):
        pantalla.blit(self.imagen_normal, self.rect)
