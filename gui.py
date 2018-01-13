import pygame
from PIL import Image
from resizeimage import resizeimage
import numpy as np
import scipy.misc
import json
from neuro_py import NeuralNetwork
from rec_image import DigitRecognizer

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Panel:
    def __init__(self):
        self.draw_on = False
        self.color = BLACK
        self.radius = 15
        self.screen = pygame.display.set_mode((500, 500))
        self.made_image = False
        self.quit = False

        pass

    def run(self):
        in_loop = True
        self.screen.fill(WHITE)
        while in_loop:
            e = pygame.event.wait()
            if e.type == pygame.QUIT:
                in_loop = False
                self.quit = True
            if e.type == pygame.MOUSEBUTTONDOWN:
                self.draw_on = True
            if e.type == pygame.MOUSEBUTTONUP:
                self.draw_on = False
            if e.type == pygame.MOUSEMOTION:
                if self.draw_on:
                    x, y = e.pos
                    rect = pygame.Rect(x, y, self.radius, self.radius)
                    pygame.draw.rect(self.screen, self.color, rect)
            if e.type == pygame.KEYUP:
                if e.key == pygame.K_r:
                    pygame.image.save(self.screen, 'screen.jpeg')

                    recognizer = DigitRecognizer()

                    print(recognizer.recognize_digit('screen.jpeg'))

                elif e.key == pygame.K_z:
                    rect = pygame.Rect(0, 0, 500, 500)
                    pygame.draw.rect(self.screen, WHITE, rect)

            pygame.display.flip()

        pass
