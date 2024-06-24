import winsound as sd
import pygame
import os

pygame.init()
pygame.mixer.music.load("D:/DLIP/DLIP_Python/LAB/FinalLAB/mixkit-arcade-chiptune-explosion-1691.wav")

pygame.mixer.music.play()
pygame.mixer.music.set_volume(1.0)

# Wait for the sound to finish playing
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

pygame.quit()
