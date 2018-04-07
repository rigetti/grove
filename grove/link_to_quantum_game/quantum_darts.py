import pygame
import numpy as np
import pyquil.api as api
from pyquil.gates import *
from pyquil.quil import Program

# open a QVM connection
qvm = api.QVMConnection()

# define colors
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
red_blue = (255, 0, 255)
red_green = (255, 255, 0)
blue_green = (0, 255, 255)
black = (0, 0, 0)
color_code = 0
dartboard_A_color = (127.5, 127.5, 127.5)
# boolean controling when game stops
game_over = False
color_rect = True
# frames per second
fps = 60
# screen dimensions
size_x = 1000
size_y = 625
size = (size_x, size_y)
# dartboard parameters
radius_outer = 60
width = 10
radius_1st_inner = radius_outer - 2 * width  # 60
radius_2nd_inner = radius_outer - 4 * width  # 20
dartboard_offset = 150

# initialize program with random measurement of either 0 or 1
p = Program(H(0)).measure(0, [0])
results_A = qvm.run(p, [0])
results_B = []

# initialize pygame
pygame.init()
# setup screen
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Quantum darts")
# setup game clock
clock = pygame.time.Clock()
# define font
font = pygame.font.SysFont('Calibri', 25, True)

# keep track of score and score text, as well as attempts
str_text_score = ''
score = 0
attempts = 0

while not game_over:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # comparison stage
                if color_code % 3 == 2:
                    mouse_pos = pygame.mouse.get_pos()
                    print ("Clicked Mouse pos: ", mouse_pos)
                    if (mouse_pos[0] >= size_x//2 - radius_outer) and (mouse_pos[0] <= size_x//2 + radius_outer):
                        if (mouse_pos[1] >= size_y//2 - dartboard_offset - radius_outer) and (mouse_pos[1] <= size_y//2 - dartboard_offset + radius_outer):
                            if results_A == [[0]]:
                                str_text_score = "SCORE!!"
                                score += 1
                            elif results_A == [[1]]:
                                str_text_score = "........POOR............"
                        elif (mouse_pos[1] >= size_y//2 + dartboard_offset - radius_outer) and (mouse_pos[1] <= size_y//2 + dartboard_offset + radius_outer):
                            if results_A == [[0]]:
                                str_text_score = "........POOR............"
                            elif results_A == [[1]]:
                                str_text_score = "SCORE!!"
                                score += 1
                        else:
                            str_text_score = "Missed the target...MAJOR FAIL!!"
                            score -= 1
                    else:
                        str_text_score = "Missed the target...MAJOR FAIL!!"
                        score -= 1
                    attempts += 1
                color_code += 1
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                pass

    # set game color
    if color_code % 3 == 0:
        # make scren blue
        screen.fill(blue)
        # empty out results from dartboard_B
        if len(results_B) > 0:
            results_B.pop()
        # run the common parts of the program
        p = Program()
        if score < 10:
            a = np.sqrt(1/2.)
        elif (score >= 10) and (score < 20):
            a = np.sqrt(3/4.)
        else:
            a = np.sqrt(1.)
        b = np.sqrt(1 - np.square(a))
        Uf_ = np.array([[a, b], [b, -a]])
        p.defgate("Uf", Uf_)
        # set dartboard_A center, and calculate results_B in either case
        if results_A == [[0]]:
            dartboard_A_center = [size_x//2, size_y//2 - dartboard_offset]
            # run the program for this case
            p.inst(I(0))
            p.inst(("Uf", 0))
            p.measure(0, [0])
            results_B = qvm.run(p, [0])
        elif results_A == [[1]]:
            dartboard_A_center = [size_x//2, size_y//2 + dartboard_offset]
            # run the program for this case
            p.inst(X(0))
            p.inst(("Uf", 0))
            p.measure(0, [0])
            results_B = qvm.run(p, [0])

    elif color_code % 3 == 1:
        screen.fill(red)
        if len(results_A) > 0:
            results_A.pop()
        # run the common parts of the program
        p = Program()
        if score < 10:
            a = np.sqrt(1/2.)
        elif (score >= 10) and (score < 20):
            a = np.sqrt(3/4.)
        else:
            a = np.sqrt(1.)
        b = np.sqrt(1 - np.square(a))
        Uf_ = np.array([[a, b], [b, -a]])
        p.defgate("Uf", Uf_)
        # set dartboard_B centers
        if results_B == [[0]]:
            dartboard_B_center = [size_x//2 + dartboard_offset, size_y//2]
            # run the program for this case
            p.inst(I(0))
            p.inst(("Uf", 0))
            p.measure(0, [0])
            results_A = qvm.run(p, [0])
        elif results_B == [[1]]:
            dartboard_B_center = [size_x//2 - dartboard_offset, size_y//2]
            # run the program for this case
            p.inst(X(0))
            p.inst(("Uf", 0))
            p.measure(0, [0])
            results_A = qvm.run(p, [0])

    elif color_code % 3 == 2:
        screen.fill(blue)

    pressed = pygame.mouse.get_pressed()
    mouse_pos = pygame.mouse.get_pos()

    text_score = font.render(str_text_score, True, black)
    screen.blit(text_score, [100, size_y//2])

    # draw dartboard
    if color_code % 3 == 0:
        pygame.draw.circle(screen, dartboard_A_color, dartboard_A_center, radius_outer, width)
        pygame.draw.circle(screen, dartboard_A_color, dartboard_A_center, radius_1st_inner, width)
        pygame.draw.circle(screen, dartboard_A_color, dartboard_A_center, radius_2nd_inner, width)
    elif color_code % 3 == 1:
        pygame.draw.circle(screen, dartboard_A_color, dartboard_B_center, radius_outer, width)
        pygame.draw.circle(screen, dartboard_A_color, dartboard_B_center, radius_1st_inner, width)
        pygame.draw.circle(screen, dartboard_A_color, dartboard_B_center, radius_2nd_inner, width)
    elif color_code % 3 == 2:
        pygame.draw.circle(screen, dartboard_A_color, dartboard_A_center, radius_outer, 10)

    # display Score
    text_score = font.render("Score: " + str(score), True, black)
    screen.blit(text_score, [size_x - 150, 20])
    text_attempts = font.render("Attempts: " + str(attempts), True, black)
    screen.blit(text_attempts, [size_x - 150, 40])
    # text_resultsA = font.render("Results_A: " + str(Results_A), True, black)
    # screen.blit(text_resultsA, [size_x - 150, 80])
    # text_resultsB = font.render("Results_A: " + str(Results_B), True, black)
    # screen.blit(text_resultsA, [size_x - 150, 100])

    pygame.display.flip()

    clock.tick(fps)
