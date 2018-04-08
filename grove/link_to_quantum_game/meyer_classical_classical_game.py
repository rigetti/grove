import pygame
import numpy as np
import pyquil.api as api
from pyquil.gates import *
from pyquil.quil import Program

# define colors
blue = (0, 0, 255)
rect_Q_color = (255, 127.5, 127.5)
rect_P_color = (127.5, 127.5, 255)
# boolean controling when game stops
game_over = False
# frames per second
fps = 60
# screen dimensions
size_x = 1000
size_y = 625
size = (size_x, size_y)
# rectangle coordinates
rect_h = 50
rect_Q_x = 0
rect_Q_y = 0
rect_P_x = 0
rect_P_y = size_y - rect_h
# fire coordinates
fire_x = -20
fire_y = -20
fire_width = 5
fire_length = 30
fire_y_change = 0
# track move_numbers, neither player has made a move at beginning
move_Q = 0
move_P = 0
# track strategy parameters
Q_prob1 = None
picard_prob = None
Q_prob2 = None
# initialize destruction log's parameters
dest_y = size_y//2
dest_width = 30
# boolean controling pyquil program run
run_quantum_program = False


# generate unitary matrix
def U_(a, b):
    return np.array([[a, b], [b, -a]])

# define flip (X) and not flip (I) operators
X_ = np.array([[0, 1], [1, 0]])
I_ = np.array([[1, 0], [0, 1]])

# initialize pygame
pygame.init()
# setup screen
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Meyer penny classical/classical game demo")
# setup game clock
clock = pygame.time.Clock()
# define font
font = pygame.font.SysFont('Calibri', 25, True)

while not game_over:

    # track moves
    first_move = ((move_Q == 0) and (move_P == 0))
    second_move = ((move_Q == 1) and (move_P == 0))
    third_move = ((move_Q == 1) and (move_P == 1))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if first_move:
                    fire_x = (2 * rect_Q_x + rect_h)/2
                    fire_y = rect_Q_y + rect_h
                elif second_move:
                    fire_x = (2 * rect_P_x + rect_h)/2
                    fire_y = rect_P_y - fire_length
                elif third_move:
                    fire_x = (2 * rect_Q_x + rect_h)/2
                    fire_y = rect_Q_y + rect_h
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                if first_move:
                    Q_prob1 = rect_Q_x / (size_x - rect_h)
                    move_Q += 1
                    fire_y_change = 15
                elif second_move:
                    picard_prob = rect_P_x / (size_x - rect_h)
                    move_P += 1
                    fire_y_change = -15
                elif third_move:
                    Q_prob2 = rect_Q_x / (size_x - rect_h)
                    move_Q += 1
                    fire_y_change = 15

    # set game color
    screen.fill(blue)

    # draw rectangles
    pygame.draw.rect(screen, rect_Q_color, (rect_Q_x, rect_Q_y, rect_h, rect_h))
    pygame.draw.rect(screen, rect_P_color, (rect_P_x, rect_P_y, rect_h, rect_h))

    # draw fire
    fire_y += fire_y_change
    pygame.draw.rect(screen, (255, 125, 125), [fire_x, fire_y, fire_width, fire_length])

    ## motions
    pressed = pygame.key.get_pressed()
    # rectangle motion
    if pressed[pygame.K_LEFT]:
        if first_move:
            rect_Q_x -= 5
        elif second_move:
            rect_P_x -= 5
        elif third_move:
            rect_Q_x -= 5
    if pressed[pygame.K_RIGHT]:
        if first_move:
            rect_Q_x += 5
        elif second_move:
            rect_P_x += 5
        elif third_move:
            rect_Q_x += 5

    # keep rectangle within screen
    if rect_Q_x > size_x - rect_h:
        rect_Q_x = size_x - rect_h
    elif rect_Q_x < 0:
        rect_Q_x = 0
    if rect_P_x > size_x - rect_h:
        rect_P_x = size_x - rect_h
    elif rect_P_x < 0:
        rect_P_x = 0

    if first_move:
        # display Q's potential 1st choice
        text_rect_Q_x = font.render("Prob(flip): " + str(rect_Q_x / (size_x - rect_h)), True, (0, 0, 0))
        screen.blit(text_rect_Q_x, [150, 50])
        text_size_ = font.render("Prob(not flip): " + str(1 - rect_Q_x / (size_x - rect_h)), True, (0, 0, 0))
        screen.blit(text_size_, [150, 20])
    elif second_move:
        # display Q's 1st choice
        text_rect_Q_x = font.render("Prob(flip): " + str(Q_prob1), True, (0, 0, 0))
        screen.blit(text_rect_Q_x, [150, 50])
        text_size_ = font.render("Prob(not flip): " + str(1 - Q_prob1), True, (0, 0, 0))
        screen.blit(text_size_, [150, 20])
        # display P's 1st choice
        text_rect_P_x = font.render("Prob(flip): " + str(rect_P_x / (size_x - rect_h)), True, (0, 0, 0))
        screen.blit(text_rect_P_x, [150, size_y - 50])
        text_size_ = font.render("Prob(not flip): " + str(1 - rect_P_x / (size_x - rect_h)), True, (0, 0, 0))
        screen.blit(text_size_, [150, size_y - 100])
    elif third_move:
        # display Q's 1st choice
        text_rect_Q_x = font.render("Prob(flip): " + str(Q_prob1), True, (0, 0, 0))
        screen.blit(text_rect_Q_x, [150, 50])
        text_size_ = font.render("Prob(not flip): " + str(1 - Q_prob1), True, (0, 0, 0))
        screen.blit(text_size_, [150, 20])
        # display P's 1st choice
        text_rect_P_x = font.render("Prob(flip): " + str(picard_prob), True, (0, 0, 0))
        screen.blit(text_rect_P_x, [150, size_y - 50])
        text_size_ = font.render("Prob(not flip): " + str(1 - picard_prob), True, (0, 0, 0))
        screen.blit(text_size_, [150, size_y - 100])
        # display Q's potential 2nd choice
        text_rect_Q_x = font.render("Prob(flip): " + str(rect_Q_x / (size_x - rect_h)), True, (0, 0, 0))
        screen.blit(text_rect_Q_x, [size_x - 400, 50])
        text_size_ = font.render("Prob(not flip): " + str(1 - rect_Q_x / (size_x - rect_h)), True, (0, 0, 0))
        screen.blit(text_size_, [size_x - 400, 20])
    else:
        ### Display all choices made
        # display Q's 1st choice
        text_rect_Q_x = font.render("Prob(flip): " + str(Q_prob1), True, (0, 0, 0))
        screen.blit(text_rect_Q_x, [150, 50])
        text_size_ = font.render("Prob(not flip): " + str(1 - Q_prob1), True, (0, 0, 0))
        screen.blit(text_size_, [150, 20])
        # display P's 1st choice
        text_rect_P_x = font.render("Prob(flip): " + str(picard_prob), True, (0, 0, 0))
        screen.blit(text_rect_P_x, [150, size_y - 50])
        text_size_ = font.render("Prob(not flip): " + str(1 - picard_prob), True, (0, 0, 0))
        screen.blit(text_size_, [150, size_y - 100])
        # display Q's 2nd choice
        text_rect_Q_x = font.render("Prob(flip): " + str(Q_prob2), True, (0, 0, 0))
        screen.blit(text_rect_Q_x, [size_x - 400, 50])
        text_size_ = font.render("Prob(not flip): " + str(1 - Q_prob2), True, (0, 0, 0))
        screen.blit(text_size_, [size_x - 400, 20])
        # program is now ready to be run
        run_quantum_program = True

    if run_quantum_program:
        # draw DESTRUCTION! log
        pygame.draw.rect(screen, (0, 0, 0), (0, dest_y, size_x, dest_width))
        ### Carry out program
        # initialize program
        qvm = api.QVMConnection()
        p = Program()

        ##### Q's 1st stochastic move
        # set the parameters for the random move
        prob = Q_prob1
        # create a dummy gate
        a = np.random.rand()
        b = np.sqrt(1 - np.square(a))
        Uf_ = U_(a, b)
        # perform the random move
        p.defgate("Uf_Q1", Uf_)
        p.inst(("Uf_Q1", 0))
        p.define_noisy_gate("Uf_Q1", [0], [np.sqrt(prob) * X_, np.sqrt(1 - prob) * I_])

        ##### Picard's stochastic move
        # set the parameters for the random move
        prob = picard_prob
        # create a dummy gate
        a = np.random.rand()
        b = np.sqrt(1 - np.square(a))
        Uf_ = U_(a, b)
        # perform the random move
        p.defgate("Uf_P", Uf_)
        p.inst(("Uf_P", 0))
        p.define_noisy_gate("Uf", [0], [np.sqrt(prob) * X_, np.sqrt(1 - prob) * I_])

        ##### Q's 2nd stochastic move
        # set the parameters for the random move
        prob = Q_prob2
        # create a dummy gate
        a = np.random.rand()
        b = np.sqrt(1 - np.square(a))
        Uf_ = U_(a, b)
        # perform the random move
        p.defgate("Uf_Q2", Uf_)
        p.inst(("Uf_Q2", 0))
        p.define_noisy_gate("Uf_Q2", [0], [np.sqrt(prob) * X_, np.sqrt(1 - prob) * I_])

        ##### Measurement
        p.inst(MEASURE(0, [0]))
        result = qvm.run(p, [0], trials=100)

        result_1s = (len([i for i in result if i == [1]]))
        result_0s = (len([i for i in result if i == [0]]))

        Picard_score = result_1s
        Q_score = result_0s

        # move DESTRUCTION! log according to scores
        delta_y = size_y/1000
        dest_y -= int(delta_y * Picard_score)
        dest_y += int(delta_y * Q_score)

        if dest_y > size_y - dest_width:
            dest_y = size_y - dest_width
        elif dest_y < 0:
            dest_y = 0

        text_1s = font.render("Picard's score: " + str(result_1s), True, (0, 0, 0))
        screen.blit(text_1s, [size_x/2, size_y/2])
        text_0s = font.render("Q's score: " + str(result_0s), True, (0, 0, 0))
        screen.blit(text_0s, [size_x/2, size_y/2 + 50])

        # avoid repeating program
        run_quantum_program = False

    pygame.display.flip()

    clock.tick(fps)
