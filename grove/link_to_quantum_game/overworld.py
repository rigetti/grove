import pygame

import numpy

from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import I, H, CZ, RY

sim = True
trials = 1000

pygame.init()

# setup screen
L2 = 5
strip = 3
L = 2*L2+strip
cell_size = 48
size_x = cell_size*L
size_y = cell_size*L
size = (size_x,size_y)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Overworld")
render_buffer = pygame.Surface(size)

# declare colors
Red = (255,0,0)
Green = (0,255,0)
Blue = (50,50,255)
Yellow = (255,255,51)
White = (255,255,255)
Black = (0,0,0)

done = False
clock = pygame.time.Clock()


# make sticky figure
def sticky(x,y):
    pygame.draw.ellipse(screen,Blue,[x-7.5-cell_size/2,y-7.5-cell_size/2,15,15])
    
# sticky location
x_coord = (L2+2)*cell_size
y_coord = (L2+2)*cell_size
x_change = 0
y_change = 0

# randomly generate world
def calculate_probs () :
    
    results = {}
    for basis in ["XX","ZZ","XZ","ZX"]:
        
        if sim:
            engine = api.QVMConnection(use_queue=True)
        else:
            engine = api.QPUConnection(device)   

        script = Program()
        
        # bell state
        script.inst( H(0) )
        script.inst( H(5) )
        script.inst( CZ(0,5) )
        script.inst( RY( numpy.pi/4, 0 ) )
        script.inst( H(0) )
        script.inst( H(5) )
        
        # set up bases
        
        if basis[0]=="X":
            script.inst( H(0) )
            
        if basis[1]=="X":
            script.inst( H(5) ) 

        # get and store results
        results[basis] = engine.run_and_measure(script, [0,5], trials=trials)
    
    p = {}
    
    p["XI"] = 0
    for basis in ["XX","XZ"]:
        for sample in results[basis]:
            p["XI"] += sample[0]/(2*trials)
                           
    p["ZI"] = 0
    for basis in ["ZX","ZZ"]:
        for sample in results[basis]:
            p["ZI"] += sample[0]/(2*trials)
                           
    p["IX"] = 0
    for basis in ["XX","ZX"]:
        for sample in results[basis]:
            p["IX"] += sample[1]/(2*trials)
                           
    p["IZ"] = 0
    for basis in ["XZ","ZZ"]:
        for sample in results[basis]:
            p["IZ"] += sample[1]/(2*trials)
       
    for basis in ["XX","ZZ","XZ","ZX"]:
        p[basis] = 0
        for sample in results[basis]:
            p[basis] += (sample[0]==sample[1])/trials
        print(basis,p[basis])

    #p["XX"] = 0
    #p["XZ"] = 1
    #p["ZX"] = 1
    #p["ZZ"] = 1
            
    return p

p = calculate_probs()
          
# initialize world
world = [["dark" for x in range(L)] for y in range(L)]
for dx in range(3):
    for dy in range(3):
        world[L2+dx][L2+dy] = "grass"
        
def get_cell ( x_coord, y_coord ) :
    
    x = round( x_coord/cell_size - 1)
    y = round( y_coord/cell_size - 1)
    
    return x, y

pairs = {}
for xp in range(L2):
    for yp in range(L2):
        pairs[ (xp,yp) ] = []
        
while not done:
    
    # reveal neighbouring cells
    x, y = get_cell( x_coord, y_coord )
    
    for ds in [[0,+1],[0,-1],[+1,0],[-1,0]]:
            xx = x+ds[0]
            yy = y+ds[1]
            if xx in range(L2,L2+3) and yy in range(L):
                world[xx][yy] = "grass"
            elif yy in range(L2,L2+3) and xx in range(L):
                world[xx][yy] = "grass"
            elif xx in range(L) and yy in range(L):
                if world[xx][yy]=="dark":
                    
                    if xx<L2:
                        qubit = 0
                        xp = xx
                    else:
                        qubit = 1
                        xp = xx-L2-strip
                        
                    if yy<L2:
                        basis = "x"
                        yp = yy
                    else:
                        basis = "z"
                        yp = yy-L2-strip
                        
                    if pairs[(xp,yp)]==[]: # if no measurements have been made on this pair
                        world[xx][yy] = numpy.random.choice(["trees","grass"]) # choose randomly
                    else:
                        if len(pairs[(xp,yp)])==1: # if one measurement has been made on this pair
                            if pairs[xp,yp][0]["qubit"]==qubit: # if it is the same qubit, different basis
                                world[xx][yy] = numpy.random.choice(["trees","grass"]) # choose randomly
                                prev_xx = xx
                                prev_yy = (yy+L2+strip)*(basis=="x") + (yy-L2-strip)*(basis=="z")
                                world[prev_xx][prev_yy] = "dark" # restore darkness for previous
                            else: # if different qubit
                                # make it agree, or disagree for both x
                                prev_xx = (xx+L2+strip)*(qubit==0) + (xx-L2-strip)*(qubit==1)
                                prev_yy = yy
                                # choose with probs from simulation
                                bases = str.upper(pairs[xp,yp][0]["basis"]+basis)
                                prob = p[bases]
                                if numpy.random.random()<prob:
                                    world[xx][yy] = world[prev_xx][prev_yy]
                                else:
                                    if world[prev_xx][prev_yy]=="grass":
                                        world[xx][yy] = "trees"
                                    else:
                                        world[xx][yy] = "grass"
                                            
                        else: # if more than one measurement has been made on this pair
                            world[xx][yy] = numpy.random.choice(["trees","grass"]) # choose randomly
                            # see if it has previously been measured
                            previously_measured = False
                            for measurement in pairs[(xp,yp)]:
                                previously_measured = previously_measured or (measurement["qubit"]==qubit)
                            if previously_measured:
                                prev_xx = xx
                                prev_yy = (yy+L2+strip)*(basis=="x") + (yy-L2-strip)*(basis=="z")
                                world[prev_xx][prev_yy] = "dark" # restore darkness for previous
                                
                    
                    # record the measurement
                    pairs[xp,yp].append({"qubit":qubit,"basis":basis})
                

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x_change = -5
            if event.key == pygame.K_RIGHT:
                x_change = 5
            if event.key == pygame.K_UP:
                y_change = -5
            if event.key == pygame.K_DOWN:
                y_change =     5
            if event.key == pygame.K_SPACE:
                fire_x = x_coord
                fire_y = y_coord
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                x_change = 0
            if event.key == pygame.K_RIGHT:
                x_change = 0
            if event.key == pygame.K_UP:
                y_change = 0
            if event.key == pygame.K_DOWN:
                y_change =     0
            if event.key == pygame.K_SPACE:
                fire_x_change = 15    
                
    # change coord
    new_x_coord = x_coord + x_change
    new_y_coord = y_coord + y_change
    
    # find new cell
    new_x, new_y = get_cell ( new_x_coord, new_y_coord )

    # see if new cell can be entered
    if new_x in range(L) and new_y in range(L):
        if world[new_x][new_y]=="grass":
            x_coord = new_x_coord
            y_coord = new_y_coord
            
    
    render_buffer.fill((0, 0, 0)) 
    
    for x in range(L):
        for y in range(L):
            render_buffer.blit(pygame.image.load(world[x][y]+'.png'), (cell_size*x,cell_size*y))
    
    screen.blit(pygame.transform.scale(render_buffer, size), (0, 0))

    ## sticky movement
    sticky(x_coord,y_coord)

    
    pygame.display.flip()
    
    clock.tick(60)

pygame.quit()
