from math import cos, floor, sin, sqrt
import numpy as np
from lossFunctions import mse, mse_prime
import sys

import activationFunctions
from numpy import pi
from lossFunctions import mse_prime
import neuralNetwork as nn
import pygame

def print2p(x):
    print("%.2f" % x)
    
def showSpace(resolution):
    screen.fill(black)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    for i in range(resolution):
        for j in range(resolution):
            v = nn.guess([float(i)/resolution,float(j)/resolution])
            k = 255*v[0][0]
            k = max(0, k)
            k = min(255, k)
            pygame.draw.circle(screen,[255 - k,0,k,255],[(500/resolution)*i+1,(500/resolution)*j+1],300/resolution)
    pygame.display.flip()

def solveXOR(nn):
    nn.reinit()
    
    input = [[0,0],[0,1],[1,0],[1,1]]
    output = [[0],[1],[1],[0]]
    
    print("--------------------")

    for i in range(20):
        nn.train(input, output, 100)
        showSpace(20)
                
    print2p(nn.guess(input[0])[0][0])
    print2p(nn.guess(input[1])[0][0])
    print2p(nn.guess(input[2])[0][0])
    print2p(nn.guess(input[3])[0][0])

    showSpace(100)
    
def solveCenter(nn):
    nn.reinit()
    input = []
    output = []
    for i in range(50):
        r = np.random.uniform(0.6, 1.0)
        angle = np.random.uniform(0, 2 * pi)
        x = (r * sin(angle) + 1)/2
        y = (r * cos(angle) + 1)/2
        input.append([x,y])
        output.append([0])
        
    for i in range(50):
        r = np.random.uniform(0.0, 0.4)
        angle = np.random.uniform(0.0, 2 * pi)
        x = (r * sin(angle) + 1)/2
        y = (r * cos(angle) + 1)/2
        input.append([x,y])
        output.append([1])
        
    print("--------------------")

    for i in range(20):
        nn.train(input, output, 20)
        showSpace(40)
      
    for i in input:          
        print2p(nn.guess(i)[0][0])

    showSpace(100)
    
pygame.init()

size = width, height = 500, 500
speed = [2, 2]
black = 0, 0, 0

screen = pygame.display.set_mode(size)

nn = nn.NeuralNetwork((2,1))
nn.lossPrime = mse_prime
nn.loss = mse
nn.addReshapeLayer((2,1))
nn.addDenseLayer(20, activationFunctions.leakyRelu, activationFunctions.leakyReluD)
nn.addDenseLayer(1, activationFunctions.leakyRelu, activationFunctions.leakyReluD)
#solveXOR(nn) 
solveCenter(nn)
    

while(True):
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
        if event.type == pygame.KEYDOWN: solveXOR(nn)
    pygame.display.flip()