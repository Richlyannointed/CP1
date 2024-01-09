## SampleCode2.py to Check VPython Installation       ##
## Written by Spencer Wheaton - 28 April 2020         ##
## Output shows a red ball bouncing on a blue surface ##

from vpython import *

floor = box(length=4, height=0.5, width=4, color=color.blue)

ball = sphere(pos = vector(0,4,0), color=color.red)
ball.velocity = vector(0,-1,0)

dt = 0.01

while 1:

    rate(144)
    if ball.pos.y < 0.8:
        ball.velocity.y = -ball.velocity.y
        
    ball.pos = ball.pos + ball.velocity*dt
    ball.velocity.y = ball.velocity.y - 9.8*dt
