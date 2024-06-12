import turtle
import random

def koch(size, n):
    if n == 0:
        turtle.fd(size)
    else:
        for angle in [0, 60, -120, 60]:
            turtle.left(angle)
            koch(size/3, n-1)

def main():
    turtle.speed(10000)
    turtle.setup(600, 600)
    turtle.penup()
    turtle.goto(-200, 100)
    turtle.pendown()
    turtle.pensize(2)
    level = 3
    for i in range(5):
        koch(400, level)
        turn_angle = random.randint(90, 270)
        turtle.right(turn_angle)
    turtle.hideturtle()
    turtle.done()

main()
