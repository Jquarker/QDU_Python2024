import turtle
import math

def drawdouble(n,i):
    n1 = n / 2
    a = (180 * (n - 2) / n)
    b = 180 - a
    c = b / 2
    d = 180 - (c * (n / 2 - 1))
    d1 = (d / 180) * math.pi
    c1 = (c / 180) * math.pi
    e = (math.sin(c1) / math.sin(d1)) * 500
    while i < n1:
        turtle.forward(500)
        turtle.left(90)
        turtle.penup()
        turtle.forward(e)
        turtle.pendown()
        turtle.left(90)
        turtle.forward(500)
        turtle.left(180 - 180 / n1)
        i+=1

def drawsingle(n):
    turtle.pendown()  # 放下画笔
    turtle.begin_fill()  # 画笔开始填充
    for i in range(n):
        turtle.fd(500)
        turtle.left(180-180/n)
    turtle.end_fill()

n = int(input("请输入星星的角数: "))
i = 0#循环变量
if(n%2!=0):
    drawsingle(n)
else:
    drawdouble(n,i)