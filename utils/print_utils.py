from utils.color import Color

def partition(len):
    partition = ''
    for i in range(100):
        partition += '-'

    return partition

def highlight(color, message):
    if color == 'white':
        print(Color.CSELECTED + message + Color.CEND)
    elif color == 'green':
        print(Color.CGREEN + message + Color.CEND)
    elif color == 'red':
        print(Color.CRED2 + message + Color.CEND)
    elif color == 'violet':
        print(Color.CVIOLET2 + message + Color.CEND)
    elif color == 'blue':
        print(Color.CBLUE2 + message + Color.CEND)

