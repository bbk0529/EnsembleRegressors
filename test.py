class Parent():
    def __init__(self):
        print("Parent")

class ChildA(Parent) : 
    def __init__(self):        
        print("ChildA")


class ChildB(Parent) : 
    def __init__(self):
        print("\n")
        super().__init__()
        print("ChildB")
        super().__init__()


if __name__ == '__main__' :
    personA = ChildA()
    personB = ChildB()