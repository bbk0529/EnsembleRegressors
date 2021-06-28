from abc import abstractmethod, ABC

class NoiseGenerator(ABC) :     
    def __init__(self, distribution: str in ['uniform', 'normal'], min, max): 
        self.distribution = 
        self.min = min
        self.max = max

    @abstractmethod
    def generate_noises():
        pass

class ClippedNoises(NoiseGenerator) : 
    def __init__(self, distribution, min, max):
        super().init(distribution, min, max)

    def generate_noises():

        

class Noises :
    pass