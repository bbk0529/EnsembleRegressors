from abc import abstractmethod, ABC

class NoiseGenerator(ABC) : 
    @abstractmethod
    def generate_noises():
        pass

class 