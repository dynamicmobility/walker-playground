import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from matplotlib.collections import Collection

# This class represents a hybrid system with a single domain
# The system evolves according to the dynamics f until it approaches the switching set (defined by the guard function crossing 0)
# When the system trajectory intersects the switching surface, the system undergoes a reset
class SingleDomainSystem(ABC):

    @abstractmethod
    def f(self, t: float, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        pass

    # The switching surface is represented as the set of states x such that guard(t, x)=0
    @abstractmethod
    def guard(self, t: float, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        pass

    # The switching surface is represented as the set of states x such that guard(t, x)=0
    @abstractmethod
    def reset(self, t: float, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        pass

    # Returns a visualization of the system at a particular state. Can be used to plot the joint configuration of a robot
    @abstractmethod
    def draw_system(self, t: float, x: ArrayLike, *args, **kwargs) -> Collection:
        pass