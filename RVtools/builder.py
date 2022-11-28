from __future__ import annotations

import importlib
from abc import ABC, abstractmethod


class Builder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the RVData objects.
    """

    @property
    @abstractmethod
    def precursor_data(self) -> None:
        pass

    @abstractmethod
    def create_universe(self) -> None:
        pass

    @abstractmethod
    def simulate_observations(self) -> None:
        pass

    @abstractmethod
    def orbit_fitting(self) -> None:
        pass


class BaseBuilder(Builder):
    """
    This class will create an inital implementation of the creation
    process.
    """

    def __init__(self):
        """
        A fresh builder instance should contain a blank precursor_data object, which is
        used in further assembly.
        """
        self.reset()

    def reset(self):
        self._rvdata = RVData()

    @property
    def universe_type(self):
        return self._universe_type

    @universe_type.setter
    def universe_type(self, value):
        self._universe_type = value

    @property
    def universe_params(self):
        return self._universe_params

    @universe_params.setter
    def universe_params(self, value):
        self._universe_params = value

    @property
    def preobs_type(self):
        return self._preobs_type

    @preobs_type.setter
    def preobs_type(self, value):
        self._preobs_type = value

    @property
    def preobs_params(self):
        return self._preobs_params

    @preobs_params.setter
    def preobs_params(self, value):
        self._preobs_params = value

    @property
    def precursor_data(self) -> RVData:
        """
        Concrete Builders are supposed to provide their own methods for
        retrieving results. That's because various types of builders may create
        entirely different precursor_datas that don't follow the same interface.
        Therefore, such methods cannot be declared in the base Builder interface
        (at least in a statically typed programming language).

        Usually, after returning the end result to the client, a builder
        instance is expected to be ready to start producing another precursor_data.
        That's why it's a usual practice to call the reset method at the end of
        the `getRVData` method body. However, this behavior is not mandatory,
        and you can make your builders wait for an explicit reset call from the
        client code before disposing of the previous result.
        """
        precursor_data = self._rvdata
        self.reset()
        return precursor_data

    def create_universe(self):
        self._rvdata.create_universe(self.universe_type, self.universe_params)

    def simulate_observations(self):
        self._rvdata.precursor_observations(self.preobs_type, self.preobs_params)

    def orbit_fitting(self):
        self._rvdata.add("PartC1")

    def probability_of_detection(self):
        self._rvdata.add("Dataframe of p_det")


class RVData:
    """
    This class holds all the different parts of the simulation:
        Universe
        PrecursorObservations
        OrbitFit
        PDet

    It makes sense to use the Builder pattern only when your precursor_datas are quite
    complex and require extensive configuration.

    Unlike in other creational patterns, different concrete builders can produce
    unrelated precursor_datas. In other words, results of various builders may not
    always follow the same interface.
    """

    def __init__(self):
        self._universe = None

    @property
    def universe(self):
        return self._universe

    # @universe.setter
    def create_universe(self, universe_type, universe_params):
        universelib = importlib.import_module(f"RVtools.universes.{universe_type}")
        self._universe = universelib.create_universe(universe_params)

    def precursor_observations(self, preobs_type, preobs_params):
        universelib = importlib.import_module(f"RVtools.preobss.{preobs_type}")
        self._universe = universelib.create_universe(preobs_params)
        breakpoint()

    def list_parts(self) -> None:
        print(f"RVData parts: {', '.join(self.parts)}", end="")


class Director:
    """
    The Director is only responsible for executing the building steps in a
    particular sequence. It is helpful when producing precursor_datas according to a
    specific order or configuration. Strictly speaking, the Director class is
    optional, since the client can control builders directly.
    """

    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> Builder:
        return self._builder

    @builder.setter
    def builder(self, builder: Builder) -> None:
        """
        The Director works with any builder instance that the client code passes
        to it. This way, the client code may alter the final type of the newly
        assembled precursor_data.
        """
        self._builder = builder

    """
    The Director can construct several precursor_data variations using the same
    building steps.
    """

    def build_universe(self) -> None:
        self.builder.create_universe()

    def build_orbit_fits(self) -> None:
        self.builder.create_universe()
        self.builder.simulate_observations()
        self.builder.orbit_fitting()

    def build_full_info(self) -> None:
        self.builder.create_universe()
        self.builder.simulate_observations()
        self.builder.orbit_fitting()
        self.builder.probability_of_detection()


if __name__ == "__main__":
    """
    The client code creates a builder object, passes it to the director and then
    initiates the construction process. The end result is retrieved from the
    builder object.
    """

    director = Director()
    builder = BaseBuilder()
    director.builder = builder

    print("Standard basic rvdata: ")
    director.build_minimal_viable_rvdata()
    builder.precursor_data.list_parts()

    print("\n")

    print("Standard full featured rvdata: ")
    director.build_full_info()
    builder.precursor_data.list_parts()

    print("\n")

    # Remember, the Builder pattern can be used without a Director class.
    print("Custom rvdata: ")
    builder.create_universe()
    builder.simulate_observations()
    builder.precursor_data.list_parts()
