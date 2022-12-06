from __future__ import annotations

import hashlib
import importlib
from abc import ABC, abstractmethod
from pathlib import Path

import dill

from RVtools.orbitfit import OrbitFit
from RVtools.pdet import PDet
from RVtools.preobs import PreObs


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
    def simulate_rv_observations(self) -> None:
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
        self.rvdata = RVData()
        self.cache_universe = False
        self.cache_preobs = False
        self.cache_orbitfit = False

    @property
    def run_title(self):
        return self._run_title
        # self.run_title = f"{random.getrandbits(128):032x}"[:8]

    @run_title.setter
    def run_title(self, run_title):
        self._run_title = run_title
        self.rvdata.run_title = run_title
        # self._rvdata.run_title = run_title

    @property
    def cache_universe(self):
        return self._cache_universe

    @cache_universe.setter
    def cache_universe(self, val):
        self._cache_universe = val
        self.rvdata.cache_universe = val

    @property
    def cache_preobs(self):
        return self._cache_preobs

    @cache_preobs.setter
    def cache_preobs(self, val):
        self._cache_preobs = val
        self.rvdata.cache_preobs = val

    @property
    def cache_orbitfit(self):
        return self._cache_orbitfit

    @cache_orbitfit.setter
    def cache_orbitfit(self, val):
        self._cache_orbitfit = val
        self.rvdata.cache_orbitfit = val

    # @property
    # def universe_type(self):
    #     return self._universe_type

    # @universe_type.setter
    # def universe_type(self, value):
    #     self._universe_type = value

    # @property
    # def universe_params(self):
    #     return self._universe_params

    # @universe_params.setter
    # def universe_params(self, value):
    #     self._universe_params = value

    # @property
    # def preobs_type(self):
    #     return self._preobs_type

    # @preobs_type.setter
    # def preobs_type(self, value):
    #     self._preobs_type = value

    # @property
    # def preobs_params(self):
    #     return self.preobs_params

    # @preobs_params.setter
    # def preobs_params(self, value):
    #     self.preobs_params = value

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
        precursor_data = self.rvdata
        self.reset()
        return precursor_data

    def create_universe(self):
        self.rvdata.create_universe(self.universe_params)

    def simulate_rv_observations(self):
        self.rvdata.precursor_observations(self.preobs_params)

    def orbit_fitting(self):
        self.rvdata.orbit_fitting(self.orbitfit_params)

    def probability_of_detection(self):
        self.rvdata.calc_pdet(self.pdet_params)


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
        self.universe = None
        self.basepath = Path(".cache/")

    def create_universe(self, universe_params):
        universe_type = universe_params["universe_type"]
        if self.cache_universe:
            assert hasattr(self, "run_title"), "run_title must exist"

            # Create hash from the universe parameters
            hashstr = ""
            for key in universe_params.keys():
                hashstr += f"{key}:{universe_params[key]}\n"
            self.universe_hash = hashlib.sha1(hashstr.encode("UTF-8")).hexdigest()[:8]

            self.universe_path = Path(
                self.basepath, self.run_title, self.universe_hash, "universe.p"
            )
            self.run_dir = self.universe_path.parents[0]
            universe_params["cache_path"] = self.universe_path
            if self.universe_path.exists():
                # Load it
                with open(self.universe_path, "rb") as f:
                    self.universe = dill.load(f)
            else:
                universelib = importlib.import_module(
                    f"RVtools.cosmoses.{universe_type}"
                )
                self.universe = universelib.create_universe(universe_params)

                # Make directory and cache it
                self.run_dir.mkdir(parents=True, exist_ok=True)
                with open(self.universe_path, "wb") as f:
                    dill.dump(self.universe, f)
        else:
            universelib = importlib.import_module(f"RVtools.cosmoses.{universe_type}")
            self.universe = universelib.create_universe(universe_params)

    def precursor_observations(self, preobs_params):
        if self.cache_preobs:
            self.preobs_path = Path(self.run_dir, "preobs.p")
            preobs_params["cache_path"] = self.preobs_path
            if self.preobs_path.exists():
                # Load it
                with open(self.preobs_path, "rb") as f:
                    self.preobs = dill.load(f)
            else:
                self.preobs = PreObs(preobs_params, self.universe)

                # Make directory and cache it
                with open(self.preobs_path, "wb") as f:
                    dill.dump(self.preobs, f)
        else:
            self.preobs = PreObs(preobs_params, self.universe)

    def orbit_fitting(self, orbitfit_params):
        if self.cache_orbitfit:
            self.orbitfit_path = Path(self.run_dir, "orbitfit.p")
            orbitfit_params["cache_path"] = self.orbitfit_path
            if self.orbitfit_path.exists():
                # Load it
                with open(self.orbitfit_path, "rb") as f:
                    self.orbitfit = dill.load(f)
            else:
                self.orbitfit = OrbitFit(orbitfit_params, self.preobs, self.universe)

                # Make directory and cache it
                self.orbitfit_path.parents[0].mkdir(parents=True, exist_ok=True)
                with open(self.orbitfit_path, "wb") as f:
                    dill.dump(self.orbitfit, f)
        else:
            self.orbitfit = OrbitFit(orbitfit_params, self.preobs, self.universe)

    def calc_pdet(self, pdet_params):
        self.pdet = PDet(pdet_params, self.orbitfit, self.preobs, self.universe)

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
        self.builder.simulate_rv_observations()
        self.builder.orbit_fitting()

    def build_full_info(self) -> None:
        self.builder.create_universe()
        self.builder.simulate_rv_observations()
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
    builder.simulate_rv_observations()
    builder.precursor_data.list_parts()
