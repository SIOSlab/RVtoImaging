from __future__ import annotations

import hashlib
import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import astropy
import astropy.units as u
import dill

from RVtools.library import Library
from RVtools.logger import logger
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

    def __init__(self, cache_dir=".cache", workers=1):
        """
        A fresh builder instance should contain a blank rvdata object, which is
        used in further assembly.
        """
        self.cache_dir = cache_dir
        self.workers = workers
        self.reset()

    def reset(self):
        self.library = Library(self.cache_dir)
        self.rvdata = RVData(self.cache_dir, self.workers, self.library)
        # self.rvdata.workers = self.workers
        # self.cache_universe = False
        # self.cache_preobs = False
        # self.cache_orbitfit = False
        # self.cache_pdet = False

    # @property
    # def run_title(self):
    #     return self._run_title

    # @run_title.setter
    # def run_title(self, run_title):
    #     self._run_title = run_title
    #     self.rvdata.run_title = run_title
    #     # self.rvdata.cache_setup()

    # @property
    # def workers(self):
    #     return self._workers

    # @workers.setter
    # def workers(self, workers):
    #     self._workers = workers
    #     self.rvdata.workers = workers

    # @property
    # def cache_universe(self):
    #     return self._cache_universe

    # @cache_universe.setter
    # def cache_universe(self, val):
    #     self._cache_universe = val
    #     self.rvdata.cache_universe = val

    # @property
    # def cache_preobs(self):
    #     return self._cache_preobs

    # @cache_preobs.setter
    # def cache_preobs(self, val):
    #     self._cache_preobs = val
    #     self.rvdata.cache_preobs = val

    # @property
    # def cache_orbitfit(self):
    #     return self._cache_orbitfit

    # @cache_orbitfit.setter
    # def cache_orbitfit(self, val):
    #     self._cache_orbitfit = val
    #     self.rvdata.cache_orbitfit = val

    # @property
    # def cache_pdet(self):
    #     return self._cache_pdet

    # @cache_pdet.setter
    # def cache_pdet(self, val):
    #     self._cache_pdet = val
    #     self.rvdata.cache_pdet = val

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
        logger.info("Creating universe")
        self.rvdata.create_universe(self.universe_params)

    def simulate_rv_observations(self):
        logger.info("Creating precursor observations")
        self.rvdata.precursor_observations(self.preobs_params)

    def orbit_fitting(self):
        logger.info("Running orbit fitting")
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

    def __init__(self, cache_dir, workers, library):
        self.cache_dir = cache_dir
        self.workers = workers
        self.library = library

    def create_universe(self, universe_params):
        universe_type = universe_params["universe_type"]
        universe_spec = universe_params.copy()
        # if self.cache_universe:
        # Create hash from universe parameters
        new_params = {}
        if "script" in universe_params.keys():
            # Add exosims parameters into the universe params
            with open(Path(universe_params["script"])) as f:
                exosims_script = json.loads(f.read())
            for key in exosims_script.keys():
                value = exosims_script[key]
                if type(value) is dict:
                    for subkey in value.keys():
                        if (value[subkey] == "") or (value[subkey] == " "):
                            # Don't add unset parameters
                            pass
                        else:
                            new_params[subkey] = value[subkey]
                else:
                    new_params[key] = exosims_script[key]
            # Add EXOSIMS version number
            new_params["release"] = importlib.metadata.version("EXOSIMS")
            universe_spec.update(new_params)

        # Remove exosims script name if given, will probably be better as a
        # list of keys
        if "script" in universe_spec:
            universe_spec.pop("script")
        # Sorting so that the order doesn't matter
        universe_spec = {key: universe_spec[key] for key in sorted(universe_spec)}

        # Create hash from the parameters
        self.universe_hash = hashlib.sha1(
            str(universe_spec).encode("UTF-8")
        ).hexdigest()[:8]

        # Create save directory for generated universe
        self.universe_dir = Path(self.cache_dir, f"universe_{self.universe_hash}")
        self.universe_dir.mkdir(parents=True, exist_ok=True)

        # Create path for the universe object
        self.universe_path = Path(self.universe_dir, "universe.p")

        # Load or create universe
        if self.universe_path.exists():
            # Load
            with open(self.universe_path, "rb") as f:
                self.universe = dill.load(f)
            logger.info(f"Loaded universe from {self.universe_path}")
        else:
            # Create
            universelib = importlib.import_module(f"RVtools.cosmoses.{universe_type}")
            self.universe = universelib.create_universe(universe_params)

            # Cache
            with open(self.universe_path, "wb") as f:
                dill.dump(self.universe, f)
            logger.info(f"Created universe, saved to {self.universe_path}")

        # Add the star names to the specification dict
        universe_spec["stars"] = self.universe.names

        # Update library
        self.library.update(self.universe_dir, universe_spec)

    def precursor_observations(self, preobs_params):
        assert hasattr(
            self, "universe_path"
        ), "Precursor observations must have a set universe for caching"

        # Get all the specifications
        base_params = preobs_params["base_params"]
        insts = preobs_params["instruments"]
        preobs_spec = {}
        for inst in insts:
            # Creating single dictionaries for each instrument that only have strings
            # so that they play nicely with json formatting
            str_params = {}
            inst_params = base_params.copy()
            inst_params.update(inst)
            for key in inst_params.keys():
                value = inst_params[key]
                if type(value) == u.Quantity:
                    str_params[key] = value.decompose().value
                elif type(value) == astropy.time.Time:
                    str_params[key] = value.decimalyear
                else:
                    str_params[key] = value
            # Sort
            str_params = {key: str_params[key] for key in sorted(str_params)}
            preobs_spec[inst["name"]] = str_params

        syst_ids = preobs_params["systems_to_observe"]
        system_names = []
        for system_id in syst_ids:
            # Get the system's name from the universe
            system_names.append(self.universe.names[system_id])

        preobs_spec["stars"] = system_names
        preobs_spec = {key: preobs_spec[key] for key in sorted(preobs_spec)}
        # Create hash from the parameters
        self.preobs_hash = hashlib.sha1(str(preobs_spec).encode("UTF-8")).hexdigest()[
            :8
        ]

        # Caching information
        self.preobs_dir = Path(self.universe_dir, f"preobs_{self.preobs_hash}")
        self.preobs_dir.mkdir(exist_ok=True)
        self.preobs_path = Path(self.preobs_dir, "preobs.p")

        # Load or create precursor observations
        if self.preobs_path.exists():
            # Load
            with open(self.preobs_path, "rb") as f:
                self.preobs = dill.load(f)
            logger.info(f"Loaded precursor observations from {self.preobs_path}")
        else:
            # Create
            self.preobs = PreObs(preobs_params, self.universe)

            # Cache
            with open(self.preobs_path, "wb") as f:
                dill.dump(self.preobs, f)

            logger.info(f"Created precursor observations, saved to {self.preobs_path}")

        # Update library
        self.library.update(self.preobs_dir, preobs_spec)

    def orbit_fitting(self, orbitfit_params):
        # Get all the parameters
        # orbitfit_spec = {}
        # syst_ids = orbitfit_params["systems_to_fit"]
        # system_names = []
        # for system_id in syst_ids:
        #     # Get the system's name from the universe
        #     system_names.append(self.universe.names[system_id])
        orbitfit_params["cache_dir"] = self.preobs_dir
        self.orbitfit = OrbitFit(
            orbitfit_params, self.library, self.universe, self.preobs, self.workers
        )
        # Update library
        # self.library.update(self.orbitfit_dir, orbitfit_spec)
        # orbitfit_spec["stars"] = system_names
        # for key in orbitfit_params.keys():
        #     value = orbitfit_params[key]
        #     if key == "systems_to_fit":
        #         # This is already included with the system_names key
        #         pass
        #     else:
        #         orbitfit_spec[key] = value

        # Sort
        # orbitfit_spec = {key: orbitfit_spec[key] for key in sorted(orbitfit_spec)}

        # # Create hash from the parameters
        # self.orbitfit_hash = hashlib.sha1(
        #     str(orbitfit_spec).encode("UTF-8")
        # ).hexdigest()[:8]

        # self.orbitfit_dir = Path(self.preobs_dir, f"orbitfit_{self.orbitfit_hash}")
        # self.orbitfit_dir.mkdir(exist_ok=True)
        # self.orbitfit_path = Path(self.orbitfit_dir, "orbitfit.p")
        # make directory if necessary
        # orbitfit_params["cache_dir"] = str(self.orbitfit_dir)
        # if self.orbitfit_path.exists():
        #     # Load it
        #     with open(self.orbitfit_path, "rb") as f:
        #         self.orbitfit = dill.load(f)
        #     logger.info(f"Loaded orbit fit object from {self.orbitfit_path}")
        # else:
        #     self.orbitfit = OrbitFit(
        #         orbitfit_params, self.preobs, self.universe, self.workers
        #     )

        #     # Make directory and cache it
        #     self.orbitfit_path.parents[0].mkdir(parents=True, exist_ok=True)
        #     with open(self.orbitfit_path, "wb") as f:
        #         dill.dump(self.orbitfit, f)
        #     logger.info(
        #         f"Created and then saved orbitfit object to {self.orbitfit_path}"
        #     )

    def calc_pdet(self, pdet_params):
        self.pdet = PDet(pdet_params, self.orbitfit, self.universe, self.library)

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
