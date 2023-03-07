from __future__ import annotations

import hashlib
import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import dill
import numpy as np

import RVtools.utils as utils
from RVtools.logger import logger
from RVtools.orbitfit import OrbitFit
from RVtools.pdet import PDet
from RVtools.preobs import PreObs
from RVtools.scheduler import Scheduler


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
        self.rvdata = RVData(self.cache_dir, self.workers)

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

    def create_schedule(self):
        self.rvdata.run_scheduler(self.schedule_params)


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

    def __init__(self, cache_dir, workers):
        self.cache_dir = cache_dir
        self.workers = workers

    def create_universe(self, universe_params):
        universe_type = universe_params["universe_type"]
        universe_spec = universe_params.copy()
        # Create hash from universe parameters
        new_params = {}
        # delete_tmp = False
        # original_script = universe_params["script"]
        if "script" in universe_params.keys():
            # Add exosims parameters into the universe params
            necessary_EXOSIMS_keys = [
                "fixedPlanPerStar",
                "Min",
                "commonSystemInclinations",
                "commonSystemInclinationParams",
                "seed",
                "missionStart",
                "ntargs",
                "fillPhotometry",
                "filterBinaries",
                "filter_for_char",
                "earths_only",
                "int_WA",
                "int_dMag",
                "scaleWAdMag",
                "popStars",
            ]
            necessary_EXOSIMS_modules = [
                "StarCatalog",
                "PlanetPhysicalModel",
                "PlanetPopulation",
                "SimulatedUniverse",
                "TargetList",
            ]
            with open(Path(universe_params["script"])) as f:
                exosims_script = json.loads(f.read())
            if "forced_seed" in universe_params.keys():
                # Make a copy of the exosims json and save it to the cache, then
                # delete after
                original_script = universe_params["script"]
                exosims_script["seed"] = universe_params["forced_seed"]
                tmp_file = Path(
                    self.cache_dir,
                    (
                        f"{universe_params['script'].split('.')[0]}"
                        f"_seed_{exosims_script['seed']}.json"
                    ),
                )
                with open(tmp_file, "w") as f:
                    json.dump(exosims_script, f)
                delete_tmp = True
                universe_params["script"] = str(tmp_file)
            for key in exosims_script.keys():
                value = exosims_script[key]
                if key in necessary_EXOSIMS_keys or key == "modules":
                    if type(value) is dict:
                        for module in value.keys():
                            if module in necessary_EXOSIMS_modules:
                                if (value[module] == "") or (value[module] == " "):
                                    # Don't add unset parameters
                                    pass
                                else:
                                    new_params[module] = value[module]
                    else:
                        new_params[key] = exosims_script[key]
            # Add EXOSIMS version number
            new_params["release"] = importlib.metadata.version("EXOSIMS")
            universe_spec.update(new_params)

        # Remove exosims script name if given, will probably be better as a
        # list of keys
        delete_keys = ["script", "forced_seed"]
        for key in delete_keys:
            if key in universe_spec.keys():
                universe_spec.pop(key)
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
        if "script" in universe_params.keys():
            if "forced_seed" in universe_params.keys():
                universe_params["script"] = original_script
            if delete_tmp:
                tmp_file.unlink()

        # Update library
        utils.update(self.universe_dir, universe_spec)

    def precursor_observations(self, preobs_params):
        assert hasattr(
            self, "universe_path"
        ), "Precursor observations must have a set universe for caching"

        base_params = preobs_params["base_params"]
        self.surveys = []
        for survey in preobs_params["surveys"]:
            # Create descriptive name for survey based on the input parameters
            survey_params = preobs_params.copy()
            survey_params.pop("surveys")
            survey_params.update({"instruments": survey["instruments"]})
            survey_params.update({"fit_order": survey["fit_order"]})
            survey_name = ""

            # Sorting the instruments by their precisions to maintain consistency
            inst_precisions = []
            inst_baselines = []
            for inst in survey["instruments"]:
                inst_params = base_params.copy()
                inst_params.update(inst)
                inst_precisions.append(inst_params["precision"].decompose().value)
                inst_baselines.append(
                    round(
                        (
                            inst_params["end_time"].decimalyear
                            - inst_params["start_time"].decimalyear
                        )
                    )
                )
            sorted_inds = np.flip(np.argsort(inst_precisions))

            # Create name to identify the survey in files
            for ind in sorted_inds:
                survey_name += f"{inst_precisions[ind]:.2f}m_{inst_baselines[ind]}-"
            survey_name += f"{survey_params['n_systems_to_observe']}_systems-filter"
            for filter in sorted(survey_params["filters"]):
                survey_name += f"_{filter}"
            survey_params["name"] = survey_name

            # Create filename
            survey_file = Path(self.universe_dir, "surveys", f"{survey_name}.p")
            survey_params["universe_dir"] = self.universe_dir

            # Caching information
            if not survey_file.exists():
                # Create precursor observations
                survey = PreObs(survey_params, self.universe)

                # Cache
                Path(self.universe_dir, "surveys").mkdir(exist_ok=True)
                with open(survey_file, "wb") as f:
                    dill.dump(survey, f)
                logger.info(f"Ran {survey_name} survey, saved to {survey_file}")
            else:
                # Load
                with open(survey_file, "rb") as f:
                    survey = dill.load(f)
                logger.info(f"Loaded {survey_name} survey from {survey_file}")

            self.surveys.append(survey)

    def orbit_fitting(self, orbitfit_params):
        orbitfit_params["universe_dir"] = self.universe_dir
        self.orbitfit = OrbitFit(
            orbitfit_params, self.universe, self.surveys, self.workers
        )

    def calc_pdet(self, pdet_params):
        with open(Path(pdet_params["script"])) as f:
            exosims_script = json.loads(f.read())
        if "forced_seed" in pdet_params.keys():
            # Make a copy of the exosims json and save it to the cache, then
            # delete after
            original_script = pdet_params["script"]
            exosims_script["seed"] = pdet_params["forced_seed"]
            tmp_file = Path(
                self.cache_dir,
                (
                    f"{pdet_params['script'].split('.')[0]}"
                    f"_seed_{exosims_script['seed']}.json"
                ),
            )
            with open(tmp_file, "w") as f:
                json.dump(exosims_script, f)
            delete_tmp = True
            pdet_params["script"] = str(tmp_file)
        else:
            delete_tmp = False
        self.pdet = PDet(pdet_params, self.orbitfit, self.universe, self.workers)
        if delete_tmp:
            tmp_file.unlink()
        if "forced_seed" in pdet_params.keys():
            pdet_params["script"] = original_script

    def run_scheduler(self, schedule_params):
        self.scheduler = Scheduler(schedule_params, self.pdet, self.universe_dir)

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

    def run_seeds(self, seeds):
        for seed in seeds:
            self.builder.universe_params["forced_seed"] = int(seed)
            if hasattr(self.builder, "pdet_params"):
                self.builder.pdet_params["forced_seed"] = int(seed)
                self.build_full_info()
            else:
                self.build_orbit_fits()

    def build_full_info(self) -> None:
        self.builder.create_universe()
        self.builder.simulate_rv_observations()
        self.builder.orbit_fitting()
        self.builder.probability_of_detection()
        self.builder.create_schedule()


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
