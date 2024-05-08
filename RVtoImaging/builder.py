from __future__ import annotations

import hashlib
import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import dill

import RVtoImaging.utils as utils
from EXOSIMS.util.get_module import get_module_from_specs
from RVtoImaging.imagingprobability import ImagingProbability
from RVtoImaging.imagingschedule import ImagingSchedule
from RVtoImaging.logger import logger
from RVtoImaging.rvdataset import RVDataset
from RVtoImaging.rvfits import RVFits


class Builder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the RVtoImaging objects.
    """

    @property
    @abstractmethod
    def get_rv2img(self) -> None:
        pass

    @abstractmethod
    def build_universe(self) -> None:
        pass

    @abstractmethod
    def build_rv_dataset(self) -> None:
        pass

    @abstractmethod
    def build_rv_fits(self) -> None:
        pass

    @abstractmethod
    def build_img_pdet(self) -> None:
        pass

    @abstractmethod
    def build_img_schedule(self) -> None:
        pass


class BaseBuilder(Builder):
    """
    This class will create an inital implementation of the creation
    process.
    """

    def __init__(self, cache_dir=".cache", workers=1):
        """
        A fresh builder instance should contain a blank rv2img object, which is
        used in further assembly.
        """
        self.cache_dir = cache_dir
        self.workers = workers
        self.reset()

    def reset(self):
        self.rv2img = RVtoImaging(self.cache_dir, self.workers)

    @property
    def get_rv2img(self) -> RVtoImaging:
        """
        Concrete Builders are supposed to provide their own methods for
        retrieving results. That's because various types of builders may create
        entirely different rv2img that don't follow the same interface.
        Therefore, such methods cannot be declared in the base Builder interface
        (at least in a statically typed programming language).

        Usually, after returning the end result to the client, a builder
        instance is expected to be ready to start producing another rv2img.
        That's why it's a usual practice to call the reset method at the end of
        the `getrv2img` method body. However, this behavior is not mandatory,
        and you can make your builders wait for an explicit reset call from the
        client code before disposing of the previous result.
        """
        rv2img = self.rv2img
        self.reset()
        return rv2img

    def build_universe(self):
        logger.info("Building universe")
        self.rv2img.create_universe(self.universe_params)

    def build_rv_dataset(self):
        logger.info("Building RV dataset")
        self.rv2img.create_rv_dataset(self.rv_dataset_params)

    def build_rv_fits(self):
        logger.info("Building orbit fits")
        self.rv2img.create_rv_fits(self.rv_fits_params)

    # MOVE THIS TO THE SCHEDULER!
    def build_img_pdet(self):
        logger.info("Building probability of detection")
        self.rv2img.create_pdet(self.pdet_params)

    def build_img_schedule(self):
        self.rv2img.create_img_schedule(self.img_schedule_params)

    def build_to_fits(self) -> None:
        self.build_universe()
        self.build_rv_dataset()
        self.build_rv_fits()

    def build_to_img_schedule(self) -> None:
        self.build_universe()
        self.build_rv_dataset()
        self.build_rv_fits()
        self.build_img_pdet()
        self.build_img_schedule()

    def run_sim(self, seed):
        # Used for time estimation
        # self.rv_fits_params["initial_start_time"] = time.time()
        # self.rv_fits_params["total_searches"] = (
        #     len(self.seeds) * self.rv_dataset_params["approx_systems_to_observe"]
        # )
        # self.rv_fits_params["completed_searches"] = 0
        # self.rv_fits_params["loaded_searches"] = 0
        # self.rv_fits_params["total_universes"] = len(self.seeds)

        # for seed_ind, seed in enumerate(self.seeds):
        self.universe_params["forced_seed"] = int(seed)
        # self.rv_fits_params["universe_number"] = seed_ind + 1

        self.build_universe()
        if hasattr(self, "rv_fits_params"):
            self.build_rv_dataset()
        if hasattr(self, "rv_fits_params"):
            self.build_rv_fits()
        if hasattr(self, "pdet_params"):
            self.pdet_params["forced_seed"] = int(seed)
            self.build_img_pdet()
        if hasattr(self, "img_schedule_params"):
            self.build_img_schedule()
        # self.rv_fits_params["completed_searches"]+=self.rv2img.orbitfit.fits_completed
        # self.rv_fits_params["loaded_searches"] += self.rv2img.orbitfit.fits_loaded
        logger.info("Simulation complete\n")


class RVtoImaging:
    """
    This class holds all the different parts of the simulation:
        Universe
        RVDataset
        RVFitting
        ImagingScheduler

    It makes sense to use the Builder pattern only when your rv2img are quite
    complex and require extensive configuration.

    Unlike in other creational patterns, different concrete builders can produce
    unrelated rv2img. In other words, results of various builders may not
    always follow the same interface.
    """

    def __init__(self, cache_dir, workers):
        self.cache_dir = cache_dir
        self.workers = workers

    def create_universe(self, universe_params):
        universe_params = universe_params.copy()
        universe_type = universe_params["universe_type"]
        universe_spec = universe_params.copy()
        # Create hash from universe parameters
        new_params = {}
        # delete_tmp = False
        # original_script = universe_params["script"]
        if universe_type == "exosims":
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
                "guarantee_earths",
                "earths_only",
                "int_WA",
                "int_dMag",
                "scaleWAdMag",
                "popStars",
                "arange",
                "erange",
                "Irange",
                "Orange",
                "wrange",
                "prange",
                "Rprange",
                "Mprange",
                "scaleOrbits",
                "constrainOrbits",
                "eta",
                "catalogpath",
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
                        f"{universe_params['script'].split('.')[0].replace('/', '_')}"
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
        elif universe_type == "exovista":
            if "forced_seed" in universe_params.keys():
                universe_spec["universe_number"] = universe_params["forced_seed"]
                universe_params["universe_number"] = universe_params["forced_seed"]

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
        if universe_type == "exovista":
            data_path = Path(self.universe_dir, "ExoVista")
            if not data_path.exists():
                data_path.mkdir(parents=True, exist_ok=True)
            universe_params["data_path"] = str(data_path)

        # Load or create universe
        if self.universe_path.exists():
            # Load
            with open(self.universe_path, "rb") as f:
                self.universe = dill.load(f)
            logger.info(f"Loaded universe from {self.universe_path}")
        else:
            # Create
            exoverse_module = importlib.import_module(
                f"exoverses.{universe_type}.universe"
            )
            self.universe = exoverse_module.create_universe(universe_params)

            # Cache
            with open(self.universe_path, "wb") as f:
                dill.dump(self.universe, f)
            logger.info(f"Created universe, saved to {self.universe_path}")

        # Add the star names to the specification dict
        universe_spec["stars"] = self.universe.names
        # Creating the spec should be a universe class method instead of the
        # current process
        if "script" in universe_params.keys():
            if "forced_seed" in universe_params.keys():
                universe_params["script"] = original_script
            if delete_tmp:
                tmp_file.unlink()

        # Update library
        utils.update(self.universe_dir, universe_spec)

    def create_rv_dataset(self, rv_dataset_params):
        assert hasattr(
            self, "universe_path"
        ), "RV dataset must have a set universe for caching"

        # base_params = rv_dataset_params["base_params"]
        rv_dataset_params["universe_dir"] = self.universe_dir
        self.rv_dataset_params = rv_dataset_params
        self.rvdataset = RVDataset(self.rv_dataset_params, self.universe, self.workers)
        # for rv_observing_runs in rv_dataset_params["rv_observing_runs"]:
        #     # Create descriptive name for rv_dataset based on the input parameters
        #     rv_observing_run_params = rv_dataset_params.copy()
        #     rv_observing_run_params.pop("rv_observing_runs")
        #     # rv_observing_run_params.update(
        #     #     {"instruments": rv_observing_runs["instruments"]}
        #     # )
        #     # rv_observing_run_params.update(
        #     #     {"fit_order": rv_observing_runs["fit_order"]}
        #     # )
        #     rv_observing_run_name = ""

        #     # Sorting the instruments by their precisions to maintain consistency
        #     obs_run_precisions = []
        #     obs_run_baselines = []
        #     for obs_run in rv_observing_runs["instruments"]:
        #         # obs_run_params = base_params.copy()
        #         obs_run_params.update(obs_run)
        #         # TODO - ADD ALL THIS CACHING TO THE MODULE ITSELF
        #         # inst_precisions.append(inst_params["precision"].decompose().value)
        #         breakpoint()
        #         obs_run_precisions.append()
        #         obs_run_baselines.append(
        #             round(
        #                 (
        #                     obs_run_params["end_time"].decimalyear
        #                     - obs_run_params["start_time"].decimalyear
        #                 )
        #             )
        #         )
        #     sorted_inds = np.flip(np.argsort(obs_run_precisions))

        #     # Create name to identify the rv_observing_run in files
        #     for ind in sorted_inds:
        #         rv_observing_run_name += (
        #             f"{obs_run_precisions[ind]:.2f}m_{obs_run_baselines[ind]}-"
        #         )
        #     rv_observing_run_name += (
        #         f"{rv_observing_run_params['approx_systems_to_observe']}_systems-filter"
        #     )
        #     for filter in sorted(rv_observing_run_params["filters"]):
        #         rv_observing_run_name += f"_{filter}"
        #     rv_observing_run_params["name"] = rv_observing_run_name

        #     # Create filename
        #     rv_observing_run_file = Path(
        #         self.universe_dir, "rv_observing_runs", f"{rv_observing_run_name}.p"
        #     )
        #     rv_observing_run_params["universe_dir"] = self.universe_dir

        #     # Caching information
        #     if not rv_observing_run_file.exists():
        #         # Create RV dataset
        #         rv_observing_run = RVDataset(rv_observing_run_params, self.universe)

        #         # Cache
        #         Path(self.universe_dir, "rv_observing_runs").mkdir(exist_ok=True)
        #         with open(rv_observing_run_file, "wb") as f:
        #             dill.dump(rv_observing_run, f)
        #         logger.info(
        #             f"Ran {rv_observing_run_name} rv observing run,"
        #             f" saved to {rv_observing_run_file}"
        #         )
        #     else:
        #         # Load
        #         with open(rv_observing_run_file, "rb") as f:
        #             rv_observing_run = dill.load(f)
        #         logger.info(
        #             f"Loaded {rv_observing_run_name} rv observing run"
        #             f" from {rv_observing_run_file}"
        #         )

        #     self.rv_observing_runs.append(rv_observing_run)

    def create_rv_fits(self, rv_fits_params):
        rv_fits_params["universe_dir"] = self.universe_dir
        self.rv_fits_params = rv_fits_params
        self.orbitfit = RVFits(
            self.rv_fits_params, self.universe, self.rvdataset, self.workers
        )

    def create_pdet(self, pdet_params):
        with open(Path(pdet_params["script"])) as f:
            exosims_script = json.loads(f.read())
        EXOSIMS_overwrites = pdet_params.get("EXOSIMS_overwrites", {})
        if EXOSIMS_overwrites:
            exosims_script = utils.overwrite_script(exosims_script, EXOSIMS_overwrites)
            # exosims_script.update(EXOSIMS_overwrites)
        # target_list = pdet_params.get("target_list", False)
        # if target_list:
        #     exosims_script["catalogpath"] = pdet_params["target_list"]

        if "forced_seed" in pdet_params.keys():
            # Make a copy of the exosims json and save it to the cache, then
            # delete after
            original_script = pdet_params["script"]
            exosims_script["seed"] = pdet_params["forced_seed"]
            tmp_file = Path(
                self.cache_dir,
                (
                    f"{pdet_params['script'].split('.')[0].replace('/', '_')}"
                    f"_seed_{exosims_script['seed']}.json"
                ),
            )
            with open(tmp_file, "w") as f:
                json.dump(exosims_script, f)
            delete_tmp = True
            pdet_params["script"] = str(tmp_file)
        else:
            delete_tmp = False
        self.pdet = ImagingProbability(
            pdet_params, self.orbitfit, self.universe, self.workers
        )
        if delete_tmp:
            tmp_file.unlink()
        if "forced_seed" in pdet_params.keys():
            pdet_params["script"] = original_script

    def create_img_schedule(self, img_schedule_params):
        self.img_schedule_params = img_schedule_params
        self.scheduler = ImagingSchedule(
            img_schedule_params,
            self.rv_dataset_params,
            self.pdet,
            self.universe_dir,
            self.workers,
        )

    def update_SS(self, pdet_params):
        with open(Path(pdet_params["script"])) as f:
            exosims_script = json.loads(f.read())
        EXOSIMS_overwrites = pdet_params.get("EXOSIMS_overwrites", {})
        if EXOSIMS_overwrites:
            specs = utils.overwrite_script(exosims_script, EXOSIMS_overwrites)

        specs["exoverses_universe"] = self.universe
        specs["seed"] = pdet_params["forced_seed"]

        # tmp_SS = get_module_from_specs(specs, "SurveySimulation")(**specs)
        # breakpoint()
        self.pdet.SS = get_module_from_specs(specs, "SurveySimulation")(**specs)

    def list_parts(self) -> None:
        print(f"rv2img parts: {', '.join(self.parts)}", end="")


if __name__ == "__main__":
    """
    The client code creates a builder object, passes it to the director and then
    initiates the construction process. The end result is retrieved from the
    builder object.
    """

    # director = Director()
    # builder = BaseBuilder()
    # director.builder = builder

    # print("Standard basic rv2img: ")
    # director.build_minimal_viable_rv2img()
    # builder.precursor_data.list_parts()

    # print("\n")

    # print("Standard full featured rv2img: ")
    # director.build_full_info()
    # builder.precursor_data.list_parts()

    # print("\n")

    # # Remember, the Builder pattern can be used without a Director class.
    # print("Custom rv2img: ")
    # builder.create_universe()
    # builder.simulate_rv_observations()
    # builder.precursor_data.list_parts()
