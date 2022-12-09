import json
from pathlib import Path

from RVtools.logger import logger


class Library:
    """
    This object handles the library of generated data
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

    def load_library(self):
        """
        Function that runs on instantiation to load all specification files
        """
        pass

    def update(self, path, spec):
        """
        Function to update library with newly generated parameters
        """
        spec_path = Path(path, "spec.json")
        if spec_path.exists():
            # Load it to see if it has to be overwritten
            with open(spec_path, "r") as f:
                old_spec = json.load(f)

            # If they're the same then nothing needs to be done
            if spec == old_spec:
                needs_update = False
            else:
                needs_update = True
        else:
            needs_update = True

        if needs_update:
            with open(spec_path, "w") as f:
                json.dump(spec, f)
            logger.info(f"Saved new specification to {spec_path}.")
        else:
            logger.info(f"Found up to date specification at {spec_path}.")

        # return needs_update

    def check_orbitfit_dir(self, dir):
        """
        Function looks at a directory of orbit fits

        Args:
            dir (Path):
                The star's directory

        Returns:
            has_fit (bool):
                Whether the star has a previous fit attempted
            prev_max (int):
                What 'max_planets' was set to during the best fit attempted on
                the star
            fitting_done (bool):
                True when a search was conducted that returned less planets than
                the allowed 'max_planets', indicating that all planets that can
                be found were found.

        """
        dir_list = list(Path(dir).glob("*"))
        if len(dir_list) == 0:
            # No attempts at orbit fitting for this system
            has_fit = False
            prev_max = 0
            fitting_done = False

        else:
            # Has a previous attempt to do orbit fitting
            has_fit = True
            prev_run_dirs = [prev_run for prev_run in dir_list if prev_run.is_dir()]
            prev_max = 0
            highest_planets_fitted = 0
            for prev_run in prev_run_dirs:
                with open(Path(prev_run, "spec.json"), "r") as f:
                    run_info = json.load(f)

                # Get the information on that fit
                run_max = run_info["max_planets"]
                planets_fitted = run_info["planets_fitted"]

                # If more planets were searched for than previous runs
                if run_max > prev_max:
                    prev_max = run_max
                    highest_planets_fitted = planets_fitted

            if highest_planets_fitted < prev_max:
                fitting_done = True
            else:
                fitting_done = False

        return has_fit, prev_max, fitting_done

    def query(self, universe_spec, preobs_spec, orbitfit_spec, pdet_spec):
        """
        Function that queries the library for planets meeting parameter sets
        """
        pass
