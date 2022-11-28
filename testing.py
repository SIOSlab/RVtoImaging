import astropy.units as u
from astropy.time import Time

from RVtools.builder import BaseBuilder, Director

# from RVtools.universes import exovista

if __name__ == "__main__":
    director = Director()
    builder = BaseBuilder()
    director.builder = builder

    print("Standard basic precursor_data: ")
    # builder.universe_type = "exovista"
    # builder.universe_params = {"data_path": "./data/", "universe_number": 1}
    builder.universe_type = "exosims"
    builder.universe_params = {"script": "test.json", "nsystems": 10}
    director.build_universe()

    # Set up precursor observation information
    # builder.preobs_type = "KeplerSTM"
    builder.preobs_params = {
        "type": "Poisson",
        "rate": 5 / u.d,
        "start_time": Time(0, format="decimalyear"),
        "end_time": Time(1, format="decimalyear"),
    }
    # builder.preobs_params = {
    #     "type": "equal",
    #     "num": 100,
    #     "start_time": Time(0, format="decimalyear"),
    #     "end_time": Time(1, format="decimalyear"),
    # }
    builder.simulate_rv_observations()
    # builder.precursor_data.list_parts()

    # print("\n")

    # print("Standard full featured precursor_data: ")
    # director.build_full_info()
    # builder.precursor_data.list_parts()

    # print("\n")

    # # Remember, the Builder pattern can be used without a Director class.
    # print("Custom precursor_data: ")
    # builder.create_universe()
    # builder.simulate_observations()
    # builder.precursor_data.list_parts()
