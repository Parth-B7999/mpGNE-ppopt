import numpy as np
from mpgne.plant_gen import make_random_plants, make_ic
from tests.full_case_study import run_one_plant

if __name__ == '__main__':
    plants = make_random_plants(2, 1, seed=44)
    res = run_one_plant(plants[0], make_ic(plants[0], scale=0.4), 2, 0)
    print("ADMM:    ", res["methods"]["ADMM"]["avg_time"]*1000, "ms")
    print("FACET-H: ", res["methods"]["FACET-H"]["avg_time"]*1000, "ms")
