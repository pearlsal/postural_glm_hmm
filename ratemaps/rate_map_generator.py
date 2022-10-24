class RateMapGenerator(object):
    def __init__(self, comparing=False, cell_index=(10, 11),
                      include_selfmotion=True, selfmotion_lims=(0, 40, -5, 80), selfmotion_bin_size=3,
                      include_derivatives=True, include_spatial_maps=True,
                      include_generic_2d_plots=True, include_velocity_plots=False,
                      compare_task_2d=None, comparing_task_1d=(''),
                      compare_selfmotion=True, compare_spatial_maps=True, compare_velocity=False,
                      pl_subplot=(14, 10), pl_size=(70, 70),
                      cf_subplot=(14, 10), cf_size=(70, 70),
                      save_1d_ratemaps=True):
        self.comapring_animal = comparing
        self.time_offset = 0
        self.n_shuffles = 1000
        self.shuffle_offset = (15, 60)
        self.data = None

    def set_time_offset(self, offset):
        self.time_offset = offset

    def set_data(self, data):
        self.data = data

    def run_shuffling(self, n_shuffles=1000, shuffle_offset=(15, 60)):
        self.n_shuffles = n_shuffles
        self.shuffle_offset = shuffle_offset