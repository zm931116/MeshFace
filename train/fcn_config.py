from easydict import EasyDict as edict

config = edict()

config.batch_size = 32
config.lr = 1e-3
config.epsilon = 1e-3
config.img_height = 220
config.img_width = 178
config.img_channel = 3
config.capacity_factor = 200
config.reading_threads_num = 2
