import numpy as np
from matplotlib import pyplot as plt

uk_file_name = "./youtube_video_data/GB_video_data_numbers.csv"
us_file_name = "./youtube_video_data/US_video_data_numbers.csv"

t1 = np.loadtxt(uk_file_name, delimiter=",", dtype="int")
t2 = np.loadtxt(us_file_name, delimiter=",", dtype="int")

zero_data = np.zeros((t1.shape[0], 1)).astype(int)
ones_data = np.ones((t2.shape[0], 1)).astype(int)

t1 = np.hstack((t1, zero_data))
t2 = np.hstack((t2, ones_data))

final_data = np.vstack((t1, t2))

print(final_data)


