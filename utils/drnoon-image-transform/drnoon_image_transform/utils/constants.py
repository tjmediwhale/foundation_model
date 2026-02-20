# Color space (RGB order) of normal fundus images empirically found.
# It may need to be changed based on a large training dataset.
FUNDUS_RGB_MEAN = (155, 61, 45)
FUNDUS_RGB_STD = (86, 37, 27)

FUNDUS_RGB_MEAN_NORMALIZED = tuple(v / 255.0 for v in FUNDUS_RGB_MEAN)
FUNDUS_RGB_STD_NORMALIZED = tuple(v / 255.0 for v in FUNDUS_RGB_STD)
