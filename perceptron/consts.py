import numpy as np

INPUT = np.asarray([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
AND_OUTPUT = np.asarray([[0], [0], [0], [1]])
OR_OUTPUT = np.asarray([[0], [1], [1], [1]])

# INPUT_BIPOLAR = np.asarray([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
INPUT_BIPOLAR = np.asarray([[0.95, -1, -1], [0.95, -1, 0.95], [0.95, 0.95, -1], [0.95, 0.95, 0.95]])
AND_OUTPUT_BIPOLAR = np.asarray([[-1], [-1], [-1], [1]])
OR_OUTPUT_BIPOLAR = np.asarray([[-1], [1], [1], [1]])
