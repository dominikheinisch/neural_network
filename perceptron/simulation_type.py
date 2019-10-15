from enum import Enum

from perceptron import perceptron_binary, perceptron_bipolar
from adaline import adaline
from consts import *

class SimulationType:
    @staticmethod
    def generate(func, input, output, name):
        return {'func': func, 'input': input, 'output': output, 'name': name}

    ADALINE_AND = generate.__func__(adaline, INPUT_BIPOLAR, AND_OUTPUT_BIPOLAR, 'ADALINE_AND')
    ADALINE_OR = generate.__func__(adaline, INPUT_BIPOLAR, OR_OUTPUT_BIPOLAR, 'ADALINE_OR')

    PERCEPTRON_BIPOLAR_AND = generate.__func__(perceptron_bipolar, INPUT_BIPOLAR, AND_OUTPUT_BIPOLAR,
                                               'PERCEPTRON_BIPOLAR_AND')
    PERCEPTRON_BIPOLAR_OR = generate.__func__(perceptron_bipolar, INPUT_BIPOLAR, OR_OUTPUT_BIPOLAR,
                                              'PERCEPTRON_BIPOLAR_OR')

    PERCEPTRON_BINARY_AND = generate.__func__(perceptron_binary, INPUT, AND_OUTPUT, 'PERCEPTRON_BINARY_AND')
    PERCEPTRON_BINARY_OR = generate.__func__(perceptron_binary, INPUT, OR_OUTPUT, 'PERCEPTRON_BINARY_OR')
