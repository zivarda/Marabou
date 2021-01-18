import numpy as np

ON = 1
OFF = 0

class MultiRange:

    def __init__(self, lower_bounds, upper_bounds):
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.dim = self.lower_bounds.shape[0]
        self.unsat_neurons_on = []
        self.unsat_neurons_off = []


    def contains(self, other):
        """
        Check if other range is contained in this range.
        :param other: other range
        :return:
        """
        return np.all(self.lower_bounds <= other.lower_bounds) and \
               np.all(other.upper_bounds <= self.upper_bounds)

    def add_neuron(self, neuron, state):
        if state == ON:
            self.unsat_neurons_on.append(neuron)
        elif state == OFF:
            self.unsat_neurons_off.append(neuron)

    def get_neurons_number(self):
        return len(self.unsat_neurons_on) + len(self.unsat_neurons_off)

    def get_mid(self):
        """
        :return: The mid point of the range
        """
        return (self.lower_bounds + self.upper_bounds)/2

class MultiRangeDB:
    db = []

    def add(self, m_range):
        """
        Key is path in the search tree (string of numbers), Value is MultiRange
        :param path:
        :param m_range:
        :param key: if key is not None this method compare between the paths of the multi_ranges
        :return:
        """
        self.db.append(m_range)

    def contains(self, m_range, neuron, state):
        """
        Check if there is range that contains m_range such that 'neuron' is in
        the given 'state'
        :param m_range:
        :param neuron:
        :param state:
        :return:
        """
        for rng in self.db:
            arr = rng.unsat_neurons_on if state == ON else rng.unsat_neurons_off
            if rng.contains(m_range) and neuron in arr:
                return True
        return False



    # def __is_sub_path(self, path1, path2):
    #     """
    #     check if path2 is subpath of path1
    #     :param path1: (shorter path)
    #     :param path2: (longer path)
    #     :return:
    #     """
    #     new_path1 = np.array(path1.split(','))
    #     new_path2 = np.array(path2.split(','))
    #     length = new_path1.shape[0]
    #     return np.all(new_path1 == new_path2[:length])
