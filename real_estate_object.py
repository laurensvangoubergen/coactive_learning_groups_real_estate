import numpy as np


class Real_estate_object:
    def __init__(self, solns):
        self.phi = [solns[0]['type_terms'], solns[0]['area_terms'], solns[0]['room_terms'], solns[0]['layout_terms']]
        self.object = [solns[0]['type_feature_rep'], solns[0]['area_feature_rep'], solns[0]['rooms_feature_rep'], solns[0]['layout_feature_rep']]

    def get_utility(self, weights):
        utility = 0
        for i in range(len(weights)):
            utility += np.sum(weights[i] * self.phi[i])
        return utility