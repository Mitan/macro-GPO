from src.enum.MethodEnum import Methods


class MethodDescriptor:
    def __init__(self, method_type, h, num_samples, beta):

        # exploration parameter beta for mutual information
        self.beta = beta

        # number of stochastic samples generated for every node. N in the notations of the paper
        self.num_samples = num_samples

        # planning horizon value. H in the notations of the paper
        self.h = h

        if method_type == Methods.Exact:
            method_string = 'exact'
        elif method_type == Methods.Anytime:
            method_string = 'anytime'
        else:
            raise Exception("Method not supported")

        # currently supported exact or anytime algorithm
        self.method_type = method_type

        # handle to name the folder for the execute method
        self.method_folder_name = "{}_h{}_beta{}".format(method_string, self.h, self.beta)