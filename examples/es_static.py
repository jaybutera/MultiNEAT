import numpy as np
import sys
import random
import copy
from fb_api import EvoComm

class NeuralNet(object):
    def __init__ (self, in_size, hid_size, out_size):
        self.in_size = in_size+1
        self.hid_size = hid_size
        self.out_size = out_size

        # Init weights
        # ------------

        # Inputs to outputs
        self.w_inp_out = np.random.rand(in_size, out_size) # plus 1 for bias

    def sigmoid (self, x):
        return np.divide(1., np.add(1, np.exp(-x)))

    def activate (self, inputs):
        inputs.append(1.) # inputs and bias
        in_b = np.array( inputs ) # inputs and bias

        a = np.dot(in_b, self.w_inp_out)
        return self.sigmoid( a )

def tourn_select (fit_scores, pop):
    k_groups = [[] for i in range(k)]

    # Split creatures into categories
    for i, creature in enumerate(pop.iteritems()):
        k_groups[ i % k ].append( creature )

    # New empty pop
    newpop = {}

    for group in k_groups:
        # Randomly choose parents
        p1 = random.choice(group)[1]
        p2 = random.choice(group)[1]

        # Make sure the parents aren't the same
        while p2 == p1:
            p2 = random.choice(group)[1]

        # Make children
        c1 = crossover(p1, p2)
        mutate(c1)
        c2 = crossover(p1, p2)
        mutate(c2)

        # Get sorted fitnesses of group (least fit to most)
        group_ids = [i[0] for i in group]
        group_fits = [(s.Id(), s.Fitness()) for s in fit_scores if s.Id() in group_ids]
        group_fits.sort(key=lambda tup: tup[0])

        # Replace 2 least fit with children
        # Add all nets in group to pop
        id1 = group_fits[0][0]
        id2 = group_fits[0][1]
        for x in group:
            if (x[0] == id1):
                x = (x[0], c1) # Replace net with c1 and same id
            elif (x[0] == id2):
                x = (x[0], c2) # Replace net with c1 and same id

            pop[ x[0] ] = x[1]

    return pop


def crossover(p1, p2):
    assert (p1.in_size == p2.in_size), "Parent net input sizes don't match"
    assert (p1.out_size == p2.out_size), "Parent net output sizes don't match"

    # Random submatrix for crossover
    top_left_x_point = (random.randint(0, p1.in_size-1),
                        random.randint(0, p1.out_size-1))
    bot_right_x_point = (random.randint(top_left_x_point[0], p1.in_size-1),
                         random.randint(top_left_x_point[1], p1.out_size-1))

    # Make child
    #c = NeuralNet(p1.in_size, p1.hid_size, p1.out_size)
    c = copy.deepcopy(p1)

    # Perform crossover
    # Replace random submatrix of p1 to p2
    for i in range(bot_right_x_point[0], top_left_x_point[0]):
        for j in range(bot_right_x_point[1], top_left_x_point[1]):
            c.w_inp_out[i][j] = p2.w_inp_out[i][j]

    return c

def mutate (net):
    mutate_prob = .2

    rand_choice = np.random.rand(net.in_size, net.out_size)

    # Random chance to mutate
    for i in range( net.in_size-1 ):
        for j in range( net.out_size-1 ):
            if (random.random() < mutate_prob):
                net.w_inp_out[i][j] = random.random()


k = 4
pop_size = 32
input_size = 6
output_size = 2
# Generate population
pop = {i:NeuralNet(input_size,0,output_size) for i in range(pop_size)}

port = sys.argv[1]

# Communictor to simulator
comm = EvoComm()

# Connect to simulator
comm.connect(port)
print 'connected...'

while True:
    iteration = 0

    # send ids to simulator
    ids = pop.keys()
    comm.send_ids(ids)

    # Step process
    while True:
        # Get observations from simulator
        observations = comm.get_obs()

        # Next epoch if observations empty
        if not observations:
            fit_scores = comm.next_epoch()

            # Create next generation
            pop = tourn_select(fit_scores, pop)
            # Reset
            break

        # Generate actions (net_id, action)
        actions = {}
        for net_id, inp_vec in observations.iteritems():
            action = pop[net_id].activate( inp_vec )
            actions[net_id] = action

        # Send actions to simulator
        comm.send_actions(actions)

        iteration += 1

    print 'Finished generation'
