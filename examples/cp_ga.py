import gym
import numpy as np
import random
import copy

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

    def threshold(self, x):
        return np.array([ 1 if i > .5 else 0 for i in x])

    def activate (self, inputs):
        inputs = np.concatenate((inputs, np.array([1.])), axis=0)
        in_b = np.array( inputs ) # inputs and bias

        a = np.dot(in_b, self.w_inp_out)
        return self.threshold( a )

def tourn_select (fit_scores, pop):
    global k

    # Error checking
    num_creats = len( pop.keys() )
    if k > num_creats:
        k = num_creats

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
        while p2 == p1 and num_creats > 1:
            p2 = random.choice(group)[1]

        # Make children
        c1 = crossover(p1, p2)
        mutate(c1)
        c2 = crossover(p1, p2)
        mutate(c2)

        # Get sorted fitnesses of group (least fit to most)
        group_ids = [i[0] for i in group]
        #group_fits = [(s.Id(), s.Fitness()) for s in fit_scores if s.Id() in group_ids]
        group_fits = [ (net_id, fit_scores[net_id]) for net_id in group_ids]
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
    for i in range(net.in_size-1 ):
        for j in range(net.out_size ):
            if (random.random() < mutate_prob):
                net.w_inp_out[i][j] = random.random()


k = 4
pop_size = 10
input_size = 5
output_size = 1


env = gym.make('CartPole-v0')

# Generate population
pop = {i:NeuralNet(input_size,0,output_size) for i in range(pop_size)}


for epoch in range(100):
    fit_scores = {}

    # Run simulations
    for net_id, net in pop.iteritems():
        obs = env.reset()
        sum_reward = 0

        for _ in range(1000):
            #if epoch == 9:
            #env.render()

            action = net.activate( obs )[0]
            obs, reward, done, info = env.step( action )

            sum_reward += reward

            if done:
                break

        fit_scores[net_id] = sum_reward

    pop = tourn_select(fit_scores, pop)

    print 'epoch {0}: top fitness - {1}'.format(epoch, max(fit_scores.values()))
