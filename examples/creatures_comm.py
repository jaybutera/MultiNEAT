#!/usr/bin/python3
import os
import sys
import time
import ctypes
import random as rnd
import subprocess as comm
import cv2
import numpy as np
import pickle as pickle
import MultiNEAT as NEAT
from MultiNEAT import GetGenomeList, ZipFitness
from MultiNEAT import EvaluateGenomeList_Serial
from fb_api import EvoComm

import concurrent.futures

params = NEAT.Parameters()
params.PopulationSize = 32

params.DynamicCompatibility = True
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 25
params.OldAgeTreshold = 35
params.MinSpecies = 4
params.MaxSpecies = 10
params.RouletteWheelSelection = False

params.MutateRemLinkProb = 0.02
params.RecurrentProb = 0.05
params.OverallMutationRate = 0.05
params.MutateAddLinkProb = 0.08
params.MutateAddNeuronProb = 0.01
params.MutateWeightsProb = 0.80
params.MaxWeight = 8.0
params.WeightMutationMaxPower = 0.2
params.WeightReplacementMaxPower = 1.0

params.MutateActivationAProb = 0.0
params.ActivationAMutationMaxPower = 0.5
params.MinActivationA = 0.05
params.MaxActivationA = 6.0

params.MutateNeuronActivationTypeProb = 0.03

params.ActivationFunction_SignedSigmoid_Prob = 0.0
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_Tanh_Prob = 1.0
params.ActivationFunction_TanhCubic_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 1.0
params.ActivationFunction_UnsignedStep_Prob = 0.0
params.ActivationFunction_SignedGauss_Prob = 1.0
params.ActivationFunction_UnsignedGauss_Prob = 0.0
params.ActivationFunction_Abs_Prob = 0.0
params.ActivationFunction_SignedSine_Prob = 1.0
params.ActivationFunction_UnsignedSine_Prob = 0.0
params.ActivationFunction_Linear_Prob = 1.0

params.DivisionThreshold = 0.5
params.VarianceThreshold = 0.03
params.BandThreshold = 0.3
params.InitialDepth = 2
params.MaxDepth = 3
params.IterationLevel = 1
params.Leo = False
params.GeometrySeed = False
params.LeoSeed = False
params.LeoThreshold = 0.3
params.CPPN_Bias = -1.0
params.Qtree_X = 0.0
params.Qtree_Y = 0.0
params.Width = 1.
params.Height = 1.
params.Elitism = 0.1

rng = NEAT.RNG()
rng.TimeSeed()

input_size = 2
# Smell inputs
inputs=[(x,-1.,0.) for x in np.linspace(-.3,.3,input_size)]
# Acceleration inputs
inputs.extend([(-.2,-.8,.3),(.2,-.8,.3)])

output_size = 2
outputs=[(x,1.,0.) for x in np.linspace(-1,1,output_size)]

substrate = NEAT.Substrate(inputs,
                           [],
                           outputs)
                           #[(.5,.4,0),(.5,.6,0)],
                           #[(.3,1.,0.0),(-.3,1.,0.0)])

substrate.m_allow_output_hidden_links = False
substrate.m_allow_output_output_links = False
substrate.m_allow_looped_hidden_links = True
substrate.m_allow_looped_output_links = False

substrate.m_allow_input_hidden_links = True
substrate.m_allow_input_output_links = True
substrate.m_allow_hidden_output_links = True
substrate.m_allow_hidden_hidden_links = True

substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID

substrate.m_with_distance = False

substrate.m_max_weight_and_bias = 8.0

port = sys.argv[1]

# Open log file
f = open('log.txt', 'w')
fit_f = open('fit_log.txt', 'w')

g = NEAT.Genome(0,
                substrate.GetMinCPPNInputs()+1, # +1 for bias
                0,
                substrate.GetMinCPPNOutputs(),
                False,
                NEAT.ActivationFunction.SIGNED_SIGMOID,
                NEAT.ActivationFunction.SIGNED_SIGMOID,
                0,
                params)

pop = NEAT.Population(g, params, True, 1.0, 0)

# Communictor to simulator
comm = EvoComm()

# Connect to simulator
comm.connect(port)
print 'connected...'


'''
BEGIN EVOLUTION
'''

while True: # Never ending generations
    iteration = 0

    # Evaluate genomes
    genome_list = NEAT.GetGenomeList(pop)

    # For convenience
    genome_dict = {}
    for g in genome_list:
        genome_dict[g.GetID()] = g

    # Construct networks
    nets = {}
    for g in genome_list:
        net = NEAT.NeuralNetwork()
        g.BuildESHyperNEATPhenotype(net, substrate, params)
        nets[g.GetID()] = net

    # send ids to simulator
    ids = nets.keys()
    comm.send_ids(ids)

    # Step process
    while True:
        # Get observations from simulator
        observations = comm.get_obs()

        # Next epoch if observations empty
        if not observations:
            fit_scores = comm.next_epoch()
            # Apply fitness scores
            [genome_dict[s.Id()].SetFitness(s.Fitness()) for s in fit_scores]

            break

        # Generate actions (net_id, action)
        action = {}
        for net_id, inp_vec in observations.iteritems():
            net = nets[net_id]
            net.Input(inp_vec)
            net.Activate()
            outs = net.Output()

            action[net_id] = outs

        # Send actions to simulator
        comm.send_actions(action)

        iteration += 1


    '''
    LOGGING AND VISUALS
    '''

    # Print best fitness
    best_genome = pop.GetBestGenome()
    epoch_log = "---------------------------\n" + \
                'Generation: %d\n' % pop.GetGeneration() + \
                'Best fitness: %d\n' % best_genome.GetFitness() + \
                "Best fitness in history: %d\n" % pop.GetBestFitnessEver()
    print epoch_log
    f.write(epoch_log)
    fit_f.write( str(best_genome.GetFitness()) + '\n' )


    # Visualize best network's Genome
    net = NEAT.NeuralNetwork()
    pop.Species[0].GetLeader().BuildPhenotype(net)
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img += 10
    NEAT.DrawPhenotype(img, (0, 0, 500, 500), net)
    cv2.imshow("CPPN", img)
    # Visualize best network's Pheotype
    net = NEAT.NeuralNetwork()
    pop.Species[0].GetLeader().BuildESHyperNEATPhenotype(net, substrate, params)
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img += 10

    NEAT.DrawPhenotype(img, (0, 0, 500, 500), net, substrate=True)
    cv2.imshow("NN", img)
    cv2.waitKey(1)

    #

    pop.Epoch()
    continue

f.close()
fit_f.close()
