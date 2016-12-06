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
import zmq
import flatbuffers
from flatbuffers import number_types as N
import AI.Store.Ids as s_i
import AI.Obs.Observations as o_fb
import AI.Obs.Creature as o_c
import AI.Control.Actions as c_a
import AI.Control.Move as c_m
from MultiNEAT import GetGenomeList, ZipFitness
from MultiNEAT import EvaluateGenomeList_Serial

from concurrent.futures import ProcessPoolExecutor, as_completed

params = NEAT.Parameters()
params.PopulationSize = 65

params.DynamicCompatibility = True
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 100
params.OldAgeTreshold = 35
params.MinSpecies = 5
params.MaxSpecies = 10
params.RouletteWheelSelection = False

params.MutateRemLinkProb = 0.02
params.RecurrentProb = 0
params.OverallMutationRate = 0.15
params.MutateAddLinkProb = 0.08
params.MutateAddNeuronProb = 0.01
params.MutateWeightsProb = 0.90
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

input_size = 65
inputs=[(x,-1.,0.) for x in np.linspace(-1,1,input_size-3)] # 62 raycasts
inputs.extend([(-.2,-.8,.3),(.2,-.8,.3),(0.,-.6,-.5)])

output_size = 2
outputs=[(x,1.,0.) for x in np.linspace(-1,1,output_size)]

substrate = NEAT.Substrate(inputs,
                           [],
                           outputs)
                           #[(.5,.4,0),(.5,.6,0)],
                           #[(.3,1.,0.0),(-.3,1.,0.0)])

substrate.m_allow_output_hidden_links = False
substrate.m_allow_output_output_links = False
substrate.m_allow_looped_hidden_links = False
substrate.m_allow_looped_output_links = False

substrate.m_allow_input_hidden_links = True
substrate.m_allow_input_output_links = False
substrate.m_allow_hidden_output_links = True
substrate.m_allow_hidden_hidden_links = True

substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID

substrate.m_with_distance = False

substrate.m_max_weight_and_bias = 8.0

port = sys.argv[1]

# ZMQ
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:%s' % port)
#socket.connect('tcp://localhost:'+port)

g = NEAT.Genome(0,
                substrate.GetMinCPPNInputs()+1, # +1 for bias
                0,
                substrate.GetMinCPPNOutputs(),
                False,
                NEAT.ActivationFunction.TANH,
                NEAT.ActivationFunction.TANH,
                0,
                params)

pop = NEAT.Population(g, params, True, 1.0, 0)

# Initialize flat buffers builder
builder = flatbuffers.Builder(1024)

while True:
    msg = socket.recv()
    if msg == 'start':
        break

print 'connected...'

#for generation in range(1000):
while True: # Never ending generations
    # Evaluate genomes
    genome_list = NEAT.GetGenomeList(pop)

    # Construct networks
    nets = {}
    for g in genome_list:
        net = NEAT.NeuralNetwork()
        g.BuildESHyperNEATPhenotype(net, substrate, params)
        nets[g.GetID()] = net


    # Build id vector in fbuf
    num_ids = len(nets)
    s_i.IdsStartIdvecVector(builder, num_ids)

    for i in reversed( nets.keys() ):
        builder.PrependByte(i)

    idvec = builder.EndVector(num_ids)

    # Build ids fbuf
    s_i.IdsStart(builder)
    s_i.IdsAddIdvec(builder, idvec)
    ids_offset = s_i.IdsEnd(builder)
    builder.Finish(ids_offset)

    # Send out ids vector
    ids_fb = builder.Output()
    socket.send(ids_fb)

    # Step process
    while True:
        buf = socket.recv()

        # Epoch
        if ('epoch' in buf):
            break

        obs = o_fb.Observations.GetRootAsObservations(buf, 0)
        obs_len = obs.ObsLength()

        observations = [obs.Obs(i) for i in range(obs_len)]
        creat_obs = []
        #outputs = dict()
        #action = []

        a_builder = flatbuffers.Builder(1024)

        # Check that input size in simulation matches server assumption
        i_size = observations[0].ViewLength()
        if i_size != input_size:
            print 'Confiured input size [{0}] does not match client input size [{1}]\nCrashing...'.format(input_size, i_size)
            break

        for o in observations:
            net_id = o.Id()
            inp_vec = [o.View(i) for i in range(i_size)]

            net = nets[net_id]
            net.Input(inp_vec)
            net.Activate()
            outs = net.Output()

            c_m.MoveStartOutputVector(a_builder, output_size)
            #for out in outs:
            for o in outs:
                a_builder.PrependFloat32(o);
                '''
                a_builder.PrependFloat32Slot(int(i),
                        float(o),
                        # N.Float32Flags.py_type(ctypes.c_float(o).value),
                        0.0)
                '''

            output = a_builder.EndVector(output_size)

            # Store move table in py list
            c_m.MoveStart(a_builder)
            c_m.MoveAddId(a_builder, net_id)
            c_m.MoveAddOutput(a_builder, output)
            creat_obs.append( c_m.MoveEnd(a_builder) )

            '''
            action.append({
                'id' : net_id,
                'action' : outs
            })
            '''

        c_a.ActionsStartActionVector(a_builder, num_ids)

        for o in creat_obs:
            a_builder.PrependUOffsetTRelative(o)

        action_vec = a_builder.EndVector(num_ids)

        # Build Actions table
        c_a.ActionsStart(a_builder)
        c_a.ActionsAddAction(a_builder, action_vec)
        a_offset = c_a.ActionsEnd(a_builder)
        a_builder.Finish(a_offset)

        action_fb = a_builder.Output()

        socket.send(action_fb)
        #outputs['actions'] = action
        #socket.send_json(outputs)

    fit_info = buf['epoch']

    fit_obs = {fit_val['id'] : fit_val['aliveTime'] for fit_val in fit_info}

    # Set fitnesses
    #[genome_list[key].SetFitness(fit_obs[key]) for key in fit_obs]
    #[genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)] 
    # print('Gen: %d Best: %3.5f' % (generation, max(fitnesses)))

    # Print best fitness
    print("---------------------------")
    print("Generation: {0}".format(pop.GetGeneration()) )
    # print("max ", max([x.GetLeader().GetFitness() for x in pop.Species]))


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

    pop.Epoch()
    continue
