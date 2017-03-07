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
import AI.Obs.Epoch as e_fb
import AI.Control.Actions as c_a
import AI.Control.Move as c_m
from MultiNEAT import GetGenomeList, ZipFitness
from MultiNEAT import EvaluateGenomeList_Serial

import concurrent.futures
#from concurrent.futures import ProcessPoolExecutor, as_completed
#executor = ProcessPoolExecutor(4)

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

# ZMQ
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:%s' % port)
#socket.connect('tcp://localhost:'+port)

# Time benchmark decortator
def st_time(func):
    """
        st decorator to calculate the total time of a func
    """

    def st_func(iter, *args, **keyArgs):
        t1 = time.time()
        r = func(iter, *args, **keyArgs)
        t2 = time.time()
        if iter % 100 == 0:
            print "Function=%s, Time (ms)=%s" % (func.__name__, (t2 - t1)*1000)
        return r

    return st_func

'''
Encapsulate observations fb message
'''
def fb_obs (iter, buf):
    t1 = time.time()
    obs = o_fb.Observations.GetRootAsObservations(buf, 0)
    obs_len = obs.ObsLength()

    observations = [obs.Obs(i) for i in range(obs_len)]
    t2 = time.time()

    return observations

def step_nn (o): # o is observation
    net_id = o.Id()

    inp_vec = [ \
        #o.Smell().Protein(), \
        #o.Smell().Starch(), \
        o.Smell().Fat(), \
        o.AngAccel(),
        o.Accel().X(), \
        o.Accel().Y(),
        1]

    net = nets[net_id]
    net.Input(inp_vec)
    net.Activate()
    outs = net.Output()

    return outs


def gen_actions (iter, observations, builder):
    creat_actions = []

    #t1 = time.time()
    #actions = []
    #executor = concurrent.futures.ProcessPoolExecutor(8)
    #futures = [executor.submit(step_nn, o) for o in observations]

    '''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for o, outs in zip(observations, executor.map(step_nn, observations)):
            net_id = o.Id()

            c_m.MoveStartOutputVector(builder, output_size)
            #for out in outs:
            for o in outs:
                builder.PrependFloat32(o);

            output = builder.EndVector(output_size)

            # Store move table in py list
            c_m.MoveStart(builder)
            c_m.MoveAddId(builder, net_id)
            c_m.MoveAddOutput(builder, output)
            creat_actions.append( c_m.MoveEnd(builder) )


    '''
    for o in observations:
        net_id = o.Id()

        inp_vec = [ \
            #o.Smell().Protein(), \
            #o.Smell().Starch(), \
            o.Smell().Fat(), \
            o.AngAccel(),
            o.Accel().X(), \
            o.Accel().Y(),
            1.0]

        net = nets[net_id]
        net.Input(inp_vec)
        net.Activate()
        outs = net.Output()
        #print '[{0}] : {1}'.format(net_id, outs)

        c_m.MoveStartOutputVector(builder, output_size)
        #for out in outs:
        for o in outs:
            builder.PrependFloat32(o);

        output = builder.EndVector(output_size)

        # Store move table in py list
        c_m.MoveStart(builder)
        c_m.MoveAddId(builder, net_id)
        c_m.MoveAddOutput(builder, output)
        creat_actions.append( c_m.MoveEnd(builder) )

    #t2 = time.time()
    #print 'ANN sim (s) - {0}'.format(t2-t1)

    t1 = time.time()
    num_creats = len(creat_actions)
    c_a.ActionsStartActionVector(builder, num_creats)

    for o in creat_actions:
        builder.PrependUOffsetTRelative(o)

    action_vec = builder.EndVector(num_creats)
    t2 = time.time()

    if iter % 100 == 0:
        print "Action serialize (ms)=%s" % ((t2 - t1)*1000)

    return action_vec


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

# Initialize flat buffers builder
#builder = flatbuffers.Builder(1024)

while True:
    msg = socket.recv()
    if msg == 'start':
        break

print 'connected...'

#for generation in range(1000):
while True: # Never ending generations
    builder = flatbuffers.Builder(2048)
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


    # Build id vector in fbuf
    num_ids = len(nets)
    s_i.IdsStartIdvecVector(builder, num_ids)

    for i in reversed( nets.keys() ):
        builder.PrependUint16(i)

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
            socket.send('recieved')
            break

        # Build observations list
        obs_timer = st_time( fb_obs )
        observations = obs_timer(iteration, buf)

        # Check that input size in simulation matches server assumption
        '''
        i_size = observations[0].ViewLength()
        if i_size != input_size:
            print 'Confiured input size [{0}] does not match client input size [{1}]\nCrashing...'.format(input_size, i_size)
            break
        '''

        # Simulate ANNs and generate fb action vector
        t_begin = time.time()
        a_timer = st_time( gen_actions )
        action_vec = a_timer (iteration, observations, builder)

        # Build Actions table
        t1 = time.time()
        c_a.ActionsStart(builder)
        c_a.ActionsAddAction(builder, action_vec)
        a_offset = c_a.ActionsEnd(builder)
        builder.Finish(a_offset)

        action_fb = builder.Output()

        socket.send(action_fb)
        t2 = time.time()

        if (iteration % 100 == 0):
            print "Build time: {0}".format((t2-t1)*1000)
            print "Total time: {0}".format((t2-t_begin)*1000)
        #outputs['actions'] = action
        #socket.send_json(outputs)
        iteration += 1

    buf = socket.recv()

    # Apply fitness scores
    epoch = e_fb.Epoch.GetRootAsEpoch(buf, 0)
    score_len = epoch.ScoreLength()

    fit_scores = [epoch.Score(i) for i in range(score_len)]
    [genome_dict[s.Id()].SetFitness(s.Fitness()) for s in fit_scores]

    # print('Gen: %d Best: %3.5f' % (generation, max(fitnesses)))

    # Print best fitness
    print("---------------------------")
    print("Generation: {0}".format(pop.GetGeneration()) )
    print("max ", max([x.GetLeader().GetFitness() for x in pop.Species]))


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
