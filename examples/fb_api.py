import zmq
import time
import flatbuffers
from flatbuffers import number_types as N
import AI.Store.Ids as s_i
import AI.Obs.Observations as o_fb
import AI.Obs.Creature as o_c
import AI.Obs.Epoch as e_fb
import AI.Control.Actions as c_a
import AI.Control.Move as c_m

'''
    Communicate with a simulator API via flatbuffers and zeromq.
    Works in a replace-all evolution strategy.

    Method descriptions
    -------------------
    send_ids(ids)         - send id list to the server
    next_epoch            - acknowledge end epoch request from simulator and get fitness scores
    connect               - connect to simulator via zeromq
    send_actions(actions) - send brain outputs to simulator for actions
    get_obs()             - recieve creature observations from simulator
'''
class EvoComm (object):
    def __init__ (self):
        self.iteration = 0
        self.builder = flatbuffers.Builder(2048)

    def send_ids (self, ids):
        # Build id vector in fbuf
        num_ids = len(ids)
        s_i.IdsStartIdvecVector(self.builder, num_ids)

        for i in reversed( ids ):
            self.builder.PrependUint16(i)

        idvec = self.builder.EndVector(num_ids)

        # Build ids fbuf
        s_i.IdsStart(self.builder)
        s_i.IdsAddIdvec(self.builder, idvec)
        ids_offset = s_i.IdsEnd(self.builder)
        self.builder.Finish(ids_offset)

        # Send out ids vector
        ids_fb = self.builder.Output()
        self.socket.send(ids_fb)

    def next_epoch(self):
        # Reset buffer
        self.builder = flatbuffers.Builder(2048)

        # Acknowledge next epoch signal
        self.socket.send('recieved')

        # Get fitness scores
        buf = self.socket.recv()

        epoch = e_fb.Epoch.GetRootAsEpoch(buf, 0)
        score_len = epoch.ScoreLength()

        fit_scores = {epoch.Score(i).Id() : epoch.Score(i).Fitness() for i in range(score_len)}
        return fit_scores

    def connect (self, port):
        # ZMQ
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind('tcp://*:%s' % port)

        # Wait for response
        while True:
            msg = self.socket.recv()
            if msg == 'start':
                break

    def send_actions (self, actions):
        creat_actions = []

        output_size = len( actions.values()[0] )

        for net_id, action in actions.iteritems():
            c_m.MoveStartOutputVector(self.builder, output_size)
            #for out in outs:
            for a in action:
                self.builder.PrependFloat32(a);

            output = self.builder.EndVector(output_size)

            # Store move table in py list
            c_m.MoveStart(self.builder)
            c_m.MoveAddId(self.builder, net_id)
            c_m.MoveAddOutput(self.builder, output)
            creat_actions.append( c_m.MoveEnd(self.builder) )

        #t2 = time.time()
        #print 'ANN sim (s) - {0}'.format(t2-t1)

        t1 = time.time()
        num_creats = len(creat_actions)
        c_a.ActionsStartActionVector(self.builder, num_creats)

        for o in creat_actions:
            self.builder.PrependUOffsetTRelative(o)

        action_vec = self.builder.EndVector(num_creats)
        t2 = time.time()

        if self.iteration % 100 == 0:
            print "Action serialize (ms)=%s" % ((t2 - t1)*1000)

        # Build action table
        t1 = time.time()
        c_a.ActionsStart(self.builder)
        c_a.ActionsAddAction(self.builder, action_vec)
        a_offset = c_a.ActionsEnd(self.builder)
        self.builder.Finish(a_offset)

        action_fb = self.builder.Output()

        # Send actions
        self.socket.send(action_fb)
        t2 = time.time()

        if (self.iteration % 100 == 0):
            print "Build time: {0}".format((t2-t1)*1000)
            #print "Total time: {0}".format((t2-t_begin)*1000)

        self.iteration += 1

    def get_obs (self):
        # Get observations
        buf = self.socket.recv()

        # Check if end epoch signal
        if ('epoch' in buf):
            return []

        #t1 = time.time()
        obs = o_fb.Observations.GetRootAsObservations(buf, 0)
        obs_len = obs.ObsLength()

        #observations = [obs.Obs(i) for i in range(obs_len)]

        # (creature_id, observation [list])
        observations = {}

        for i in range( obs_len ):
            # Get observation for creature i
            o = obs.Obs(i)

            net_id = o.Id()

            inp_vec = [ \
                #o.Smell().Protein(), \
                #o.Smell().Starch(), \
                o.Smell().Fat(), \
                o.AngAccel(),
                o.Accel().X(), \
                o.Accel().Y(),
                1.0] # bias

            observations[net_id] = inp_vec

        #t2 = time.time()

        return observations
