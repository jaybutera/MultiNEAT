import zmq
import flatbuffers
from flatbuffers import number_types as N
import AI.Store.Ids as s_i
import AI.Obs.Observations as o_fb
import AI.Obs.Creature as o_c
import AI.Obs.Epoch as e_fb
import AI.Control.Actions as c_a
import AI.Control.Move as c_m

class EvoComm (object):
    def __init__ (self):
        self.iteration = 0
        self.builder = flatbuffers.Builder(2048)

    def connect (self):
        # ZMQ
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind('tcp://*:%s' % port)

    def send_actions (self, actions, ids):
        creat_actions = []

        output_size = len( actions[0] )

        for i in range( len(actions) ):
            net_id = ids[i]

            c_m.MoveStartOutputVector(builder, output_size)
            #for out in outs:
            for a in actions[i]:
                builder.PrependFloat32(a);

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

        if self.iteration % 100 == 0:
            print "Action serialize (ms)=%s" % ((t2 - t1)*1000)

        # Build action table
        t1 = time.time()
        c_a.ActionsStart(builder)
        c_a.ActionsAddAction(builder, action_vec)
        a_offset = c_a.ActionsEnd(builder)
        builder.Finish(a_offset)

        action_fb = builder.Output()

        # Send actions
        socket.send(action_fb)
        t2 = time.time()

        if (self.iteration % 100 == 0):
            print "Build time: {0}".format((t2-t1)*1000)
            print "Total time: {0}".format((t2-t_begin)*1000)

        self.iteration += 1

    '''
    Encapsulate observations fb message
    '''
    def get_obs (buf):
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
