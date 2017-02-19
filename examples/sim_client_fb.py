import flatbuffers
import AI.Obs.Observations as o_fb
import AI.Obs.Creature as o_c
import AI.Obs.Smell as o_smell
import AI.Obs.Epoch as o_e
import AI.Obs.Score as o_s
import AI.Store.Ids as s_i
import AI.Control.Actions as c_a
import AI.Control.Move as c_m
import random
import zmq

port = '5559'
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://localhost:%s' % port)

builder = flatbuffers.Builder(1024)

# Start command
socket.send('start')

def send_obs(ids):
    # Build creatures list
    creatures = []
    for c in ids:
        # Build view for creature
        '''
        o_c.CreatureStartViewVector(builder,view_size)

        for i in reversed( range(view_size) ):
            builder.PrependByte(random.randint(0,5))

        view = builder.EndVector(view_size)
        #
        '''

        # Build creature in fb
        o_c.CreatureStart(builder)
        o_c.CreatureAddId(builder, c)
        o_c.CreatureAddSmell(builder, o_smell.CreateSmell(builder, 1.0, 1.0, 1.0))

        #o_c.CreatureAddView(builder, view)

        creatures.append( o_c.CreatureEnd(builder) )

    # Build observations vector in fb
    o_fb.ObservationsStartObsVector(builder, len(ids))

    for c in creatures:
        builder.PrependUOffsetTRelative(c)

    obs = builder.EndVector( len(ids) )
    #

    # Builder Observations table
    o_fb.ObservationsStart(builder)
    o_fb.ObservationsAddObs(builder, obs)
    o_offset = o_fb.ObservationsEnd(builder)

    builder.Finish(o_offset)

    obs_fb = builder.Output()

    print 'Sending observation buffer...'
    socket.send(obs_fb)


def get_actions():
    # Get action vector
    buf = socket.recv()

    actions = c_a.Actions.GetRootAsActions(buf, 0)
    action_len = actions.ActionLength()
    print 'Num actions'
    print action_len

    moves_fb = [actions.Action(i) for i in range(action_len)]

    out_size = moves_fb[0].OutputLength()
    moves = {}
    for m in moves_fb:
        output = [m.Output(i) for i in range(out_size)]
        moves[m.Id()] = output

    print 'Action vector'
    print moves




for generation in range(5):
    # Get id vector
    buf = bytearray(socket.recv())
    ids_fb = s_i.Ids.GetRootAsIds(buf, 0)

    ids_len = ids_fb.IdvecLength()
    print 'Ids length: %d' % ids_len

    ids = []
    for i in range( ids_len ):
        ids.append( ids_fb.Idvec(i) )

    print ids

    num_creat = len(ids)
    #view_size = 65

    for step in range(5):
        # For and send observation
        send_obs(ids)

        # Recieve actions
        get_actions()

    # Now with less creatures (some die)
    ids = ids[:-1]
    for step in range(5):
        # For and send observation
        send_obs(ids)

        # Recieve actions
        get_actions()

    socket.send('epoch')
    socket.recv()

    tmp_scores = []
    for c in ids:
        o_s.ScoreStart(builder)
        o_s.ScoreAddId(builder, c)
        o_s.ScoreAddFitness(builder, random.random()*50)
        s = o_s.ScoreEnd(builder)

        tmp_scores.append(s)

    o_e.EpochStartScoreVector(builder, num_creat)

    for s in tmp_scores:
        builder.PrependUOffsetTRelative(s)

    scores = builder.EndVector(num_creat)

    o_e.EpochStart(builder)
    o_e.EpochAddScore(builder, scores)
    e_offset = o_e.EpochEnd(builder)
    builder.Finish(e_offset)

    socket.send( builder.Output() )
