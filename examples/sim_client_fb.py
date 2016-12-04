import flatbuffers
import AI.Obs.Observations as o_fb
import AI.Obs.Creature as o_c
import AI.Store.Ids as s_i
import random
import zmq

port = '5560'
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://localhost:%s' % port)

builder = flatbuffers.Builder(1024)

# Start command
socket.send('start')

# Get id vector
buf = bytearray(socket.recv())
print type(buf)
ids_fb = s_i.Ids.GetRootAsIds(buf, 0)

ids_len = ids_fb.IdvecLength()
print 'Ids length: %d' % ids_len

ids = []
for i in range( ids_len ):
    ids.append( ids_fb.Idvec(i) )

print ids

num_creat = len(ids)
view_size = 65

# Build creatures list
creatures = []
for c in ids:
    # Build view for creature
    o_c.CreatureStartViewVector(builder,view_size)

    for i in reversed( range(view_size) ):
        builder.PrependByte(random.randint(0,5))

    view = builder.EndVector(view_size)
    #

    # Build creature in fb
    o_c.CreatureStart(builder)
    o_c.CreatureAddId(builder, c)
    o_c.CreatureAddView(builder, view)

    creatures.append( o_c.CreatureEnd(builder) )

# Build observations vector in fb
o_fb.ObservationsStartObsVector(builder, num_creat)

for c in creatures:
    builder.PrependUOffsetTRelative(c)

obs = builder.EndVector(num_creat)
#

# Builder Observations table
o_fb.ObservationsStart(builder)
o_fb.ObservationsAddObs(builder, obs)
o_offset = o_fb.ObservationsEnd(builder)

builder.Finish(o_offset)

obs_fb = builder.Output()
print 'Sending observation buffer...'
socket.send(obs_fb)
