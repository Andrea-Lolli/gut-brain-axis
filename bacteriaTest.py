import sys
import math
import numpy as np
import networkx as nx
import repast4py
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass

import numba
from numba import int32, int64
from numba.experimental import jitclass

from repast4py import core, space, schedule, logging, random
from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params

from repast4py.space import ContinuousPoint as cpt
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType

from repast4py.network import write_network, read_network

model = None

@numba.jit((int64[:], int64[:]), nopython=True)
def is_equal(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]


spec = [
    ('mo', int32[:]),
    ('no', int32[:]),
    ('xmin', int32),
    ('ymin', int32),
    ('ymax', int32),
    ('xmax', int32)
]


@jitclass(spec)
class GridNghFinder:

    def __init__(self, xmin, ymin, xmax, ymax):
        self.mo = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, -1, -1, -1], dtype=np.int32)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def find(self, x, y):
        xs = self.mo + x
        ys = self.no + y

        xd = (xs >= self.xmin) & (xs <= self.xmax)
        xs = xs[xd]
        ys = ys[xd]

        yd = (ys >= self.ymin) & (ys <= self.ymax)
        xs = xs[yd]
        ys = ys[yd]

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)
    
class Bacteria_a(core.Agent):

    TYPE = 0
    mitosis = False
    pt = None

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Bacteria_a.TYPE, rank=rank)

    def save(self):
        return (self.uid, self.mitosis, self.pt)

    def step(self):
        cpt = model.grid.get_location(self)
        ngh = model.ngh_finder.find(cpt.x, cpt.y)
        #print("\nTEST BATTERIO A: \n")
        at = dpt(0, 0)
        for pt in ngh:
            at._reset_from_array(pt)
            if len(list(model.grid.get_agents(at))) == 0: #TODO da cambiare di sicuro c'è un modo migliore (fa un po' schifo...)
                #print("trovata cella vuota! {}".format(pt))
                self.mitosis = True
                self.pt = at
                #print("aggiungo batterio A")
                break

    
class Bacteria_b(core.Agent):

    TYPE = 1
    testParam = 0

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Bacteria_b.TYPE, rank=rank)

    def save(self):
        return (self.uid, self.testParam)

    def step(self):
        return


agent_cache = {}

def restore_agent(agent_data: Tuple):
    uid = agent_data[0]
    # 0 is id, 1 is type, 2 is rank
    if uid[1] == Bacteria_a.TYPE:
        if uid in agent_cache:
            ba = agent_cache[uid]
        else:
            ba = Bacteria_a(uid[0], uid[2])
            agent_cache[uid] = ba

        # restore the agent state from the agent_data tuple
        ba.mitosis = agent_data[1]
        ba.pt = agent_data[2]
        return ba
    else:
        if uid in agent_cache:
            return agent_cache[uid]
        else:
            bb = Bacteria_b(uid[0], uid[2])
            agent_cache[uid] = bb
            return bb


@dataclass
class Counts:
    """Dataclass used by repast4py aggregate logging to record
    the number of bacteria after each tick.
    """
    bacteria_a: int = 0
    bacteria_b: int = 0


class Model:

    def __init__(self, comm, params):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        self.grid = space.SharedGrid('grid', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)
        self.context.add_projection(self.grid)

        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)
        
        self.counts = Counts()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, params['counts_file'])

        rank = comm.Get_rank()
        rng = repast4py.random.default_rng
        world_size = comm.Get_size()

        # Add bacteria a
        total_ba_count = params['bacteria1.count']
        local_ba_count = int(total_ba_count / world_size)
        if self.rank < total_ba_count % world_size:
            local_ba_count += 1

        for i in range(local_ba_count):
            pt = self.grid.get_random_local_pt(rng) # TODO CAMBIARE! sta cosa è tutto tranne che random... 
            ba = Bacteria_a(i, rank)
            self.context.add(ba)
            self.grid.move(ba, pt)

        # Add bacteria b
        total_bb_count = params['bacteria2.count']
        local_bb_count = int(total_bb_count / world_size)
        if self.rank < total_bb_count % world_size:
            local_bb_count += 1   

        for i in range(local_bb_count):
            #print("aggiunto batterio B numero: {}, al rank: {}".format(i, self.rank))
            pt = self.grid.get_random_local_pt(rng) # TODO controllare che le posizioni non sono duplicate...
            bb = Bacteria_b(i, rank)
            self.context.add(bb)
            self.grid.move(bb, pt)

        self.bacteria_a_id = local_ba_count

    def step(self): 
        tick = self.runner.schedule.tick
        self.log_counts(tick)
        self.context.synchronize(restore_agent)
        
        pts = []
        for agent in self.context.agents(Bacteria_a.TYPE):
            agent.step()
            if agent.mitosis:
                if agent.pt not in pts:
                    pts.append(agent.pt)
                agent.pt = None
                agent.mitosis = False

        #duplicazione batteri
        for pt in pts:
            if len(list(model.grid.get_agents(pt))) == 0:
                if not(pt.x < 0 or pt.x >= 10 or pt.y < 0 or pt.y >= 10):
                    model.add_bacteria_a(pt)

        model.render()

        nb1, nb2 = 0, 0
        for agent in self.context.agents():
            if isinstance(agent, Bacteria_a):
                nb1 += 1
            elif isinstance(agent, Bacteria_b):
                nb2 += 1
        #print("Batteri a: {}, Batteri b: {}, rank: {}".format(nb1, nb2, self.rank))
        

    def log_counts(self, tick):
        num_agents = self.context.size([Bacteria_a.TYPE, Bacteria_b.TYPE])
        self.counts.bacteria_a = num_agents[Bacteria_a.TYPE]
        self.counts.bacteria_b = num_agents[Bacteria_b.TYPE]
        self.data_set.log(tick)

        # Do the cross-rank reduction manually and print the result
        if tick % 10 == 0:
            ba_count = np.zeros(1, dtype='int64')
            bb_count = np.zeros(1, dtype='int64')
            self.comm.Reduce(np.array([self.counts.bacteria_a], dtype='int64'), ba_count, op=MPI.SUM, root=0)
            self.comm.Reduce(np.array([self.counts.bacteria_b], dtype='int64'), bb_count, op=MPI.SUM, root=0)
            #if (self.rank == 0):
                #print("Batteri a: {}, Batteri b: {}, rank: {}".format(ba_count[0], bb_count[0], self.rank),flush=True)

    def at_end(self):
        self.data_set.close()

        for agent in self.context.agents():
            cpt = model.grid.get_location(agent)
            if isinstance(agent, Bacteria_a):
                print("posizione batterio A: x {}, y: {}".format(cpt.x, cpt.y))
            if isinstance(agent, Bacteria_b):
                print("posizione batterio B: x {}, y: {}".format(cpt.x, cpt.y))

        #self.agent_logger.close()

    def move(self, agent, x, y):
        self.grid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))

    def start(self):
        self.runner.execute()

    def run(self):
        self.runner.execute()

    def remove_agent(self, agent):
        self.context.remove(agent)


    # TODO sarebbe meglio trovare il modo di toglierlo, alcuni batteri si sovrappongono quando vengono creati ai confini del rank
    # si può levare se non importa il count preciso dei batteri
    def render(self):
        pts = []
        agents = []
        for agent in model.context.agents(Bacteria_a.TYPE, Bacteria_b.TYPE):
            cpt = model.grid.get_location(agent)
            if cpt not in pts:
                pts.append(cpt)
            else:
                #print("eliminato un agente duplicato: x {}, y: {}".format(cpt.x, cpt.y))
                agents.append(agent)
        for agent in agents:
            model.remove_agent(agent)

    #TODO cambiare self.rank e controllare il rank adatto con il local_bound (non è detto che serve c'è il restore)
    def add_bacteria_a(self, pt): 
        ba = Bacteria_a(self.bacteria_a_id, self.rank) 
        self.bacteria_a_id += 1
        self.context.add(ba)
        pt1 = self.grid.move(ba, pt)
        #print("aggiungo batterio alla posizione: x: {}, y:{} - (x: {}, y:{})".format(pt1.x,pt1.y, pt.x, pt.y))


def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.run()


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)