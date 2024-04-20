import sys
import math
import numpy as np
import networkx as nx
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
        self.mo = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)
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


#Test network
def generate_network_file(fname: str, n_ranks: int, n_agents: int):
    """Generates a network file using repast4py.network.write_network.

    Args:
        fname: the name of the file to write to
        n_ranks: the number of process ranks to distribute the file over
        n_agents: the number of agents (node) in the network
    """
    g = nx.connected_watts_strogatz_graph(n_agents, 2, 0.25)
    try:
        import nxmetis
        write_network(g, 'neuron_network', fname, n_ranks, partition_method='metis')
    except ImportError:
        write_network(g, 'neuron_network', fname, n_ranks)


class GutCell(core.Agent):

    TYPE = 0
    random_param_test = 0

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=GutCell.TYPE, rank=rank)

    def save(self):
        return (self.uid, self.random_param_test)

    def step(self):
        print("boh...")


class Neuron(core.Agent):

    TYPE = 1
    random_param_test = 0

    def __init__(self, nid: int, agent_type: int, rank: int):
        super().__init__(nid, Neuron.TYPE, rank)

    def save(self):
        return (self.uid, self.random_param_test)

    def update(self, data: bool):
        a = 0

    # def update(self, data: bool):
    #     """Updates the state of this agent when it is a ghost
    #     agent on some rank other than its local one.

    #     Args:
    #         data: the new agent state (received_rumor)
    #     """
    #     if not self.received_rumor and data:
    #         # only update if the received rumor state
    #         # has changed from false to true
    #         model.rumor_spreaders.append(self)
    #         self.received_rumor = data


# Network test
def create_rumor_agent(nid, agent_type, rank, **kwargs):
    return Neuron(nid, agent_type, rank)
    

agent_cache = {}


def restore_agent(agent_data: Tuple):
    """Creates an agent from the specified agent_data.

    This is used to re-create agents when they have moved from one MPI rank to another.
    The tuple returned by the agent's save() method is moved between ranks, and restore_agent
    is called for each tuple in order to create the agent on that rank. Here we also use
    a cache to cache any agents already created on this rank, and only update their state
    rather than creating from scratch.

    Args:
        agent_data: the data to create the agent from. This is the tuple returned from the agent's save() method
                    where the first element is the agent id tuple, and any remaining arguments encapsulate
                    agent state.
    """
    uid = agent_data[0]
    # 0 is id, 1 is type, 2 is rank
    if uid[1] == GutCell.TYPE:
        if uid in agent_cache:
            gc = agent_cache[uid]
        else:
            gc = GutCell(uid[0], uid[2])
            agent_cache[uid] = gc

        # restore the agent state from the agent_data tuple
        gc.random_param_test = agent_data[1]
        return gc
    elif uid[1] == Neuron.TYPE:
        return Neuron(uid[0], uid[1], uid[2])


@dataclass
class Counts:
    """Dataclass used by repast4py aggregate logging to record
    the number of gutCells after each tick.
    """
    gut_cells: int = 0
    neurons: int = 0


#OKE!


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
        self.space = space.SharedCSpace('space', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                        buffer_size=2, comm=comm, tree_threshold=100)
        self.context.add_projection(self.space)
        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        self.counts = Counts()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, params['counts_file'])

        world_size = comm.Get_size()

        local_bounds = self.space.get_local_bounds()
        total_gc_count = params['gut_cells.count']
        pp_gc_count = int(total_gc_count / world_size)
        if self.rank < total_gc_count % world_size:
            pp_gc_count += 1

        for i in range(pp_gc_count):
            gc = GutCell(i, self.rank)
            self.context.add(gc)
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(gc, x, y)


        #NetworkStuff:
        fpath = params['network_file']
        #self.context = ctx.SharedContext(comm)
        read_network(fpath, self.context, create_rumor_agent, restore_agent)
        self.net = self.context.get_projection('rumor_network')
        

    def at_end(self):
        self.data_set.close()

    def move(self, agent, x, y):
        self.space.move(agent, cpt(x, y))
        self.grid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))

    def step(self):
        tick = self.runner.schedule.tick
        self.log_counts(tick)
        self.context.synchronize(restore_agent)

    def run(self):
        self.runner.execute()

    def remove_agent(self, agent):
        self.context.remove(agent)

    def log_counts(self, tick):
        # Get the current number of zombies and humans and log
        num_agents = self.context.size([GutCell.TYPE, Neuron.TYPE])
        self.counts.gut_cells = num_agents[GutCell.TYPE]
        self.counts.neurons = num_agents[Neuron.TYPE]
        self.data_set.log(tick)

        # Do the cross-rank reduction manually and print the result
        if tick % 10 == 0:
            gc_count = np.zeros(1, dtype='int64')
            neuron_count = np.zeros(1, dtype='int64')
            self.comm.Reduce(np.array([self.counts.gut_cells], dtype='int64'), gc_count, op=MPI.SUM, root=0)
            self.comm.Reduce(np.array([self.counts.neurons], dtype='int64'), neuron_count, op=MPI.SUM, root=0)
            if (self.rank == 0):
                print("Tick: {}, gCells Count: {}, Neurons: {}".format(tick, gc_count[0], neuron_count[0]),
                      flush=True)


def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.run()


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)
