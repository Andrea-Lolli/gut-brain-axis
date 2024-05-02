import sys
import math
import networkx as nx
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
from repast4py.network import write_network, read_network
from repast4py import context as ctx
from typing import Dict, Tuple
from numba import int32, int64
from numba.experimental import jitclass
from repast4py.parameters import create_args_parser, init_params
from repast4py import core, random, space, schedule, logging, parameters
import repast4py
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType

"""

Malattie associate all'alfa-sinucleina (ad es. Morbo di Parkinson, demenza a corpi di Lewy):

    Tremore a riposo, specialmente nelle mani.
    Rigidità muscolare.
    Bradicinesia (lentezza nei movimenti).
    Problemi di equilibrio e coordinazione.
    Cambiamenti cognitivi, come problemi di memoria e di concentrazione.
    Alterazioni del sonno, inclusi problemi di sonno REM e sonno disturbato.
    Cambiamenti dell'umore, come depressione, ansia o irritabilità.
    Allucinazioni visive.
    Riduzione dell'olfatto (anosmia) o del gusto.

Malattie associate alla beta-amiloide (ad es. Malattia di Alzheimer):

    Perdita di memoria a breve termine, specialmente per gli eventi recenti.
    Difficoltà nell'esecuzione di compiti quotidiani, come la pianificazione delle attività.
    Confusione mentale o disorientamento negli spazi familiari.
    Problemi di linguaggio, come difficoltà a trovare le parole giuste.
    Difficoltà di concentrazione e di decisione.
    Cambiamenti dell'umore, come irritabilità o apatia.
    Problemi di sonno, inclusi problemi di insonnia o sonno disturbato.
    Riduzione delle capacità visuo-spaziali.
    Perdita di interesse per le attività sociali o hobbies.

Malattie associate alla proteina tau (ad es. Malattia di Alzheimer, demenze frontotemporali):

    Cambiamenti nella personalità e nel comportamento, come impulsività o disinibizione.
    Difficoltà nel mantenere le relazioni sociali o nell'empatia.
    Alterazioni dell'umore, come depressione o apatia.
    Problemi di linguaggio, come la parola errante o la difficoltà a comprendere il linguaggio.
    Cambiamenti nella memoria e nella cognizione, come la perdita di memoria episodica.
    Riduzione dell'inibizione comportamentale o dell'autoregolazione.
    Problemi di sonno, inclusi problemi di insonnia o sonno disturbato.
    Cambiamenti nell'alimentazione, come una maggiore o minore appetito.
    Riduzione delle capacità visuo-spaziali o della coordinazione motoria.

"""


model = None


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


@dataclass(slots=True)
class Stato:
    tick: int = 0
    eta : int = 0
    nutrienti_b: int = 0
    nutrienti_n: int = 0
    ratio_bn : float = 0.0
    prodotto_b: int = 0
    prodotto_n: int = 0
    citochine_gut: int = 0
    citochine_brain: int = 0
    #MHC_attivo: bool = False
    # da discutere
    alfa_sin : int = 0
    beta_a: int = 0
    tau: int = 0
    infiammazione_g : bool = False
    infiammazione_b: bool = False
    infiammazione_bc: bool = False
    err : bool = False
    def ratio(self) -> float:
        self.ratio_bn = (self.prodotto_b) / (self.prodotto_n)
        return self.ratio_bn


class Soglia(core.Agent):

    TYPE = 0

    def __init__(self, nid: int, agent_type: int, rank: int):
        super().__init__(nid, agent_type, rank)
        self.Env = Stato()
        self.flag = False
        

    def gather_local_b(self, env: Stato):
        self.Env.citochine_brain += env.citochine_brain
        #self.Env.MHC_attivo = env.MHC_attivo
        self.flag = True

    def gather_local_g(self, env: Stato):
        self.Env.citochine_gut = env.citochine_gut
        self.Env.nutrienti_b = env.nutrienti_b
        self.Env.nutrienti_n = env.nutrienti_n
        self.Env.prodotto_b = env.prodotto_b
        self.Env.prodotto_n = env.prodotto_n
        self.Env.ratio_bn = env.ratio()
        self.flag = True 
        
    
    def G_to_B(self):
        pass

    def B_to_G(self):
        pass

    def log(self):
        pass
    
    def save(self) -> Tuple:
        return (self.uid, self.Env, self.flag)
    
    def update(self, env, flag):
        if flag:
            self.Env = env
            self.flag = False  


def crea_nodo(nid, agent_type, rank, **kwargs):
    if agent_type == 0:
         return Soglia(nid, agent_type, rank)
    if agent_type == 3:
         return Neuro(nid, agent_type, rank)
    if agent_type == 4:
         return GliaTest(nid, agent_type, rank)

def restore_nodo(agent_data):
    uid = agent_data[0]
    if  uid[1] == 0:
            return Soglia(uid[0], uid[1], uid[2])
    if  uid[1] == 3:
            return Neuro(uid[0], uid[1], uid[2])
    if  uid[1] == 4:
            return GliaTest(uid[0], uid[1], uid[2])
    

class Batterio_Benefico(core.Agent):
    
    TYPE = 1

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=Batterio_Benefico.TYPE, rank=rank)
        self.pt = pt
        self.riserva = 10
        self.hp = 10
        self.stato="vivo"

    def assorbe(self, nutrimento):
        """assorbe nutrimento"""
        if nutrimento >= 10:
            self.riserva = self.riserva + 10
            nutrimento = nutrimento - 10
            return nutrimento
        else:
            return nutrimento

    def produci(self):
        """metabolita"""
        if self.riserva >= 5:
            self.riserva = self.riserva - 5
            return 5
        else:
            return 0

    def duplica(self) -> bool:
        """se stesso"""
        if self.riserva >= 10:
            self.riserva = self.riserva - 10
            return True
        else:
            return False

    def consuma(self):
        """per sopravvivere"""
        if self.hp < 1:
            self.stato = "morto"
        else:
            if self.riserva >=2:
                self.riserva = self.riserva - 2
            else:
                self.hp = self.hp - 2


    def save(self) -> Tuple:
        #print("SAVE 1")
        return (self.uid, self.pt.coordinates, self.riserva, self.hp, self.stato)
    
class Batterio_Nocivo(core.Agent):
    
    TYPE = 2

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=Batterio_Nocivo.TYPE, rank=rank)
        self.pt = pt
        self.riserva = 10
        self.hp = 10
        self.stato="vivo"

    def assorbe(self, nutrimento):
        """assorbe nutrimento"""
        if nutrimento >= 10:
            self.riserva = self.riserva + 10
            nutrimento = nutrimento - 10
            return nutrimento
        else:
            return nutrimento

    def produci(self):
        """metabolita"""
        if self.riserva >= 5:
            self.riserva = self.riserva - 5
            return 5
        else:
            return 0

    def duplica(self) -> bool:
        """se stesso"""
        if self.riserva >= 10:
            self.riserva = self.riserva - 10
            return True
        else:
            return False

    def consuma(self):
        """per sopravvivere"""
        if self.hp < 1:
            self.stato = "morto"
        else:
            if self.riserva >=2:
                self.riserva = self.riserva - 2
            else:
                self.hp = self.hp - 2
        

    def save(self) -> Tuple:
        return (self.uid, self.pt.coordinates, self.riserva, self.hp, self.stato)


class Neuro(core.Agent):

    TYPE = 3

    def __init__(self, nid: int, agent_type: int, rank: int):
        super().__init__(nid, agent_type, rank)
        self.eta = 0
        self.accumulo_tau = 0
        self.accumolo_beta = 0
        self.accumolo_alfa = 0
        self.patologia = 0
        self.flag = False
        self.rng = np.random.default_rng()
        self.stato = "non_compromesso"
    
    def step(self, stato: Stato):
        self.eta += 1
        stato.eta = self.eta
        # azioni interne
        self.check()
        self.prod()
        self.comunica()
        self.autofagia()
        stato.alfa_sin += self.prod_alfa()
        stato.beta_a += self.prod_beta()
        stato.tau += self.prod_tau()
        # info
        self.flag = True

    def prod(self):
        tripla = self.rng.integers(low=0, high=10, size=3)
        self.accumolo_alfa += tripla[0]
        self.accumolo_beta += tripla[1]
        self.accumulo_tau += tripla[2]
        self.patologia += max(tripla[0], tripla[1], tripla[2])

    def check(self):
        """qua è da decidere"""
        soglia = 100 /  self.eta
        if self.patologia > soglia * 10:
            self.stato = "compromesso"

    def prod_tau(self):
        return self.accumulo_tau
    
    def prod_beta(self):
        return self.accumolo_beta
    
    def prod_alfa(self):
        return self.accumolo_alfa
    
    def prod_patologia(self):
        return self.patologia
    
    def get_eta(self):
        return self.eta
    
    def citochine(self):
        pass

    def autofagia(self):
        tripla = self.rng.integers(low=0, high=10, size=3)
        if self.stato != "compromesso":
            self.accumolo_alfa -= tripla[0]
            self.accumolo_beta -= tripla[1]
            self.accumulo_tau -= tripla[2]
            self.patologia -= max(tripla[0], tripla[1], tripla[2])
        else:
            self.patologia += max(tripla[0], tripla[1], tripla[2])

    def comunica(self):
        count = 0
        for ngh in model.neural_net.graph.neighbors(self):
            count += 1
            if ngh.uid[1] == 3:
                self.to_neuro(ngh,count)
            if ngh.uid[1] == 4:
                self.to_glia(ngh,count)
        

    def to_glia(self, glia,count):
        self.patologia -= glia.fagocitosi(self.patologia/count)

    def to_neuro(self, neuro,count):
        neuro.riceve(neuro.patologia,count)

    def riceve(self, patologia,count):
        self.patologia += (patologia/count)

    def save(self) -> Tuple:
        return (self.uid,
        self.eta,
        self.accumulo_tau,
        self.accumolo_beta,
        self.accumolo_alfa,
        self.patologia,
        self.flag,
        self.stato)
    
    def update(self,eta,accumulo_tau,
        accumolo_beta,
        accumolo_alfa,
        patologia,
        flag,
        stato):   
        if flag:
            self.eta = eta
            self.accumulo_tau = accumulo_tau
            self.accumolo_alfa = accumolo_alfa
            self.accumolo_beta = accumolo_beta
            self.patologia = patologia
            self.stato = stato
            self.flag = False



class GliaTest(core.Agent):

    TYPE = 4

    def __init__(self, nid: int, agent_type: int, rank: int):
        super().__init__(nid, agent_type, rank)
        self.da_fagocitare = 0 # capacità di digerire elementi esterni.
        self.da_autofagocitare = 0 # capacità di diferire elementi inteni.
        self.citochine_recepite = 0 # influiscono sull'autofagia
        self.stato= "omeostasi"    
        self.flag = False

    def step(self, stato:Stato):
        
        if self.stato == "omeostasi":
            self.da_fagocitare = 0
            self.da_autofagocitare = 0

        if self.stato == "DAM1":
            self.da_autofagocitare = 0
            self.da_fagocitare -= (self.da_fagocitare*2)/3
        
        if self.stato == "DAM2":
            self.da_autofagocitare += stato.eta / 2
            self.da_fagocitare -= (self.da_fagocitare/3)
            stato.citochine_brain += stato.citochine_brain + 1
        

                   
    def fagocitosi(self, patogeno):
        self.da_fagocitare += patogeno
        return patogeno

    def save(self):
        return(self.uid,self.stato,self.da_autofagocitare, self.da_fagocitare, self.citochine_recepite, self.flag)
    
    def update(self, stato, da_a, da_f, cit, flag):
        if flag:
            self.stato=stato,
            self.da_autofagocitare = da_a,
            self.da_fagocitare = da_f,
            self.citochine_recepite=cit,
            self.flag = False


class Glia(core.Agent):
    """altri stati DAM-1 e DAM-2, con DAM-2 irreversibile"""
    """TREM 1 e TREM 2 sono le stesso recettore ma con una portata diversa in base allo stato"""
    TYPE = 4

    def __init__(self, nid: int, agent_type: int, rank: int):
        super().__init__(nid, agent_type, rank)
        self.da_fagocitare = 0 # capacità di digerire elementi esterni.

        self.da_autofagocitare = 0 # capacità di diferire elementi inteni.

        self.citochine_recepite = 0 # influiscono sull'autofagia

        self.stato= "omeostasi"    
        self.flag = False
    
    def step(self, stato: Stato):
        # se è in stato attivo dovrebbe rilasciare citochine
        #stato.citochine_brain += self.rilascio_citochine()
        #self.fagocitosi()
        #self.autofagocitosi(stato)
        pass
            
    def to_DAM_1(self):
        self.stato = "DAM1"
        
    def to_DAM_2(self):
        self.stato = "DAM2"
    
    def to_Homo(self):
        self.stato = "omeostati"

    
    def rilascio_citochine(self, n : Neuro):
        pass

    def rilascio_citochine(self, g: 'Glia'):
        pass

    def autofagocitosi(self):
        pass

    def fagocitosi(self):
        if self.stato == "omeostasi":
            self.to_DAM_1()
            self.da_fagocitare -= 5
        if self.stato == "DAM1":
            self.da_fagocitare += 5
            self.to_Homo()
        if self.stato == "DAM1" and self.da_fagocitare % 10 == 0 :
            self.stato = self.to_DAM_2()
        if self.stato == "DAM2":
            pass
        pass


    def save(self) -> Tuple:
        return (self.uid,
        self.da_fagocitare,
        self.da_autofagocitare,
        self.citochine_recepite,
        self.stato,    
        self.flag)
    
    def update(self,da_fagocitare,da_autofagocitare,citochine_recepite,stato,flag):
        """da completare"""
        if flag:
            self.da_fagocitare = da_fagocitare
            self.da_autofagocitare = da_autofagocitare
            self.citochine_recepite = citochine_recepite 
            self.stato = stato
            self.flag = False



class Model:

    def __init__(self, comm: MPI.Intracomm, params: Dict):

        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()


        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])

        self.tick = 0

        fpath = params['network_file']
        read_network(fpath, self.context, crea_nodo, restore_nodo)
        self.net = self.context.get_projection('int_net')

        fpath = params['neural_file']
        read_network(fpath, self.context, crea_nodo, restore_nodo)
        self.neural_net = self.context.get_projection('neural_net')

        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Single, buffer_size=0, comm=comm)
        self.context.add_projection(self.grid)
        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)
        
        self.count_b = 0
        self.count_n = 0
        rng = repast4py.random.default_rng
        
        self.nutrimento_benefico = params['nutrimento.benefico']
        self.nutrimento_nocivo = params['nutrimento.nocivo']

        for i in range(params['benefico.count']):
            # get a random x,y location in the grid
            pt = self.grid.get_random_local_pt(rng)
            # create and add the walker to the context
            b = Batterio_Benefico(i, self.rank, pt)
            self.context.add(b)
            self.grid.move(b, pt)
            self.count_b += 1
        for i in range(params['nocivo.count']):
            # get a random x,y location in the grid
            pt = self.grid.get_random_local_pt(rng)
            # create and add the walker to the context
            b = Batterio_Nocivo(i, self.rank, pt)
            self.context.add(b)
            self.grid.move(b, pt)
            self.count_n += 1




    def step00(self):
        #self.terminal()
        #self.context.synchronize(restore_nodo)
        if self.rank == 0:
            for i in self.context.agents(agent_type=0):
                print("agente", i)
                for ii in self.net.graph.neighbors(i):
                    print("vicini",ii)
            for i in self.context.agents(agent_type=3):
                print("agente",i)
                for ii in self.neural_net.graph.neighbors(i):
                    print("vicini",ii)
            for i in self.context.agents(agent_type=4):
                print("agente", i)
                for ii in self.neural_net.graph.neighbors(i):
                    print("vicini",ii)
        self.step_00()
            

    def step(self):
        self.tick += 1

        stato = Stato()
        stato.tick = self.tick

        for agent in self.context.agents(agent_type=0):
            if agent.uid[2] == self.rank:
                temp = self.gut_step(stato)
                agent.gather_local_g(temp)
                temp = self.brain_step(stato)
                agent.gather_local_b(temp)
                print(temp)

        self.context.synchronize(restore_nodo)

        for nodo in self.context.agents(agent_type=0):
                if nodo.uid[2] == self.rank:
                    for i in self.net.graph.neighbors(nodo):
                        nodo.Env.prodotto_b += i.Env.prodotto_b
                        nodo.Env.prodotto_n += i.Env.prodotto_n
                        nodo.Env.citochine_gut += i.Env.citochine_gut
                        nodo.Env.citochine_brain += i.Env.citochine_brain
                        nodo.Env.alfa_sin += i.Env.alfa_sin
                        nodo.Env.beta_a += i.Env.beta_a
                        nodo.Env.tau += i.Env.tau
        
        
                        



        #self.terminal()



    def start(self):
        self.runner.execute()
    


    def terminal(self):
        """alla fine va eliminato"""
        if self.rank == 0:
            print("benefico ",self.rank,self.nutrimento_benefico)
            print("novico " , self.rank,self.nutrimento_nocivo)
            for b in self.context.agents():
                if b.uid[1] == 1 or b.uid[1] == 2:
                    print("Rank {} agente {} {} {} {} {}".format(self.rank,b , b.hp, b.riserva, b.stato, b.pt))
                if b.uid[1] == 3 or b.uid[1] == 4:
                    print("Rank {} agente {} {} ".format(self.rank, b , b.stato))
                if b.uid[1] == 0:
                    print("Rank {} agente {} ".format(self.rank, b.uid))
            

    def brain_step(self, stato:Stato) -> Stato:
        for e in self.neural_net.graph:
            if e.uid[2] == self.rank and (e.uid[1] == Neuro.TYPE or e.uid[1] == GliaTest.TYPE):
                e.step(stato)
        return stato

    def gut_step(self,stato:Stato)-> Stato:
        local_b = self.grid.get_local_bounds()
        for b in self.context.agents(shuffle=True):
            if b.uid[1] == Batterio_Benefico.TYPE and b.stato=="vivo" :
                self.nutrimento_benefico = b.assorbe(self.nutrimento_benefico)                
                b.consuma()               
                stato.prodotto_b += b.produci()              
                if b.duplica():
                    flag = True
                    nghs = self.ngh_finder.find(b.pt.x, b.pt.y)
                    for ngh in nghs:                
                        at = dpt(0,0,0)
                        at._reset_from_array(ngh)
                        a = self.grid.get_agent(at) 
                        if a == None and at.x >= local_b.xmin and at.x < local_b.xmin + local_b.xextent and at.y >= local_b.ymin and at.y < local_b.ymin + local_b.yextent:
                            self.count_b += 1
                            new_b = Batterio_Benefico(self.count_b, self.rank, at)
                            self.context.add(new_b)
                            self.grid.move(new_b, at)                          
                            flag = False
                        if flag == False:
                            break  
            if b.uid[1] == Batterio_Nocivo.TYPE and b.stato=="vivo":
                self.nutrimento_nocivo = b.assorbe(self.nutrimento_nocivo)
                b.consuma()
                stato.prodotto_n += b.produci()
                if b.duplica():
                    flag = True
                    nghs = self.ngh_finder.find(b.pt.x, b.pt.y)
                    for ngh in nghs:                
                        at = dpt(0,0,0)
                        at._reset_from_array(ngh)
                        a = self.grid.get_agent(at) 
                        if a == None and at.x >= local_b.xmin and at.x < local_b.xmin + local_b.xextent and at.y >= local_b.ymin and at.y < local_b.ymin + local_b.yextent:
                            self.count_n += 1
                            new_b = Batterio_Nocivo(self.count_n, self.rank, at)
                            self.context.add(new_b)
                            self.grid.move(new_b, at)                       
                            flag = False
                        if flag == False:
                            break
        return stato    


def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)