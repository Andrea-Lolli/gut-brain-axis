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
    """recupera i vicini da un dato punto della griglia"""

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
    """

    Classe che sintetizza lo stato dei due sistemi connessi.
    Ipoteticamente un plausibile sistemi vascolare.
    tick, tick di sistema
    eta, età di un individuo predisposto verso una malattia degenerativa
    nutrienti_b, nutrienti per un plausibile batterio benefico
    nutrienti_n, nutrienti per un plausibile batterio nocivo
    ratio_bn, sintetizza il rapporto tra benefico e nocivo
    prodotto_n, metabolita nociva
    prodotto_b, metabolita benefico
    citochine_gut, livello delle citochine e determina l'infiammazione locale
    citochine_brain, livello delle citochine e determina l'infiammazione locale
    alfa, beta, tau, determinate il tipo di neurodegenerazione
    infiammazione_locale_, flag di riferimento

    """
    tick: int = 0
    eta : int = 0
    nutrienti_b: np.float16 = 0
    nutrienti_n: np.float16 = 0
    ratio_bn : np.float16 = 0.0
    prodotto_b: np.float16 = 0
    prodotto_n: np.float16 = 0
    citochine_gut: np.float16 = 0
    citochine_brain: np.float16 = 0
    alfa_sin : np.float16 = 0
    beta_a: np.float16 = 0
    tau: np.float16 = 0
    infiammazione_local_g : bool = False
    infiammazione_local_b: bool = False
    #infiammazione_local_bg: bool = False
    def ratio(self) -> np.float16:
        """
        
        rapporto tra i batteri, benefici e non, a livello locale

        """
        self.ratio_bn = (self.prodotto_n) / (self.prodotto_b+1)
        return self.ratio_bn
    def cit_gut(self) -> np.float16 :
        """
        
        aumento base della produzione di citochine a livello intestinale, proporzionale al rapporto tra i tipi di batteri
        
        """
        self.citochine_gut = np.random.default_rng().exponential(scale=1/self.eta) * self.ratio()
        return self.citochine_gut


class Soglia(core.Agent):

    """

    Agente che vigila sull'esecuzione dell'elaborazione.
    Punto di contatto tra due sistemi, definibile come interfaccia, regola il comportamento locale e sintetizza 
    lo stato dei due sistemi comunicanti
    Stato, @dataclass
    flag, update ghost

    """

    TYPE = 0

    def __init__(self, nid: int, agent_type: int, rank: int):
        super().__init__(nid, agent_type, rank)
        self.Env = Stato()
        self.flag = False
        

    def gather_local_b(self, env: Stato):
        """
        
        Aggiorna il punto di contatto del sistema "brain", tramite assegnamenti, al tempo t

        """
        self.Env.tick = env.tick
        self.Env.eta = env.eta
        self.Env.citochine_brain = env.citochine_brain
        self.Env.alfa_sin = env.alfa_sin
        self.Env.beta_a = env.beta_a
        self.Env.tau = env.tau
        self.flag = True

    def gather_local_g(self, env: Stato):
        """
        
        Aggiorna il punto di contatto del sistema "gut", tramite assegnamenti, al tempo t 

        """
        self.Env.citochine_gut = env.citochine_gut
        self.Env.nutrienti_b = env.nutrienti_b
        self.Env.nutrienti_n = env.nutrienti_n
        self.Env.prodotto_b = env.prodotto_b
        self.Env.prodotto_n = env.prodotto_n
        self.Env.ratio_bn = env.ratio()
        self.Env.citochine_gut = env.cit_gut()
        self.flag = True
    

    def comportamento_locale(self):
        """

        Componenete locale, combinazione di Soglia locale , sotto-sistema locale "brain" (una parte della neural net)
        e sotto-sistema local "gut" (una parte della griglia).
        Per comportamento locale si intende la gestione dei casi in cui una delle parti del comportamento locale risulti 
        sbilanciata verso uno stato infiammatorio. Saranno le quantità dei metaboliti benefici e nocivi a determinare l'intensità del 
        bilanciamento.

        """
        if (self.Env.infiammazione_local_g == False) and (self.Env.infiammazione_local_b == True):
            self.Env.citochine_brain -= (self.Env.prodotto_b * self.Env.citochine_brain)/ params['fattore.di.scala']
            self.Env.citochine_gut += (self.Env.prodotto_n * self.Env.citochine_brain)/ params['fattore.di.scala']
            
        if (self.Env.infiammazione_local_g == True ) and  (self.Env.infiammazione_local_b == False):
            self.Env.citochine_brain += (self.Env.prodotto_n * self.Env.citochine_gut)/ params['fattore.di.scala']
            self.Env.citochine_gut -= (self.Env.prodotto_b * self.Env.citochine_gut)/ params['fattore.di.scala']
            
            
    

    def log(self):
        pass
    
    def save(self) -> Tuple:
        return (self.uid, self.Env, self.flag)
    
    def update(self, env, flag):
        """
        
        Update per ghost

        """
        if flag:
            self.Env = env
            self.flag = False  


def crea_nodo(nid, agent_type, rank, **kwargs):
    """
    
    Autoesplicativo, repast4py

    """
    if agent_type == 0:
         return Soglia(nid, agent_type, rank)
    if agent_type == 3:
         return Neuro(nid, agent_type, rank)
    if agent_type == 4:
         return GliaTest(nid, agent_type, rank)

def restore_nodo(agent_data):
    """

    Autoesplicativo repast4py

    """
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

    """
    Agente che rappresenta il neurone.
    Produce, comunica e smaltisce.
    In natura tutte e tre le possibili proteine sono prodotte dal neurone, durante la sua vita. 
    Di conseguenza, c'è una possibilittà, proporzionale all'età di produrre una proteina patologica.
    """

    TYPE = 3

    def __init__(self, nid: int, agent_type: int, rank: int):
        super().__init__(nid, agent_type, rank)
        self.eta = 0
        self.accumulo_tau = 0 #definisce il tipi di DAM
        self.accumolo_beta = 0 #definisce il tipo di DAM 
        self.accumolo_alfa = 0 #definisce il tipo DAM
        self.patologia = 0 # combinazione degli effetti della patologia proteica e delle citochine
        self.flag = False
        self.stato = "non_compromesso"
    
    def step(self, stato: Stato):
        
        """
        Serie di azioni compiute dal neurone, a tick. 
        Produce elementi, proteine.
        Controlla il suo stato.
        Comunica con i propri vicini - una glia - ....
        """
        self.eta = stato.eta
        self.prod()
        self.check()
        self.comunica()
        self.autofagia()
        stato.alfa_sin += self.prod_alfa()
        stato.beta_a += self.prod_beta()
        stato.tau += self.prod_tau()
        self.flag = True

    def prod(self):
        tripla = np.random.normal(self.eta, 1, size=3)
        tripla = 1 / (1 + np.exp(-0.5 * (tripla - self.eta)))
        tripla = abs(tripla)
        self.accumolo_alfa += tripla[0]
        self.accumolo_beta += tripla[1]
        self.accumulo_tau += tripla[2]
        self.patologia += max(tripla[0],tripla[1],tripla[2]) 

    def check(self):
        soglia = self.accumolo_alfa + self.accumolo_beta + self.accumulo_tau
        if self.patologia > soglia and self.eta >= params['init_degenerazione'] :
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

    def autofagia(self):
        tripla = np.random.normal(self.eta, 1, size=3)
        tripla = 1 / (1 + np.exp(-0.5 * (tripla - self.eta)))
        tripla = abs(tripla)
        if self.stato != "compromesso":
            self.accumolo_alfa -= tripla[0]
            self.accumolo_beta -= tripla[1]
            self.accumulo_tau -= tripla[2]
            self.patologia -= min(tripla[0],tripla[1],tripla[2])
        else:
            self.patologia += max(tripla[0], tripla[1], tripla[2])

    def comunica(self):
        """
        comunica solo con il glia associato o ai glia associati
        """
        for ngh in model.neural_net.graph.neighbors(self):
            if ngh.uid[1] == 4 :
                self.to_glia(ngh)
                # prelievo da ghost
            if ngh.uid[1] == 3:
                self.accumolo_alfa += ngh.prod_alfa() / params["fattore.di.scala"]
                self.accumolo_beta += ngh.prod_beta() / params['fattore.di.scala']
                self.accumulo_tau += ngh.prod_tau() /  params['fattore.di.scala']
                self.patologia += ngh.prod_patologia() / params['fattore.di.scala']


        
    def to_glia(self, glia):
        to_delete = self.patologia / self.eta
        glia.fagocitosi(to_delete)
        self.patologia -= abs(to_delete)


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
        """

        Le citochine cumolate per i vari rank vengono redistribuite e presen in esame per verificare
        lo stato interno del microglia. 
        Iinfluiscono sulle citochine recepite, all'interno del cervello anche le citochine provenienti dall'intestino.
        Tre stati: 
        In omeostasi rimuove ogni elemento passato azzerando il proprio contatore.
        Il passaggio allo stato successivo mette in relazione le citochine recepite, la quantità non ancora fagocitata e gli scarti interni.
        In DAM1 (Disease-Associated Microglia) primo stato delle malattie neurodegenerative, il glia, può rimuovere una parte 
        della "patologia" esterna e accumulare naturalmente le proprie scorie.
        In DAM2, il processo biologico spinge verso una situazione di infiammazione cronica.

        """

        self.citochine_recepite += stato.citochine_brain
        self.citochine_recepite += stato.citochine_gut
        
        if self.stato == "omeostasi":
            self.da_fagocitare = 0
            self.da_autofagocitare = 0
            self.stato = "DAM1"
            self.flag = True
            return stato
        
        if self.stato == "DAM1":
            self.da_autofagocitare = np.random.default_rng().exponential(scale=1/ stato.eta) # giusto
            self.da_fagocitare -= (self.da_fagocitare) / stato.eta

        if stato.infiammazione_local_b == True:
            self.stato = "DAM2"
        
        if self.stato == "DAM1" and  (self.citochine_recepite * self.da_fagocitare * self.da_autofagocitare >= params['soglia_citochine']) :
            self.stato = "DAM2"
            stato.infiammazione_local_b = True
            self.flag = True
            return stato    
        
        # ipotesti modello. NB è diversa dall'ipotesi di modello.
        if self.stato == "DAM2" and (self.citochine_recepite * self.da_fagocitare * self.da_autofagocitare >= params['soglia_citochine'])  :
            self.da_autofagocitare += np.random.default_rng().exponential(scale=1/ stato.eta) * stato.citochine_brain
            self.da_fagocitare -= (self.da_fagocitare/3)
            stato.citochine_brain += self.da_autofagocitare
            stato.infiammazione_local_b = True
            self.flag = True
            return stato




                   
    def fagocitosi(self, patogeno):
        self.da_fagocitare += patogeno 
        

    def save(self):
        return(self.uid,self.stato,self.da_autofagocitare, self.da_fagocitare, self.citochine_recepite, self.flag)
    
    def update(self, stato, da_a, da_f, cit, flag):
        if flag:
            self.stato=stato,
            self.da_autofagocitare = da_a,
            self.da_fagocitare = da_f,
            self.citochine_recepite = cit,
            self.flag = False



class Model:

    def __init__(self, comm: MPI.Intracomm, params: Dict):

        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()


        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule.schedule_repeating_event(1,2, self.clear)
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
        

        for i in range(params['benefico.count']):
            pt = self.grid.get_random_local_pt(rng)
            b = Batterio_Benefico(i, self.rank, pt)
            self.context.add(b)
            self.grid.move(b, pt)
            self.count_b += 1
        for i in range(params['nocivo.count']):
            pt = self.grid.get_random_local_pt(rng)
            b = Batterio_Nocivo(i, self.rank, pt)
            self.context.add(b)
            self.grid.move(b, pt)
            self.count_n += 1


    def step(self):
        self.tick += 1

        stato = Stato()
        stato.eta = self.tick
        stato.tick = self.tick

        stato.nutrienti_b = params['nutrimento.benefico'] / stato.eta
        stato.nutrienti_n = params['nutrimento.nocivo'] / stato.eta


        for agent in self.context.agents(agent_type=0):

            if agent.uid[2] == self.rank:
                
                stato.citochine_brain = (agent.Env.citochine_brain / self.net.graph.size())
                stato.citochine_gut = (agent.Env.citochine_gut / self.net.graph.size())

                temp = self.gut_step(stato)
                agent.gather_local_g(temp)
                temp = self.brain_step(stato)
                agent.gather_local_b(temp)

        #self.terminal()
        self.context.synchronize(lambda x : x)
        #sself.terminal()
        
        for nodo in self.net.graph:
                
                if nodo.uid[2] == self.rank and nodo.uid[1] == Soglia.TYPE:

                    for i in self.net.graph.neighbors(nodo):

                        nodo.Env.prodotto_b += i.Env.prodotto_b
                        nodo.Env.prodotto_n += i.Env.prodotto_n

                        nodo.Env.citochine_gut += i.Env.citochine_gut 
                        nodo.Env.citochine_brain += i.Env.citochine_brain 

                        nodo.Env.alfa_sin += i.Env.alfa_sin
                        nodo.Env.beta_a += i.Env.beta_a
                        nodo.Env.tau += i.Env.tau


                    
                    nodo.Env.citochine_brain = nodo.Env.citochine_brain / (nodo.Env.citochine_brain + 1)
                    nodo.Env.citochine_gut = nodo.Env.citochine_gut / (nodo.Env.citochine_gut + 1)
                    nodo.Env.infiammazione_local_b = stato.infiammazione_local_b
                    nodo.Env.infiammazione_local_g = stato.infiammazione_local_g
                    #nodo.Env.infiammazione_local_bg = stato.infiammazione_local_bg

        
        self.context.synchronize(lambda x : x)

        for nodo in self.net.graph:
                
            if nodo.uid[2] == self.rank and nodo.uid[1] == Soglia.TYPE:

                nodo.comportamento_locale()
        
    
            
        self.terminal()
        #fine metodo



    def start(self):
        self.runner.execute()
    
    def clear(self):
        roba_da_matti = []
        for agent in self.context.agents():
            if agent.uid[2] == self.rank and (agent.uid[1] == Batterio_Benefico.TYPE or agent.uid[1] == Batterio_Nocivo.TYPE) and agent.stato == "morto":
                roba_da_matti.append(agent)   
                
        for i in roba_da_matti:
            self.context.remove(i)

    

    def terminal(self):
        """alla fine va eliminato"""
        if self.rank == 0:
            for b in self.context.agents():
                if b.uid[1] == 1 or b.uid[1] == 2:
                    print("Rank {} agente {} {} {} {} {}".format(self.rank,b , b.hp, b.riserva, b.stato, b.pt))
                if b.uid[1] == 3:
                     print("Rank {} agente {} {} {} {} {} {} patologia : {}".format(self.rank, b , b.eta, b.stato, b.accumolo_alfa, b.accumolo_beta, b.accumulo_tau, b.patologia))
                     for bb in self.neural_net.graph.neighbors(b):
                        if bb.uid[1] == 3:
                            print("Rank {} agente {} {} {} {} {} {} patologia : {}".format(self.rank, bb , bb.eta, bb.stato, bb.accumolo_alfa, bb.accumolo_beta, bb.accumulo_tau, bb.patologia))
                if b.uid[1] == 4:
                    print("Rank {} agente {} {} {} {} recepite: {}".format(self.rank, b , b.stato, b.da_autofagocitare, b.da_fagocitare, b.citochine_recepite))
                    for bb in self.neural_net.graph.neighbors(b):
                        print("Rank {} agente {} {}".format(self.rank,bb, bb.stato))
                if b.uid[1] == 0:
                    print("Rank {} agente {} {}".format(self.rank, b.uid, b.Env))
                    for bb in self.net.graph.neighbors(b):
                        print("Rank {} agente {} {}".format(self.rank, bb.uid, bb.Env))
        
        
            

    def brain_step(self, stato:Stato) -> Stato:
        for e in self.neural_net.graph:
            if e.uid[2] == self.rank and (e.uid[1] == Neuro.TYPE or e.uid[1] == GliaTest.TYPE):
                e.step(stato)
        return stato

    def gut_step(self,stato:Stato)-> Stato:
        local_b = self.grid.get_local_bounds()
        for b in self.context.agents(shuffle=True):
            if b.uid[1] == Batterio_Benefico.TYPE and b.stato=="vivo" :
                #self.nutrimento_benefico = b.assorbe(self.nutrimento_benefico)                
                stato.nutrienti_b = b.assorbe(stato.nutrienti_b)
                b.consuma()               
                stato.prodotto_b += b.produci() #accumula              
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
                #self.nutrimento_nocivo = b.assorbe(self.nutrimento_nocivo)
                stato.nutrienti_n= b.assorbe(stato.nutrienti_n)
                b.consuma()
                stato.prodotto_n += b.produci() #accumula
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
        
        if stato.ratio() * stato.citochine_gut> 1:
            stato.infiammazione_local_g = True
        
        # penalità infiammazione
        if stato.infiammazione_local_g == True:
            stato.nutrienti_b -= stato.nutrienti_b * stato.cit_gut()
            stato.nutrienti_n += stato.nutrienti_n * stato.cit_gut()

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