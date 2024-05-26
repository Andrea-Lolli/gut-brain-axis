import sys
import math
import networkx as nx
import os
import random as rn
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


def crea_nodo(nid, agent_type, rank, **kwargs):
    """
    Autoesplicativo, repast4py
    """
    if agent_type == Soglia.TYPE:
         return Soglia(nid, agent_type, rank)
    if agent_type == Neuro.TYPE:
         return Neuro(nid, agent_type, rank)
    if agent_type == GliaTest.TYPE:
         return GliaTest(nid, agent_type, rank)


def restore_nodo(agent_data):
    """
    Autoesplicativo repast4py
    """
    uid = agent_data[0]
    if  uid[1] == Soglia.TYPE:
            return Soglia(uid[0], uid[1], uid[2])
    if  uid[1] == Neuro.TYPE:
            return Neuro(uid[0], uid[1], uid[2])
    if  uid[1] == GliaTest.TYPE:
            return GliaTest(uid[0], uid[1], uid[2])
    

def logistica_gaussiana(x, k):
    return 1 / (1 + np.exp(-k * x**2))


@dataclass
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
    infiammazione_locale_, flag di riferimento
    """

    eta : int = 0
    nutrienti_b: float = 0.0
    nutrienti_n: float  = 0.0
    prodotto_b: float  = 0.0
    prodotto_n: float  = 0.0
    citochine_gut: float  = 0.0
    citochine_brain: float  = 0.0
    infiammazione_local_g : bool = False
    infiammazione_local_b: bool = False
    # Test finale feedback
    init_degenerazione: float = 0.0

    def ratio_nb(self) -> float:
        if self.prodotto_b != 0:
            return self.prodotto_n / self.prodotto_b
        if self.prodotto_b == 0 or self.prodotto_n == 0 :
            return 0.5
    
    def ratio_bn(self)-> float:
        if self.prodotto_n != 0:
            return self.prodotto_b / self.prodotto_n 
        if self.prodotto_b == 0 or self.prodotto_n == 0 :
            return 0.5

    
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

    def save(self) -> Tuple:
        return (self.uid, self.Env, self.flag)
    
    def update(self, env, flag):
        """
        Update per ghost
        """
        if flag:
            self.Env = env
            self.flag = False

    def gather_local_b(self, env: Stato):
        """
        Aggiorna il punto di contatto del sistema "brain", tramite assegnamenti, al tempo t
        """
        self.Env.init_degenerazione = env.init_degenerazione
        self.Env.citochine_brain = env.citochine_brain
        self.Env.infiammazione_local_b = env.infiammazione_local_b
        self.flag = True

    def gather_local_g(self, env: Stato):
        """
        Aggiorna il punto di contatto del sistema "gut", tramite assegnamenti, al tempo t 
        """
        self.Env.eta = env.eta
        self.Env.nutrienti_b = env.nutrienti_b
        self.Env.nutrienti_n = env.nutrienti_n
        self.Env.prodotto_b = env.prodotto_b
        self.Env.prodotto_n = env.prodotto_n
        self.Env.citochine_gut = env.citochine_gut
        self.Env.infiammazione_local_g = env.infiammazione_local_g
        self.flag = True
        
    def comportamento_locale(self):
        """
        Componenete locale, combinazione di Soglia locale , sotto-sistema locale "brain" (una parte della neural net)
        e sotto-sistema local "gut" (una parte della griglia).
        Per comportamento locale si intende la gestione dei casi in cui una delle parti del comportamento locale risulti 
        sbilanciata verso uno stato infiammatorio. Saranno le quantità dei metaboliti benefici e nocivi a determinare l'intensità del 
        bilanciamento.
        """
        if (self.Env.infiammazione_local_g == False) and (self.Env.infiammazione_local_b == True) and self.Env.citochine_brain > 0:
            self.Env.citochine_gut += abs((self.Env.ratio_nb() * self.Env.citochine_brain) /  abs((self.Env.ratio_nb() * self.Env.citochine_brain)+1 ))
            self.Env.citochine_brain -= abs((self.Env.ratio_nb() * self.Env.citochine_brain) /  abs((self.Env.ratio_nb() * self.Env.citochine_brain)+1 ))

        if (self.Env.infiammazione_local_g == True ) and  (self.Env.infiammazione_local_b == False) and self.Env.citochine_gut > 0:
            self.Env.citochine_brain += abs((self.Env.ratio_nb() * self.Env.citochine_gut) /  abs((self.Env.ratio_nb() * self.Env.citochine_gut)+1 ))
            self.Env.citochine_gut -= abs((self.Env.ratio_nb() * self.Env.citochine_gut) /  abs((self.Env.ratio_nb() * self.Env.citochine_gut)+1 )) 

        if (self.Env.infiammazione_local_b == True and self.Env.infiammazione_local_b == True) or (self.Env.infiammazione_local_b == False and self.Env.infiammazione_local_b == False):
            self.Env.citochine_brain = (self.Env.citochine_brain + self.Env.citochine_gut)/2
            self.Env.citochine_gut = (self.Env.citochine_brain + self.Env.citochine_gut)/2

        self.flag =  True  
    
    def share(self):
        size = MPI.COMM_WORLD.Get_size()
        size_g = model.net.graph.size()

        conta_tb = 0
        conta_fb = 0
        conta_tg = 0
        conta_fg = 0
        cit_b  =[]
        cit_g = []
        mb = []
        mn = []
        for nodo in model.net.graph.neighbors(self):

            cit_b.append(nodo.Env.citochine_brain)
            cit_g.append(nodo.Env.citochine_gut)
            mb.append(nodo.Env.prodotto_b)
            mn.append(nodo.Env.prodotto_n)

            if nodo.Env.infiammazione_local_b == True:
                conta_tb +=1
            else:
                conta_fb +=1
                
            if nodo.Env.infiammazione_local_g == True:
                conta_tg += 1
            else:
                conta_fg += 1
        
        if conta_tb > conta_fb:
            self.Env.infiammazione_local_b = True
            self.Env.citochine_brain = abs(max(cit_b)) 
        else:
            self.Env.infiammazione_local_b = False
            self.Env.citochine_brain = abs(min(cit_b)) 
        
        if conta_tg > conta_fg:
            self.Env.infiammazione_local_g = True
            self.Env.citochine_gut = abs(max(cit_g))
        else:
            self.Env.infiammazione_local_g = False
            self.Env.citochine_gut = abs(min(cit_g))
        
        # TODO Controllo l'andamento dei prodotti e nutrienti
        if self.Env.prodotto_b < max(mb):
            self.Env.prodotto_b = max(mb)/size
            self.Env.nutrienti_b += abs(min(mb)/size)
        if self.Env.prodotto_n < max(mn):
            self.Env.prodotto_n = max(mn)/size
            self.Env.nutrienti_n += abs(min(mn)/size)

        decay_factor = 0.98 # Fattore di decadimento, può essere regolato
        self.Env.prodotto_b *= decay_factor
        self.Env.prodotto_n *= decay_factor

        self.Env.init_degenerazione -= (self.Env.citochine_brain + self.Env.citochine_gut) / (size * size_g * 2)

        self.flag = True 


class Batterio_Benefico(core.Agent):
    """
    Agente che rappresenta un batterio benefico.
    Si riproduce all'interno dell'intestino se ha risorse necessarie per farlo, 
    il sistema immunitario è più efficiente quando questi batteri sono presenti nella flora intestinale in maggiori quantità
    """
    TYPE = 1

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=Batterio_Benefico.TYPE, rank=rank)
        self.pt = pt
        self.riserva = 10
        self.hp = 10
        self.stato="vivo"

    def assorbe(self, nutrimento, stato:Stato):
        """assorbe nutrimento"""
        if stato.infiammazione_local_g == False and nutrimento > 0.0:
            self.riserva += 10
            nutrimento -=  10 * stato.ratio_bn()
            return nutrimento
        if stato.infiammazione_local_g == True and nutrimento > 0.0:
            self.riserva += 10
            nutrimento -=  10
            return nutrimento
        if nutrimento <= 0:
            nutrimento = 0.0
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
        random_number = rn.random()
        if self.riserva >= 10 and random_number < 0.6:
            self.riserva = self.riserva - 10
            return True
        else:
            return False

    def consuma(self):
        """per sopravvivere"""
        if self.hp < 1:
            self.stato = "morto"
        else:
            if self.riserva >=3:
                self.riserva = self.riserva - 3
            else:
                self.hp = self.hp - 3

    def save(self) -> Tuple:
        return (self.uid, self.pt.coordinates, self.riserva, self.hp, self.stato)
    
class Batterio_Nocivo(core.Agent):
    """
    Agente che rappresenta un batterio "nocivo".
    Si riproduce all'interno dell'intestino se ha risorse necessarie per farlo, 
    il sistema immunitario viene inibito quando questi batteri sono presenti nella flora intestinale in maggiori quantità
    """
    TYPE = 2

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=Batterio_Nocivo.TYPE, rank=rank)
        self.pt = pt
        self.riserva = 10
        self.hp = 10
        self.stato="vivo"

    def assorbe(self, nutrimento, stato:Stato):
        """assorbe nutrimento"""
        if stato.infiammazione_local_g == True and nutrimento > 0.0:
            self.riserva += 10
            nutrimento -=  10 * stato.ratio_nb()
            return nutrimento
        
        if stato.infiammazione_local_g == False and nutrimento > 0.0:
            self.riserva += 10
            nutrimento -=  10
            return nutrimento
          
        if nutrimento <= 0:
            nutrimento = 0.0
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
        random_number = rn.random()
        if self.riserva >= 10 and random_number < 0.6:
            self.riserva = self.riserva - 10
            return True
        else:
            return False

    def consuma(self):
        """per sopravvivere"""
        if self.hp < 1:
            self.stato = "morto"
        else:
            if self.riserva >=3:
                self.riserva = self.riserva - 3
            else:
                self.hp = self.hp - 3     

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
        self.patologia = 0.0 # Combinazione degli effetti della patologia proteica e delle citochine
        self.flag = False
        self.stato = "non_compromesso"
    
    def save(self) -> Tuple:
        return (self.uid,
        self.patologia,
        self.flag,
        self.stato
        )
    
    def update(self,
        patologia,
        flag,
        stato):   
        if flag:
            self.patologia = patologia
            self.stato = stato
            self.flag = False

    def step(self, stato: Stato): 
        """
        Serie di azioni compiute dal neurone, a tick. 
        """
        self.prod(stato)
        self.comunica() # Rispetto al modello
        self.autofagia(stato)
        self.check(stato)
        self.flag= True
    
    def prod(self, stato:Stato):
        size = MPI.COMM_WORLD.Get_size()
        temp = abs(np.random.normal(stato.eta/size , stato.eta/size ))
        temp = abs(logistica_gaussiana(temp, 0.0005))
        self.patologia += temp/2 # Il massimo incremento è 0.5

    def autofagia(self, stato:Stato):
        size = MPI.COMM_WORLD.Get_size()
        tripla = []
        tripla = abs(np.random.normal(stato.eta/size,stato.eta/size, size=3))
        tripla = logistica_gaussiana(tripla,0.0005)
        if self.stato != "compromesso" and stato.infiammazione_local_g == False:
            self.patologia -= abs(min(tripla[0],tripla[1],tripla[2])) / size      
        else:
            self.patologia += abs(max(tripla[0],tripla[1],tripla[2])) / size

    def comunica(self):
        """
        comunica solo con il glia associato o ai glia associati
        """
        size = MPI.COMM_WORLD.Get_size()
        for ngh in model.neural_net.graph.neighbors(self):
            if ngh.uid[1] == GliaTest.TYPE :
                self.to_glia(ngh)
                # Prelievo da ghost
            if ngh.uid[1] == Neuro.TYPE:
                self.patologia += abs(ngh.patologia / (ngh.patologia +1) ) /size
        
    def to_glia(self, glia):
        # Spedisce al glia una certà quantità e questo valore diminuisce in base all'eta.
        # Per ora rilascia un 1/10 di malattia che poi il glia andrà a consumare
        size = MPI.COMM_WORLD.Get_size()
        to_delete = self.patologia / size 
        glia.fagocitosi(to_delete) 
        self.patologia -= abs(to_delete)

    def check(self,stato:Stato):
        """
        devo combinare le azioni delle citochine del cervello per determinare
        lo stato di salute del neurone associato
        """
        if stato.infiammazione_local_b == True and stato.eta >= stato.init_degenerazione: 
            self.stato = "compromesso"

class GliaTest(core.Agent):
    """
    Agente che rappresenta la glia, svolge funzioni essenziali per il sistema immunitario... TODO
    """
    TYPE = 4

    def __init__(self, nid: int, agent_type: int, rank: int):
        super().__init__(nid, agent_type, rank)
        self.da_fagocitare = 0 # Capacità di digerire elementi esterni.
        self.da_autofagocitare = 0 # Capacità di diferire elementi inteni.
        self.citochine_recepite = 0 # Influiscono sull'autofagia
        self.stato= "omeostasi"    
        self.flag = False
    
    def save(self):
        return(self.uid,self.stato,self.da_autofagocitare, self.da_fagocitare, self.citochine_recepite, self.flag)
    
    def update(self, stato, da_a, da_f, cit, flag):
        if flag:
            self.stato=stato,
            self.da_autofagocitare = da_a,
            self.da_fagocitare = da_f,
            self.citochine_recepite = cit,
            self.flag = False
    
    def fagocitosi(self, patogeno):
        self.da_fagocitare += patogeno 

    def step(self, stato:Stato)-> Stato:
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
        # Le citochine sono globali
        self.citochine_recepite = stato.citochine_brain

        p = self.citochine_recepite * self.da_fagocitare * self.da_autofagocitare * stato.prodotto_b
        n = self.citochine_recepite * self.da_fagocitare * self.da_autofagocitare * stato.prodotto_n
        
        if stato.infiammazione_local_b == False:

            if self.stato == "omeostasi":
                # Oppure localmente non sono infiammato
                self.da_fagocitare = self.da_fagocitare / model.neural_net.graph.size()
                self.da_autofagocitare += logistica_gaussiana(self.da_fagocitare, 0.1) /  model.neural_net.graph.size()
                self.stato = "DAM1"
                stato.citochine_brain += abs(self.da_autofagocitare/stato.eta * self.da_fagocitare/stato.eta) 
                stato.infiammazione_local_b = False
                self.flag = True
                return
            
            # Ipotesti modello. NB è diversa dall'ipotesi di modello.
            
            if self.stato == "DAM1" and stato.eta >= stato.init_degenerazione:
                stato.citochine_brain += abs(self.da_autofagocitare/stato.eta * self.da_fagocitare/stato.eta) #((self.da_autofagocitare * self.da_fagocitare) +1)
                stato.infiammazione_local_b = True
                self.da_autofagocitare += self.citochine_recepite / (self.citochine_recepite + 1) 
                self.da_fagocitare -= self.da_fagocitare / (self.da_fagocitare + 1)
                self.stato = "DAM2"
                self.flag = True
                return

            if self.stato == "DAM1" :
                # L'autofagia dipende dalle citochine? Si
                stato.citochine_brain += logistica_gaussiana(self.da_fagocitare, 0.1) /  model.neural_net.graph.size()
                self.da_autofagocitare += self.citochine_recepite / (self.citochine_recepite + 1) 
                self.da_fagocitare -= self.da_fagocitare / (self.da_fagocitare + 1)
                stato.infiammazione_local_b = False
                self.stato = "omeostasi"
                return
            
            if self.stato == "DAM2" and  (abs(n) > abs(p))  and stato.eta >= stato.init_degenerazione:
                stato.prodotto_n -= (stato.prodotto_n -stato.prodotto_b)
                if stato.prodotto_n <= 0 : 
                    stato.prodotto_n = 0
                stato.citochine_brain += abs((self.da_autofagocitare/stato.eta) * (self.da_fagocitare/stato.eta) * (stato.citochine_brain/stato.eta) )  #((self.da_autofagocitare * self.da_fagocitare * stato.citochine_brain)+1)
                stato.infiammazione_local_b = True
                return
            
            if self.stato == "DAM2" and (abs(p) > abs(n)) and stato.eta >= stato.init_degenerazione:
                self.stato = "DAM1"
                stato.infiammazione_local_b = False
                stato.prodotto_b -= (stato.prodotto_b -stato.prodotto_n)
                if stato.prodotto_n <= 0 : 
                    stato.prodotto_n = 0
                stato.citochine_brain -= stato.citochine_brain / 2
                return
 
        if stato.infiammazione_local_b == True:

            if self.stato == "DAM2" and  (abs(n) > abs(p)) and stato.eta >= stato.init_degenerazione:
                stato.prodotto_n -= (stato.prodotto_n -stato.prodotto_b)
                if stato.prodotto_n <= 0 : 
                    stato.prodotto_n = 0
                stato.citochine_brain += abs((self.da_autofagocitare/stato.eta) * (self.da_fagocitare/stato.eta) * (stato.citochine_brain/stato.eta)) #((self.da_autofagocitare * self.da_fagocitare * stato.citochine_brain)+1)
                stato.infiammazione_local_b = True
                return
            
            if self.stato == "DAM2" and  (abs(p) > abs(n)):
                stato.prodotto_b -= (stato.prodotto_b -stato.prodotto_n)
                if stato.prodotto_n <= 0 : 
                    stato.prodotto_n = 0
                self.stato == "DAM1"
                stato.citochine_brain -= stato.citochine_brain / 2
                stato.infiammazione_local_b = False
                return


class Model:

    def __init__(self, comm: MPI.Intracomm, params: Dict, filepath, net_file_name):

        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()
        self.filepath = filepath

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_end_event(self.at_end)
        self.runner.schedule.schedule_repeating_event(1,2, self.clear)
        self.runner.schedule_stop(params['stop.at'])
        self.tick = 0

        fpath = params['network_file']
        read_network(fpath, self.context, crea_nodo, restore_nodo)
        self.net = self.context.get_projection('int_net') # TODO sarebbe da parametrizzare 

        fpath = params['neural_file']
        read_network(fpath, self.context, crea_nodo, restore_nodo)
        self.neural_net = self.context.get_projection(net_file_name) 

        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Single, buffer_size=0, comm=comm)
        self.context.add_projection(self.grid)

        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)
        self.count_b = 0
        self.count_n = 0
        local_b = self.grid.get_local_bounds()

        # Aggiungo i batteri ad ogni rank in posizioni randomiche
        for i in range(params['benefico.count']):
            x = rn.randint(local_b.xmin, local_b.xmin + local_b.xextent -1)
            y = rn.randint(local_b.ymin, local_b.ymin + local_b.yextent -1)
            pt = dpt(x,y)
            b = Batterio_Benefico(i, self.rank, pt)
            self.context.add(b)
            self.grid.move(b, pt)
            self.count_b += 1
        for i in range(params['nocivo.count']):
            x = rn.randint(local_b.xmin, local_b.xmin + local_b.xextent -1)
            y = rn.randint(local_b.ymin, local_b.ymin + local_b.yextent -1)
            pt = dpt(x,y)
            b = Batterio_Nocivo(i, self.rank, pt)
            self.context.add(b)
            self.grid.move(b, pt)
            self.count_n += 1
        self.log_agents() # Log della configurazione iniziale
            
            
    def step(self):
        
        self.tick += 1

        # Nuovo stato locale
        stato = Stato()
        stato.eta = self.tick

        # Inizializza i parametri al primo tick
        if self.tick == 1:
            stato.nutrienti_b = params['nutrimento.benefico']
            stato.nutrienti_n = params['nutrimento.nocivo']
            stato.init_degenerazione = params['init.degenerazione']

        soglia_locale = self.context.agent((self.rank+1, 0, self.rank)) # La soglia che gestisce il rank

        # Cambio alimentazione al tick 100 (aggiunti nutrienti benefici) 
        # TODO sarebbe meglio metterlo altrove
        if self.tick == 100:
            soglia_locale.Env.nutrienti_b += 4000
            soglia_locale.Env.nutrienti_n += 1000

        # Carico tick precedente
        stato.citochine_brain = soglia_locale.Env.citochine_brain
        stato.citochine_gut = soglia_locale.Env.citochine_gut

        stato.prodotto_b = soglia_locale.Env.prodotto_b 
        stato.prodotto_n = soglia_locale.Env.prodotto_n 

        if self.tick != 1:
            stato.nutrienti_b = soglia_locale.Env.nutrienti_b 
            stato.nutrienti_n = soglia_locale.Env.nutrienti_n
            stato.init_degenerazione = soglia_locale.Env.init_degenerazione

        stato.infiammazione_local_g = soglia_locale.Env.infiammazione_local_g
        stato.infiammazione_local_b = soglia_locale.Env.infiammazione_local_b 
        
        # Passo lo stato all'intestino
        self.gut_step(stato)

        # Recupero le informazioni salvandole nella soglia
        soglia_locale.gather_local_g(stato)
        
        # Passo lo stato al cervello
        self.brain_step(stato)

        # Recupero le informazioni salvandole nella soglia
        soglia_locale.gather_local_b(stato)

        # Teoricamente sincronizza solo i ghost delle network 
        # quindi questo agente soglia vedrà i sui vicini connessi aggiornati 
        self.context.synchronize(lambda x : x)

        # Comportamento locale
        soglia_locale.comportamento_locale()

        # Sincronizzo nuovamente per aggiornare i ghost con i nuovi valori
        self.context.synchronize(lambda x : x) 
                
        soglia_locale.share()

        # Sincronizzo e vado oltre
        self.context.synchronize(lambda x : x)

        # Pulisco i batteri
        self.clear()
        self.log_agents()


    def clear(self):
        roba_da_matti = []
        for agent in self.context.agents():
            if agent.uid[2] == self.rank and (agent.uid[1] == Batterio_Benefico.TYPE or agent.uid[1] == Batterio_Nocivo.TYPE) and agent.stato == "morto":
                roba_da_matti.append(agent)   
                
        for i in roba_da_matti:
            self.context.remove(i)


    def gut_step(self,stato:Stato) -> Stato:

        local_b = self.grid.get_local_bounds()

        for b in self.context.agents(shuffle=True):
            if b.uid[1] == Batterio_Benefico.TYPE and b.stato=="vivo" :                                
                stato.nutrienti_b = b.assorbe(stato.nutrienti_b, stato)
                b.consuma()               
                stato.prodotto_b += b.produci() # Accumula              
                if b.duplica():
                    flag = True
                    nghs = self.ngh_finder.find(b.pt.x, b.pt.y)
                    rn.shuffle(nghs)
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
                stato.nutrienti_n= b.assorbe(stato.nutrienti_n, stato)
                b.consuma()
                stato.prodotto_n += b.produci() # Accumula
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
        
        # Verifica stato dell'intestino locale
        if stato.ratio_nb() > 1 and stato.eta >= stato.init_degenerazione:
            stato.infiammazione_local_g = True
            stato.citochine_gut += stato.ratio_nb() 
            stato.nutrienti_n += stato.ratio_nb()
            stato.nutrienti_b += stato.ratio_bn()
            return stato
        
        if stato.ratio_nb() < 1 and stato.citochine_gut > 0 and stato.eta >= stato.init_degenerazione:
            stato.citochine_gut -= stato.ratio_bn() 
            stato.nutrienti_b += stato.ratio_bn()
            stato.nutrienti_n += stato.ratio_bn()
            stato.infiammazione_local_g = False
            return stato
        
        if stato.ratio_nb == 1:
            stato.infiammazione_local_g = False
            return stato
            
        return stato
    

    def brain_step(self, stato:Stato) -> Stato:
        
        # Prima i neuroni influenzano il comportamento dei glia con il loro 
        for e in self.neural_net.graph:
            if e.uid[2] == self.rank and (e.uid[1] == Neuro.TYPE):
                e.step(stato)
        
        count_t = 0
        count_f = 0
        # I microglia attivano le proprie funzioni
        for e in self.neural_net.graph:        
            if e.uid[2] == self.rank and (e.uid[1] == GliaTest.TYPE):
                e.step(stato)
                if stato.infiammazione_local_b == True:
                    count_t +=1
                else:
                    count_f +=1

        if count_f > count_t:
            stato.infiammazione_local_b = False
        else:
            stato.infiammazione_local_b = True

        return stato
    

    def start(self):
        self.runner.execute()

    
    def at_end(self):
        # TODO aggiungere tempo della simulazione?
        if self.rank == 0:
            print("Done! 👋")
    

    def log_agents(self):   
        filepath = self.filepath

        # Apre il file in scrittura o lo crea se non esiste
        with open(filepath, "a") as file:
            for agent in self.context.agents():
                if agent.TYPE == Batterio_Benefico.TYPE or agent.TYPE == Batterio_Nocivo.TYPE:
                    pt = self.grid.get_location(agent)
                    if pt is not None:
                        file.write("{} {} {} {} {}\n".format(self.tick, agent.TYPE, pt.x, pt.y, self.rank))
                elif agent.TYPE == Soglia.TYPE: # and self.tick != 0:
                    file.write("{} {} {} {} {} {} {} {} {}\n".format(self.tick, agent.TYPE, "{:.3f}".format(agent.Env.nutrienti_b), "{:.3f}".format(agent.Env.nutrienti_n), "{:.3f}".format(agent.Env.prodotto_b), "{:.3f}".format(agent.Env.prodotto_n),"{:.3f}".format(agent.Env.citochine_gut), "{:.3f}".format(agent.Env.citochine_brain), self.rank))


def run(params: Dict, filepath, net_file_name):
    global model
    model = Model(MPI.COMM_WORLD, params, filepath, net_file_name)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)

    net_file_name = 'big_net' # TODO Da mettere come input nel parser...

    # Inizializza file di output
    output_folder = "./output" # Cartella di output
    execution_number = 1
    filepath = ""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  
    # Genera nome del file dinamico
    while True:
        filename = f"testGBA{execution_number}.txt"
        filepath = os.path.join(output_folder, filename) # Percorso del file di output
        if not os.path.exists(filepath):
            break
        execution_number += 1

    run(params, filepath, net_file_name)