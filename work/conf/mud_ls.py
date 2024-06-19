import sys

import tqdm
import textgrid as tg
import torch
import numpy as np
from datetime import datetime as dt

from util import *
from mar import *

from torch.utils.data import DataLoader
from torch.nn.functional import nll_loss
from torch.optim import SGD

import wandb

class ModLS():
    """
        Clase empleada para el modelado acústico basado en redes neuronales del estilo
        de PyTorch y compatible con las funciones 'entorch()' y 'recorch()'.

        Los objetos de la clase (modelos) pueden leerse de fichero, con el argumento
        'ficMod', o inicializarse desde cero, usando el argumento 'ficLisUni' para
        obtener la lista de unidades a modelar y/o reconocer.

        Tanto la función de pérdidas como el optimizador pueden especificarse en la
        invocación: la función de coste con el argumento 'funcLoss', que por defecto
        es igual a 'nll_loss'; y el optimizador con el argumento 'Optim', que por
        defecto es la clase 'SGD' con el paso de aprendizaje *congelado* a
        'lr=1.e-5'.
    """

    def __init__(self, ficLisUni=None, ficMod=None, red=None,
                 funcLoss=nll_loss, Optim=lambda params: SGD(params, lr=1.e-2), device=None, ficLog=None, wandbLog=True):
        if not device:
            self.device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            try:
                self.device = 'cpu' # torch.device(device)
            except:
                raise Exception(f'No se puede acceder al dispositivo {device}')

        if ficMod:
            self.leeMod(ficMod)
        else:
            unidades = leeLis(ficLisUni)

            self.red = red
            self.red.unidades = unidades
        
        if ficLog:
            try:
                chkPathName(ficLog)
                self.fpLog = open(ficLog, 'wt', buffering=1)
            except OSError as err:
                raise Exception(f'No se puede abrir el fichero {ficLog}: {err.strerror}')
            except:
                raise Exception(f'No se puede abrir el fichero {ficLog}')
        else:
            self.fpLog = None

        self.red = self.red.to(self.device)
        self.funcLoss = funcLoss
        self.optim = Optim(self.red.parameters())

        self.wandb = wandbLog
        if wandbLog:
            wandb.login()
            wandb.init(project="acme", config={"tarefa": "librispeech"})

    def escrMod(self, ficMod):
        try:
            chkPathName(ficMod)
            torch.jit.script(self.red).save(ficMod)
        except OSError as err:
            raise Exception(f'No se puede escribir el modelo {ficMod}: {err.strerror}')
        except:
            raise Exception(f'No se puede escribir el modelo {ficMod}')
    
    def leeMod(self, ficMod):
        try:
            self.red = torch.jit.load(ficMod)
        except OSError as err:
            raise Exception(f'No se puede leer el modelo {ficMod}: {err.strerror}')
        except:
            raise Exception(f'No se puede leer el modelo {ficMod}')
    
    def inicEntr(self):
        self.optim.zero_grad()
    
    @torch.no_grad()
    def __call__(self, prm):
        return self.red(prm).argmax(dim=-1)
    
    def inicEval(self):
        self.loss = 0.
        self.numUni = 0.
        self.numSen = 0.
        self.corr = 0.

    @torch.no_grad()
    def addEval(self, señal):
        trn = señal.trn.to(self.device)
        prm = señal.prm.to(self.device)
        salida = self.red(prm).squeeze()

        self.loss += self.funcLoss(salida, trn).item()
        self.numUni += len(trn)
        self.numSen += 1

        self.corr += torch.sum(salida.argmax(dim=-1) == trn)
        
    def recaEval(self):
        self.loss /= self.numSen
        self.corr /= self.numUni

    def printEval(self, epo):
#        print(f'{epo=:{".1f" if isinstance(epo, float) else "d"}}\t{self.lr=:.4e}\t{self.loss=:.4f}\t{self.corr=:.2%}\t({dt.now():%d/%b/%y %H:%M:%S})\n', flush=True)
        print('\u001b[2K\x0d', file=sys.stderr, end='')
        print(f'{epo=:{".1f" if isinstance(epo, float) else "d"}}\t{self.lr=:.4e}\t{self.loss=:.4f}\t{self.corr=:.2%}\t({dt.now():%d/%b/%y %H:%M:%S})\n', flush=True, end='')

        if self.wandb:
            log = {'lr': self.lr, 'loss': self.loss, 'corr': self.corr}
            for indice, group in enumerate(self.optim.param_groups):
                log[f'lr{indice}'] = group['lr']

            wandb.log(log)

        if not np.isfinite(self.loss):
            print(file=sys.stderr, flush=True)
            print(f'Valor no finito ({self.loss=}) detectado en la función de coste\n', file=sys.stderr, flush=True)
            sys.exit()

    @torch.enable_grad()
    def entrena(self, lotesEnt, lotesDev=None, epo=0, numEval=10):
        self.red = self.red.to(self.device)
        numLotes = len(lotesEnt)
        evalAct = 1
        for loteAct, lotes in enumerate(tqdm.tqdm(lotesEnt, ascii=' >=')):
            numUni = corr = 0

            def closure():
                nonlocal numUni, corr

                lossAct = 0
                numSen = 0
                numUni = 0
                corr = 0

                self.inicEntr()
                lote = lotes            # Porque puede llamarse más de una vez...
                for señal in lote:
                    trn = señal.trn.to(self.device)
                    prm = señal.prm.to(self.device)
                    salida = self.red(prm).squeeze()
                    loss = self.funcLoss(salida, trn)
                    if torch.is_grad_enabled():
                        loss.backward()
                    
                    lossAct += loss.item()
                    numUni += len(trn)
                    numSen += 1
                    if self.fpLog:
                        corr += torch.sum(salida.argmax(dim=-1) == trn)

                return lossAct / numSen

            lossAnt = self.optim.step(closure=closure)

            lr = self.optim.param_groups[0]['lr']
            losses = self.optim.param_groups[0]['losses']
            mellora = self.optim.param_groups[0]['mellora']
            ret = self.optim.param_groups[0]['ret']
            # l0, l1, l2, l4, lr1, lr2, ee, lr, a_, maxParam, maxGrad, maxBuff = self.optim.step(closure=closure)
            self.lr = lr
            
            if self.fpLog:
                corr = float(corr) / numUni
                print(f'{lr=:.4e}\t{lossAnt=:.4f}\t{corr=:.2%}\t{losses}\t{mellora}\t{ret}', file=self.fpLog)
                #print(f'{lr=:.4e}\t{lossTot=:.4f}\t{corr=:.2%}\t{a=:8.3e}\t{a_=: 8.3e}\t{s=: 8.3f}\t{s_=: 8.3f}\t{k=: 8.3f}\t{mellora}', file=self.fpLog)
                # ind = sorted([0, 1, 2], key=lambda x: {0: l0, 1: l1, 2: l2}[x])
                # print(f'{l0=:.5e}\t{l1=:.5e}\t{l2=:.5e}\t{l4=:.5e}\t{lr1=:.4e}\t{lr2=:.4e}\t{ee=:.4e}\t{lr=:.4e}\t{a_=:.4e}\t{lossTot=:.4f}\t{corr=:.2%}\t{ind=}\t{maxParam=:f}\t{maxGrad=:f}\t{maxBuff=:f}', file=self.fpLog)

            if lotesDev and numEval and round(numLotes * evalAct / numEval) == loteAct:
                self.inicEval()
                for lote in lotesDev:
                    for señal in lote:
                        self.addEval(señal)
            
                self.recaEval()
                self.printEval(epo + evalAct / numEval)
                evalAct += 1



def calcDimIni(dirPrm, *ficLisPrm):
    """
        Función de conveniencia para determinar el número de coeficientes de las
        señales parametrizadas en el directorio 'dirPrm'. Es útil para dimensionar
        adecuadamente la capa de entrada de una red neuronal.
    """

    pathPrm = pathName(dirPrm, leeLis(*ficLisPrm)[0], 'prm')
    return torch.load(pathPrm)['prm'].shape[-1]
    

def calcDimSal(ficLisUni):
    """
        Función de conveniencia para determinar el número de unidades de una lista de
        unidades acústicas. Es útil para dimensionar adecuadamente la capa de salida
        de una red neuronal.
    """
    
    return len(leeLis(ficLisUni))


def creaLisUni(dirTrn, lisSen, ficLis):
    """
    Lee los ficheros de transcripción indicados por 'lisSen' y escribe la lista de las 
    unidades acústica encontradas en el fichero 'ficLis'.

    Por ahora, sólo consulta el segundo ítem del fichero TextGrid, que se supone que incluye
    la transcripción en fonemas...
    """

    lisUni = []
    for nomSen in tqdm.tqdm(lisSen, ascii=' >='):
        pathTrn = pathName(dirTrn, nomSen, "TextGrid")
        trnTG = tg.TextGrid.fromFile(pathTrn)

        for segm in range(len(trnTG[1])):
            mark = trnTG[1][segm].mark[:2]
            if mark and mark not in lisUni: lisUni.append(mark)
    
    try:
        chkPathName(ficLis)
        with open(ficLis, 'wt') as fpLis:
            for unidad in sorted(lisUni):
                print(unidad, file=fpLis)
    except:
        raise Exception('Error al escribir {ficLis}')


def cargaTrn(dirTrn, lisSen, lisUni=None, fm=16000):
    """
    Lee los ficheros de transcripción indicados por 'lisSen' y devuelve la transcrición de cada 
    uno de los ficheros, en la forma de tupla de tres elementos: muestra de inicio, muestra de
    final y valor de la unidad acústica.

    Si se indica un diccionario de unidades, 'lisUni', el valor de la unidad es su índice en el
    diccionario (se eleva una excepción si la unidad no aparece). Si no se indica el diccionario,
    el valor es directamente el contenido del fichero (campo 'mark').
    """

    trn = [[None]] * len(lisSen)
    for sen, nomSen in tqdm.tqdm(list(enumerate(lisSen)), ascii=' >='):
        pathTrn = pathName(dirTrn, nomSen, "TextGrid")
        trnTG = tg.TextGrid.fromFile(pathTrn)

        trn[sen] = [None] * len(trnTG[1])
        for segm in range(len(trnTG[1])):
            minTime, maxTime, mark = trnTG[1][segm].minTime, trnTG[1][segm].maxTime, trnTG[1][segm].mark
            mark = mark[:2]
            if lisUni and mark:
                try:
                    mark = lisUni.index(mark)
                except:
                    raise Exception(f'Unidad "{mark}" no encontrada en la lista de unidades')
            
            trn[sen][segm] = (round(minTime * fm), round(maxTime * fm), mark)
        if trn[sen][-1][2] == '':
            del trn[sen][-1]
    
    return trn


class DataSet(torch.utils.data.Dataset):
    def __init__(self, dirPrm, dirTrn, ficLisUni, *ficLisSen):
        super().__init__()

        lisSen = leeLis(*ficLisSen)
        lisUni = leeLis(ficLisUni)

        self.dirPrm = dirPrm
        self.lisSen = lisSen
        self.lisUni = lisUni

        print(f'Carga de las transcripciones de {ficLisSen} ({dt.now():%d/%b/%y %H:%M:%S}):', flush=True)
        self.trnSen = cargaTrn(dirTrn, lisSen, lisUni)

    def __len__(self):
        return len(self.lisSen)
    
    def __getitem__(self, ind):
        pathPrm = pathName(self.dirPrm, self.lisSen[ind], 'prm')
        try:
            dicPrm = torch.load(pathPrm)
            prm = dicPrm['prm']
            dspVnt = dicPrm['dspVnt']
            lngVnt = dicPrm['lngVnt']
        except:
            raise Exception(f'No se puede leer {pathPrm}')
        
        finTrn = self.trnSen[ind][-1][1] // dspVnt + 1
        trn = torch.zeros(finTrn, dtype=torch.int64)
        for ini, fin, uni in self.trnSen[ind]:
            iniUni = round(ini / dspVnt)
            finUni = min(round(fin / dspVnt) + 1, finTrn)
            trn[iniUni : finUni] = uni


        finTrn = int(np.floor((self.trnSen[ind][-1][1] - lngVnt / 2.) / dspVnt) + 1)
        trn = torch.zeros(finTrn, dtype=torch.int64)
        for ini, fin, uni in self.trnSen[ind]:
            iniUni = int(max(np.ceil((ini - lngVnt / 2.) / dspVnt), 0));
            finUni = int(min(np.floor((fin - lngVnt / 2.) / dspVnt) + 1, finTrn));
            trn[iniUni : finUni] = uni



        prm = prm[:, : finTrn, :]
        with torch.no_grad():
            prm = torch.nn.BatchNorm1d(prm.shape[-1])(prm.permute(0, 2, 1)).permute(0, 2, 1)

        return prm[:, : finTrn, :], trn, self.lisSen[ind]

from collections import namedtuple
señal = namedtuple('señal', ('prm', 'trn', 'sen'))

def miCollateFun(batch):
    for ind in range(len(batch)):
        batch[ind] = señal(prm=batch[ind][0], trn=batch[ind][1], sen=batch[ind][2])

    return batch

def lotesLS(dirPrm, dirTrn, ficLisUni, tamBatch, *ficLisSen):
    """
        Función que proporciona lotes de señales compatibles con las funciones 'entrena()'
        y 'reconoce()' para usar con la base de datos LibriSpeech en tareas de 
        reconocimiento de fonemas trama a trama usando las redes neuronales de PyTorch.

        Cada señal está representada por una tupla nominada (namedtuple) en la que se
        tienen los campos siguientes:

        prm: señal parametrizada con un formato compatible con el admitido por las
             redes definidas en TorchAudio (como DeepSpeech o Wav2Vec): BxCxTxF, donde
             B es el minilote, C es el canal, T es el tiempo y F es la feature. Este
             formato también es compatible con las redes definidas en neuras/mlp.py.
             
             Dado que tanto el tamaño del minilote, como el número de canales, como la
             duración de las señales usadas en el reconocimiento de vocales es uno,
             las dimensiones que tendrá la señal parametrizada es 1x1x1xF, donde F
             es el número de coeficientes de la señal parametrizada.-
             
        trn: transcripción de la señal con un formato compatible con el admitido por
             las funciones de coste definidas en torch.nn.functional (como nll_loss):
             BxT, donde B es el minilote y T es el tiempo. La transcripción en sí misma
             es el índice de la unidad en la lista de unidades.
             
             Dado que tanto el tamaño del minilote como la duración de la señal son
             igual es a uno, la dimensión de la transcripción ha de ser 1x1.

        sen: Nombre de la señal. Sólo se usa en el reconocimiento para saber el nombre
             del fichero en el que se ha de escribir el resultado.
    """

    paramsBatch = {
        'batch_size'    : tamBatch,
        'shuffle'       : True,
        'num_workers'   : 2,
        'collate_fn'    : miCollateFun,
        'drop_last'     : True,
        'pin_memory'    : False,
    }

    return DataLoader(DataSet(dirPrm, dirTrn, ficLisUni, *ficLisSen), **paramsBatch)
