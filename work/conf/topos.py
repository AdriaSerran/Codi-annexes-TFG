import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torchaudio.models import DeepSpeech, Wav2Letter
from torch.nn import LSTM
from torch.optim import SGD, Adam, LBFGS, Rprop
import torch

torch.random.manual_seed(1789)

from mlp import *

dirPrm = 'Prm'
dirTrn = 'Trn'

guiTrain='Gui/train-clean-100.gui'
guiDevel='Gui/dev-clean.gui'

ficLisUni = 'Lis/fonemas.lis'

numCof = calcDimIni(dirPrm, guiTrain)
tamVoc = calcDimSal(ficLisUni)
