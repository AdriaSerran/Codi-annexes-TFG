import torch

class MLP_3(torch.nn.Module):
    """
        Clase que implementa el perceptrón de tres capas, en el que la primera recibe
        la señal parametrizada y la tercera devuelve un *log_softmax* del mismo orden
        que el vocabulario a reconocer.

        La no linealidad implementada en las dos primeras capas es la unidad lineal
        rectificada (ReLU). La capa de salida usa el logaritmo del máximo suavizado
        (log_softmax)
    """
    def __init__(self, dimIni=40, dimInt=128, dimSal=5):
        super().__init__()

        self.capa1 = torch.nn.Linear(in_features=dimIni, out_features=dimInt)
        self.capa2 = torch.nn.Linear(in_features=dimInt, out_features=dimInt)
        self.capa3 = torch.nn.Linear(in_features=dimInt, out_features=dimSal)
    
    def forward(self, x):
        x = self.capa1(x)
        x = torch.nn.functional.relu(x)
        x = self.capa2(x)
        x = torch.nn.functional.relu(x)
        x = self.capa3(x)
        x = torch.nn.functional.log_softmax(x, dim=-1)

        return x.reshape(1, 1, -1)


from torch.nn import ReLU, LogSoftmax, Sigmoid

def mlp_N(numCap=3, dimIni=40, dimInt=128, dimSal=5, clsAct=ReLU(), clsSal=LogSoftmax(dim=-1)):
    """
        Función que devuelve un perceptrón multi-capa (MLP), en el que la primera de
        ellas recibe la señal parametrizada (de dimensión 'dimIni') y la última 
        devuelve un vector del mismo orden que el vocabulario a reconocer ('dimSal').
        Las capas intermedias son del tamaño indicado por 'dimInt'.

        El número de capas viene determinado por el argumento 'numCap' y debe ser
        igual o superior a dos.
        
        Salvo la capa de salidas, todas las capas usan la no linealidad indicada por
        la clase 'clsAct'; por defecto, la unidad lineal rectificada (ReLU). La capa
        de salida usa la clase indicada por 'clsSal'; por defecto, el logaritmo del 
        máximo suavizado (log_softmax)
    """
    if numCap < 2:
        raise Exception('El número mínimo de capas del perceptrón es 2')

    capas = [torch.nn.Linear(dimIni, dimInt)]
    capas.append(clsAct)
    for _ in range(numCap - 2):
        capas.append(torch.nn.Linear(dimInt, dimInt))
        capas.append(clsAct)
    capas.append(torch.nn.Linear(dimInt, dimSal))
    capas.append(clsSal)
    capas.append(torch.nn.Flatten(2))

    return torch.nn.Sequential(*capas)
