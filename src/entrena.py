#! /usr/bin/python3.10 -u

import tqdm
from datetime import datetime as dt

def entrena(modelo, nomMod, lotesEnt, lotesDev=[], numEpo=1, numEval=10):
    print(f'Inicio de {numEpo} épocas de entrenamiento ({dt.now():%d/%b/%y %H:%M:%S}):', flush=True)
    for epo in range(numEpo):
        modelo.entrena(lotesEnt, lotesDev, epo=epo, numEval=numEval)

        if lotesDev:
            modelo.inicEval()
            for lote in tqdm.tqdm(lotesDev, ascii=' >='):
                for señal in lote:
                    modelo.addEval(señal)
        
            modelo.recaEval()
            modelo.printEval(epo + 1)

        if nomMod: modelo.escrMod(nomMod)

    print(f'Completadas {numEpo} épocas de entrenamiento ({dt.now():%d/%b/%y %H:%M:%S})', flush=True)


#################################################################################
# Invocación en línea de comandos
#################################################################################

if __name__ == '__main__':
    from docopt import docopt
    import sys

    Sinopsis = rf"""
    Entrena un modelo acústico para el reconocimiento del habla.

    Usage:
        {sys.argv[0]} [options] [<nomMod>]
        {sys.argv[0]} -h | --help
        {sys.argv[0]} --version

    Opciones:
        -e INT, --numEpo=INT                Número de épocas de entrenamiento [default: 50]
        -v INT, --numEval=INT               Número de evaluaciones por época de entrenamiento [default: 10]
        -x SCRIPT..., --execPre=SCRIPT...   Scripts Python a ejecutar antes del modelado
        -E EXPR..., --lotesEnt=EXPR...      Expresión que proporciona los lotes de entrenamiento
        -D EXPR..., --lotesDev=EXPR...      Expresión que proporciona los lotes de evaluación
        -M EXPR..., --modelo=EXPR...        Expresión que crea o lee el modelo inicial

    Argumentos:
        <nomMod>  Nombre del fichero en el que se almacenará el modelo

    Notas:
        La opción --execPre premite indicar uno o más scripts a ejecutar antes del entrenamiento.
        Para indicar más de uno, los diferentes scripts deberán estar separados por coma.

        Las opciones --lotesEnt, --lotesDev y --modelo permiten indicar una o más expresiones 
        Python a evaluar para obtener los lotes de entrenamiento y evaluación y el modelo, 
        respectivamente. Para indicar más de una expresión, éstas deberán estar separadas por 
        punto y coma.
    """

    args = docopt(Sinopsis, version=f'{sys.argv[0]}: Ramses v4.1 (2022)')

    numEpo = int(args['--numEpo'])
    numEval = int(args['--numEval'])

    nomMod = args['<nomMod>']

    if (scripts := args['--execPre']):
        for script in scripts.split(','):
            exec(open(script).read())
    
    if (exprs := args['--lotesEnt']):
        for expr in exprs.split(';'):
            lotesEnt = eval(expr)

    if (exprs := args['--lotesDev']):
        for expr in exprs.split(';'):
            lotesDev = eval(expr)

    if (exprs := args['--modelo']):
        for expr in exprs.split(';'):
            modelo = eval(expr)

    # import cProfile
    # cProfile.run('entrena(modelo=modelo, nomMod=nomMod, lotesEnt=lotesEnt, lotesDev=lotesDev, numEpo=numEpo, numEval=numEval)')
    entrena(modelo=modelo, nomMod=nomMod, lotesEnt=lotesEnt, lotesDev=lotesDev, numEpo=numEpo, numEval=numEval)
