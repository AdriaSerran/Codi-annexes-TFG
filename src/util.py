from pathlib import Path


def leeLis(*ficLis):
    """
    Lee el contenido de uno o más ficheros de texto, devolviendo las palabras en ellos
    contenidas en la forma de lista de cadenas de texto.
    """

    lista = []
    for fic in ficLis:
        with open(fic, 'rt') as fpLis:
            lista += [pal for linea in fpLis for pal in linea.split()]

    return lista


def pathName(dirFic, nomFic, extFic):
    """
    Construye el path completo del fichero a partir de su directorio raíz 'dirFic', su
    nombre de señal 'nomFic' y su extensión 'extFic'.

    El resultado es un objeto de la clase 'Path'.
    """

    if extFic and not extFic.startswith('.'): extFic = '.' + extFic

    return Path(dirFic).joinpath(nomFic).with_suffix(extFic)


def chkPathName(pathName):
    """
    Crea, en el caso de que no exista previamente, el directorio del fichero
    'pathName'.
    """

    Path(pathName).parent.mkdir(parents=True, exist_ok=True)

checkPathName = chkPathName
