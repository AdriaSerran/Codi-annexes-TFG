Modo de empleo
--------------

El script `src/entrena.py` permite realizar el entrenamiento de una red neuronal de PyTorch definiendo en la
invocación la propia red, los conjuntos de entrenamiento y validación, y el algoritmo de optimización.

Para poder usar distintos algoritmos y estructuras, `src/entrena.py` permite indicar qué scripts deben ser 
ejecutados con antelación al propio entrenamiento (opción `--execPre`). Tanto los lotes de entrenamiento y
validación como el modelo a entrenar se especifican mediante expresiones Python que se ejecutan internamente.

Creo que en el directorio `work` hay todo lo necesario para ejecutar un entrenamiento usando la base de datos
LibriSpeech con una función de coste definida sobre el reconocimiento de la trama. Sólo incluyo las señales
parametrizadas (directorio `Prm`, usando 40 coeficientes MFCC, si no recuerdo mal) y la segmentación en fonemas
(directorio `Trn`, los autores de la segmentación están nombrados en el fichero `Trn/autores.txt`).

Todo esto es un auténtico follón, pero en el script Bash `work/entrena.sh` hay un ejemplo de ejecución que he
utilizado con éxito para probar distintas redes con diversos algoritmos de optimización, todo ello con sus
respectivos parámetros y argumentos.

Por ejemplo, para entrenar una red DeepSpeech con función de coste la entropía cruzada y optimizador Adam (todo
ello definido en el propio script), y un valor del momentum igual a 0.99, un tamaño del lote de 10 señales y
un paso de aprendizaje igual a 1e-3, la orden es:

```.bash
	./entrena.sh 99 10 1e-3
```

Como sólo estaba interesado en la evolución del entrenamiento, no he previsto los algoritmos necesarios para
realizar reconocimiento de ningún tipo.

Prerrequitos
------------

En principio, el algoritmo usa Python 3.9 (se puede cambiar modificando el *hashbang* de `src/entrena.py`). No tengo
claro qué paquetes deben ser instalados con antelación. He incluido el fichero `requirements.txt` con el *freeze*
de mi zona de usuario. Seguro que sobran paquetes, pero seguro también que están todos los necesarios.
