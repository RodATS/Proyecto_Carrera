<h2>VGG19</h2>

En el Entrenamiento 1: se hace un entrenamiento de 224 x 224 x 3, con el globalAverage2D, como indica el paper y se realizá el entrenamiento. Nos da como Accuracy: 0.6978.


En el Entrenamiento 2: se hace un entrenamiento de 224 x 224 x 3, con el globalAverage2D y se aplicaron a las imágenes rotaciones aleatorias
(-30°, 30°), volteo vertical y horizontal, y zoom (0.8, 1.2), como indica el paper y se realizá el entrenamiento. Nos da como Accuracy: 0.6978.

En el Entrenamiento 3: Se aplicó lo mismo que el Entrenamiento 2 y se añadirá las segmentaciones a las imágenes.
