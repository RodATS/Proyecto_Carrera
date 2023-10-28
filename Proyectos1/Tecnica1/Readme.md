<h2>VGG19</h2>

Enlace al Paper: https://www.ias-iss.org/ojs/IAS/article/view/2346
Enlace al Github del Paper: https://github.com/miag-ull/rim-one-dl/blob/master/README.md


En el Entrenamiento 1: se hace un entrenamiento de 224 x 224 x 3, con el globalAverage2D, como indica el paper y se realizá el entrenamiento. Nos da como Accuracy: 0.6978.

En el Entrenamiento 2: se hace un entrenamiento de 224 x 224 x 3, con el globalAverage2D y se aplicaron a las imágenes rotaciones aleatorias
(-30°, 30°), volteo vertical y horizontal, y zoom (0.8, 1.2), como indica el paper y se realizá el entrenamiento. Nos da como Accuracy: 0.6978.

En el Entrenamiento 3: Se aplicó lo mismo que el Entrenamiento 2 y se añadirá las segmentaciones a las imágenes. Nos da: loss: 0.8942 - accuracy: 0.6782

En el Entrenamiento 4: Se aplicó lo mismo que el Entrenamiento 3 y se aumentaron las épocas. 

En el Entrenamiento 5: Se aplicó lo mismo que el Entrenamiento 4, epocas = 15, batch_size=10, en include_Top=False.

En el Entrenamiento 6: Se aplicó lo mismo que el Entrenamiento 5, batch_size=32.
