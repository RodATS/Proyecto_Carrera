<h1>Xception</h1>

Paper de la técnica: https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-019-0649-y
Aquí se encuentran los códigos y como armar las carpetas para obtener los resultados: https://figshare.com/s/c2d31f850af14c5b5232

En results_Xception.csv se encontrarán los resultados luego de pasar imágenes por el modelo a entrenado, dando por resultado las probailidaddes de ser glaucoma o no. Se observa que acierta en todas las imágenes.

En el documento se verá los accuracy de cada Fold que se realizó en el proyecto, al utilizar diferentes bases de datos.

Después de correr el modelo entrenado para hallar el Accuracy, se hizo el análisis con 100 imágenes y se usó el 30% para el test set. Los resultados se pueden observar en el histograma donde se utiliza el Dataset RIM-ONE, se obtuvo: 95% acc interval 77.22% and 91.18%
![image](https://github.com/RodATS/Proyecto_Carrera/assets/77297145/89618be4-0f2e-4c7b-8b09-c862acbee964)
