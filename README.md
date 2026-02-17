Proyecto de Predicción de Temperatura del Mar (SST) con LSTM

Este proyecto permite entrenar un modelo LSTM para predecir la
temperatura superficial del mar (SST) a partir de datos históricos.

------------------------------------------------------------------------

1) Clonar el repositorio

Copia el repositorio en la carpeta que elijas utilizando el siguiente
comando:

git clone

Reemplaza con la dirección correspondiente.

Luego ingresa a la carpeta del proyecto:

cd nombre-del-proyecto

------------------------------------------------------------------------

2) Instalar dependencias

Instala las dependencias necesarias que se encuentran en el archivo
requirements.txt:

pip install -r requirements.txt

------------------------------------------------------------------------

3) Preparar los datos

Para iniciar el entrenamiento es necesario tener un archivo .csv dentro
de la carpeta:

Data/

El archivo debe contener dos columnas obligatorias:

-   time → Fecha del registro
-   sst → Temperatura en grados Celsius correspondiente a esa fecha

Ejemplo de estructura del archivo:

time | sst
2020-01-01 | 26.5
2020-01-02 | 26.7

------------------------------------------------------------------------

4) Convertir archivos .nc a .csv (Opcional)

Si descargaste un dataset en formato .nc, puedes convertirlo a .csv
utilizando el programa ubicado en:

Tools/ncToCSV.py

Ejecuta el siguiente comando:

python Tools/ncToCSV.py

Sigue las instrucciones que aparecerán en pantalla para generar el
archivo .csv compatible.

------------------------------------------------------------------------

5) Entrenar el modelo

Una vez que el archivo .csv esté correctamente ubicado dentro de la
carpeta Data/, ejecuta:

python train_lstm.py

Esto iniciará el proceso de entrenamiento del modelo LSTM.

------------------------------------------------------------------------

6) Realizar predicciones

Después de completar el entrenamiento, ejecuta:

python predict.py

Esto iniciará el proceso de predicción utilizando el modelo entrenado.

------------------------------------------------------------------------

Notas

-   Asegúrate de que el archivo .csv tenga exactamente los nombres de
    columnas: time y sst.


