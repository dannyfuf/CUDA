### 1. Para ejecutar el código, se debe tener instalada las siguientes librerías de python 
- matplotlib
- numpy
- os
- urllib

### 2. Para ejecutar el código se compila con:
```sh
nvcc main.cu -o main
```

### 3. Se ejecuta con:
```sh
main.exe
```
### 4. Durante la ejecución:
- El código pedirá el nombre de la imagen. En caso de que la imagen se llame `img.png` solamente se debe imgresar `img`.
- El nombre de la imagen más la extensión no debe superar los 500 caracteres de longitud.


### 5. Consideraciones:
- Dependiendo del comando con que se ejecuta python (pueden ser `python` o `python3`) se debe cambiar <br>
  la variable global `PYTHON_COMMAND`. Esta variable global se encuentra en las primeras lineas del `main.cu`.
    - Para `python` se tiene que definir como `0`
    - Para `python3` se tiene que definir como `1`
- Los scripts de python que convierten las imagenes a `.txt` y de `.txt` a imagen se ejecutan directamente desde el el `.cu`,
  por lo que no hay que convertir la imagen antes de ejecutar el `.cu`.
- Solamente se aceptan imagenes en formato `.png`.