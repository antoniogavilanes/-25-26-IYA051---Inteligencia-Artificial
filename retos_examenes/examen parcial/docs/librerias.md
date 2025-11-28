# Creando el entorno

## Crear el entorno virtual
```
python -m venv venv
```
## Permitir ejecucion de scripts
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Activar el entorno virtual
```
.\venv\Scripts\Activate.ps1
```
## Actualizar pip
```
python -m pip install --upgrade pip
```
## instalar dependencias
```
pip install opencv-python opencv-contrib-python numpy
```
## Ejecutar el proyecto
```
python -m src.main
```