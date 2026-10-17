@echo off
REM Ejecutar desde la raiz del repositorio
python profiling\benchmark.py --profile
snakeviz profile.out
