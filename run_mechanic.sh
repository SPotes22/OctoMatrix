#!/bin/bash
# run_mechanic.sh
# Ejecutar el análisis completo del Mechanic

echo "🛠️  Ejecutando Matrix Mechanic..."

cd /home/arachne/OpenPotesApps/OctoMatrix_modular_warrior/ThePipeLine

# Activar entorno
source env/bin/activate

# Instalar dependencias de visualización
pip install matplotlib seaborn

# Ejecutar Mechanic
python ../matrix_mechanic.py

echo "✅ Mechanic completado. Ver:"
echo "   - output/confusion_matrix.png"
echo "   - output/relations_export.json"
echo "   - Consola para análisis de errores"
