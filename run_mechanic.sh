#!/bin/bash
#
# run_mechanic - [integra .pkl exportado]
# ----
# Copyright (C) 2025 Santiago Potes - BlackByte
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ----
# run_mechanic.sh
# Ejecutar el análisis completo del Mechanic

echo "🛠️  Ejecutando Matrix Mechanic..."

cd /home/arachne/OpenPotesApps/OctoMatrix_modular_warrior/ThePipeLine

# Activar entorno
source env/bin/activate

# Instalar dependencias de visualización
pip install matplotlib seaborn

# Ejecutar Mechanic
python matrix_mechanic.py

echo "✅ Mechanic completado. Ver:"
echo "   - output/confusion_matrix.png"
echo "   - output/relations_export.json"
echo "   - Consola para análisis de errores"
