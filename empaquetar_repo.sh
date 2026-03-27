#!/bin/bash

OUTPUT="repo_completo.txt"

# Limpiamos el archivo de salida si ya existe
> "$OUTPUT"

echo "⏳ Escaneando y empaquetando el código..."

# Buscamos archivos ignorando los directorios de dependencias y control de versiones
find . -type f \
  -not -path "*/\.git/*" \
  -not -path "*/node_modules/*" \
  -not -path "*/venv/*" \
  -not -path "*/env/*" \
  -not -path "*/\.idea/*" \
  -not -path "*/\.vscode/*" \
  -not -path "*/__pycache__/*" \
  -not -path "*/build/*" \
  -not -path "*/dist/*" \
  -not -name "$OUTPUT" \
  -not -name "empaquetar_repo.sh" | sort | while read -r archivo; do
  
  # MAGIA DE DEBIAN: Verificamos que el archivo sea texto (código) y no un binario/imagen
  if file -b --mime-type "$archivo" | grep -q '^\(text/\|application/json\)'; then
      echo -e "\n========================================================" >> "$OUTPUT"
      echo "Ruta del archivo: $archivo" >> "$OUTPUT"
      echo "========================================================" >> "$OUTPUT"
      cat "$archivo" >> "$OUTPUT"
  fi
done

echo "✅ ¡Listo! Todo tu código está consolidado en el archivo: $OUTPUT"
