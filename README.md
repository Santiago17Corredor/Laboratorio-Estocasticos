# Flujo del Sonido en el Clasificador FM-Noise - Paso a Paso

## Descripción General
Este documento explica cómo el audio es capturado, procesado y clasificado como FM (ruido modulado en frecuencia) o WN (ruido blanco) en la aplicación.

---

## PASO 1: Inicio de la Aplicación
**Ubicación:** `main.py`

```
main.py → Llama a launch() desde src/interface/interface.py
```

- La aplicación comienza cuando se ejecuta `main.py`
- Se importa y ejecuta la función `launch()` que inicializa la interfaz gráfica

---

## PASO 2: Interfaz Gráfica y Captura de Audio
**Ubicación:** `src/interface/interface.py`

```
src/interface/interface.py
├── Crea ventana gráfica con botones [Empezar a grabar] y [Detener]
├── Configura parámetros:
│   ├── SR (Sample Rate): 44100 Hz
│   └── BLOCK_SIZE: 1024 muestras por bloque
├── Inicializa AudioTransformer desde src/utils/audioTransformer.py
└── Abre stream de micrófono con sounddevice
```

**Qué sucede:**
- El usuario hace clic en "Empezar a grabar"
- Se abre un stream de audio del micrófono usando la librería `sounddevice`
- El audio entra en bloques de 1024 muestras a la técnica `callback()`

---

## PASO 3: Cola de Audio y Buffer
**Ubicación:** `src/interface/interface.py` - Función `callback()`

```
Micrófono → callback(indata) → audio_queue → AudioTransformer.agregar()
```

**Qué sucede:**
- Cada bloque de audio (1024 muestras) se recibe en la función `callback()`
- Se convierte a formato float64 de numpy: `indata[:, 0].astype(np.float64)`
- Se coloca en una `queue.Queue` para procesamiento
- Simultáneamente se llama a `transformer.agregar(bloque)` en `AudioTransformer`

---

## PASO 4: Acumulación de Audio en Ventanas
**Ubicación:** `src/utils/audioTransformer.py` - Clase `AudioTransformer`

```
AudioTransformer:
├── Recibe bloques de 1024 muestras
├── Los une en un buffer interno (_buffer)
└── Cuando acumula 88,200 muestras (2 segundos), procesa una ventana
    (FRAMES_VENTANA = SR * DURACION_SEG = 44100 * 2)
```

**Método `agregar()`:**
- Concatena el nuevo bloque al buffer existente
- Verifica si hay suficientes muestras (≥ 88,200)
- Si hay una ventana completa, la separa y procesa
- El buffer se actualiza eliminando las muestras ya procesadas

```python
Ejemplo:
Buffer = [1,2,3,4,5,...] (acumulándose)
Cuando Buffer ≥ 88,200 → Extrae ventana de 88,200
Buffer actualizado = lo que sobra
```

---

## PASO 5: Procesamiento de Audio (Transformaciones Matemáticas)
**Ubicación:** `src/processing/process.py`

Cada ventana de 2 segundos (88,200 muestras) pasa por 3 transformaciones:

### 5.1 Cálculo de Autocovariance
```
src/processing/process.py → calcAutocovariance(ventana)
├── Usa la librería statsmodels: acovf()
├── Calcula la autocovariance de la ventana
├── Resultado: Vector que describe las correlaciones internas del audio
└── Salida: Array numpy de float64
```

### 5.2 Transformada de Fourier
```
src/processing/process.py → calcFourier(acov)
├── Aplica FFT (Fast Fourier Transform) al vector de autocovariance
├── np.fft.fft(acov)
├── Convierte el dominio temporal al dominio de frecuencias
└── Salida: Array de números complejos con magnitudes y fases
```

### 5.3 Cálculo de la Norma (Magnitud)
```
src/processing/process.py → calcNorm(fourier)
├── Extrae el valor absoluto (magnitud) de cada componente de frecuencia
├── np.abs(fourier)
├── Ignora la fase, solo mantiene amplitudes
└── Salida: Array de números reales (magnitudes en el dominio de frecuencias)
```

**En AudioTransformer:**
```
src/utils/audioTransformer.py → _procesarVentana(ventana)
Ejecuta estos 3 pasos y almacena los resultados:
├── acov = calcAutocovariance(ventana)
├── fourier = calcFourier(acov)
├── norm = calcNorm(fourier)
└── Guarda en _resultados = [[acov, fourier, norm], ...]
```

---

## PASO 6: Promediado de Fragmentos
**Ubicación:** `src/utils/audioTransformer.py` - Métodos `cantidadFragmentos()` y `promedioActual()`

```
La interfaz muestra:
"Suficiencia para comparación: X fragmentos procesados"
```

**Qué sucede:**
- Cada ventana procesada se añade a la lista `_resultados`
- Se pueden procesar múltiples ventanas (si el usuario graba más de 2 segundos)
- La interfaz gráfica muestra cuántos fragmentos se han acumulado
- Esto permite comparaciones más robustas

---

## PASO 7: Detención de Grabación y Guardado de Datos
**Ubicación:** `src/utils/audioTransformer.py` - Método `detener()`

```
Usuario hace clic en [Detener] → detener() → calcula promedio
```

**Proceso:**
```
1. Toma todos los fragmentos procesados de _resultados
2. Ejecuta calcAvgVector() para promediar:
   - Todas las autocovariances
   - Todas las transformadas de Fourier
   - Todas las normas
3. Guarda el resultado en archivos (respaldo en disco):
   - src/dataSet/micProcessed.txt (vector de norma promediado)
   - src/dataSet/micAcov.txt (autocovariance promediado)
4. Retorna el dict avg con los vectores ya calculados ← REFACTORIZADO
```

**Estructura de src/dataSet/:**
```
src/dataSet/
├── fmVector.txt         (Vector FM de referencia - norma promediada)
├── fmAcov.txt          (Autocovariance FM de referencia)
├── wnVector.txt        (Vector WN de referencia - norma promediada)
├── wnAcov.txt          (Autocovariance WN de referencia)
├── micProcessed.txt    (Vector micrófono - norma promediada) ← GENERADO EN PASO 7
└── micAcov.txt         (Autocovariance micrófono) ← GENERADO EN PASO 7
```

---

## PASO 8: Cargue de Vectores de Referencia
**Ubicación:** `src/models/classifier.py` - Función `classify(mic_norm, mic_acov)`

```
classify() recibe los vectores del micrófono como parámetros ← REFACTORIZADO
y carga desde src/dataSet/ solo los vectores de referencia FM y WN:
```

**Archivos de entrenamiento previo:**
```
Los vectores FM y WN fueron creados previamente en src/loaders/loaderDataSet.py:
1. src/data/FM/ → archivos de audio FM (entrenamiento)
2. src/data/WN/ → archivos de audio WN (ruido blanco)
3. Se procesan todos los archivos igual que en PASOS 5-7
4. Se promedian todos → generan vectores de referencia en src/dataSet/
```

**En la clasificación se cargan desde disco:**
```
- fm_norm  = carga src/dataSet/fmVector.txt
- fm_acov  = carga src/dataSet/fmAcov.txt
- wn_norm  = carga src/dataSet/wnVector.txt
- wn_acov  = carga src/dataSet/wnAcov.txt
```

**Los vectores del micrófono llegan como parámetros (ya en memoria):**
```
- mic_norm → avg["norm"] retornado por detener() en PASO 7
- mic_acov → avg["acov"] retornado por detener() en PASO 7
```

---

## PASO 9: Normalización de Vectores
**Ubicación:** `src/models/classifier.py` - Función `_normalize(v)`

```
Todos los vectores se normalizan para comparación justa:
```

**Proceso:**
```
normalized = vector / (||vector|| + 1e-10)

1. Calcula la norma euclidiana (magnitud total) del vector
2. Divide cada componente entre esa magnitud
3. Resultado: Vector de magnitud unitaria (valores entre 0 y 1)
4. Esto permite comparar formas sin importar amplitud absoluta

Vectores normalizados:
├── mic_n = normalizado(mic_norm)
├── fm_n  = normalizado(fm_norm)
└── wn_n  = normalizado(wn_norm)
```

---

## PASO 10: Cálculo de Distancias (Comparación)
**Ubicación:** `src/models/classifier.py` - Función `classify()`

```
Se comparan los espectros normalizados del micrófono con los de referencia:
```

**Métrica de distancia - Distancia Manhattan (L1):**
```
dist_fm = mean(|mic_n - fm_n|)
dist_wn = mean(|mic_n - wn_n|)

Ejemplo numérico:
- Si mic_n =     [0.1, 0.5, 0.9]
- Y fm_n =       [0.1, 0.4, 0.92]
- Diferencia =   [0.0, 0.1, 0.02]
- Distancia FM = (0.0 + 0.1 + 0.02) / 3 = 0.04
```

**Qué significa:**
- Distancia pequeña = El audio es similar al patrón
- Distancia grande = El audio es diferente al patrón

---

## PASO 11: Toma de Decisión y Resultado
**Ubicación:** `src/models/classifier.py` - Función `classify()`

```
Comparación final:

if dist_fm < dist_wn:
    Resultado = "FM"
else:
    Resultado = "Ruido Blanco"
```

**Salida de la clasificación:**
```
print("Distancia con FM          : {dist_fm:.5f}")
print("Distancia con Ruido Blanco: {dist_wn:.5f}")
print("Resultado: {resultado}")

Retorna: (resultado, dist_fm, dist_wn, mic_n, fm_n, wn_n, mic_acov, fm_acov, wn_acov)
```

---

## PASO 12: Visualización en la Interfaz
**Ubicación:** `src/interface/interface.py` - Método `_actualizarGrafica()`

```
Los resultados se muestran en la interfaz gráfica:

Panel Izquierdo:
├── [Empezar a grabar] - Botón para iniciar
├── [Detener] - Botón para finalizar
├── "Estado de grabación" - Muestra "Grabando..." o "Detenido"
├── "Suficiencia para comparación" - Muestra número de fragmentos
└── "Clasificación" - Muestra "FM" o "Ruido Blanco"

Panel Derecho:
├── Gráfica de la forma de onda conforme se graba
├── Gráfica de espectrus (si hay gráficas adicionales)
└── Visualización de componentes de Fourier normalizados
```

---

## RESUMEN DEL FLUJO COMPLETO

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FLUJO COMPLETO DEL AUDIO                         │
└─────────────────────────────────────────────────────────────────────┘

1. [main.py]
   ↓
2. [src/interface/interface.py] - Crea GUI y abre micrófono
   ↓
3. [Micrófono] → Captura en bloques de 1024 muestras
   ↓
4. [src/utils/audioTransformer.py] - Acumula bloques en buffer
   ↓
5. [src/processing/process.py] - Procesa ventanas de 2 segundos:
   ├── calcAutocovariance()  → Correlación interna
   ├── calcFourier()         → Transformada para frecuencias
   └── calcNorm()            → Magnitudes de frecuencias
   ↓
6. [src/utils/audioTransformer.py] - Almacena fragmentos en _resultados
   ↓
7. Usuario detiene grabación
   ↓
8. [src/utils/audioTransformer.py.detener()] → Promedia todos los
   fragmentos, guarda en src/dataSet/ y retorna avg ← REFACTORIZADO
   ↓
9. [src/models/classifier.py] → Recibe mic_norm y mic_acov como
   parámetros (ya en memoria) y carga FM/WN desde src/dataSet/ ← REFACTORIZADO
   ↓
10. [src/models/classifier.py] → Normaliza todos los vectores
   ↓
11. [src/models/classifier.py] → Calcula distancias:
   - Distancia con FM (fmVector.txt)
   - Distancia con WN (wnVector.txt)
   ↓
12. [src/models/classifier.py] → Compara distancias:
   - Si dist_fm < dist_wn → "FM"
   - Si dist_wn < dist_fm → "Ruido Blanco"
   ↓
13. [src/interface/interface.py] → Muestra resultado en pantalla
   ├── Etiqueta de clasificación
   ├── Gráficas de espectros
   └── Valores de distancia

```

---

## REFACTORIZACIÓN: Eliminación del Round-Trip en Disco

### Problema original

En la versión original del código, la comunicación entre `detener()` y `classify()` pasaba
obligatoriamente por el sistema de archivos:

```
detener()                         classify()
   │                                  │
   ├── calcula avg                    ├── np.loadtxt(micProcessed.txt)  ← lee disco
   ├── np.savetxt(micProcessed.txt)   └── np.loadtxt(micAcov.txt)       ← lee disco
   └── np.savetxt(micAcov.txt)
         ↑
    escribe disco
```

Los vectores del micrófono ya estaban calculados en memoria dentro de `detener()`,
pero se descartaban. `classify()` los tenía que volver a leer del disco, generando
un **round-trip innecesario**: memoria → disco → memoria.

### Solución aplicada

Se eliminó el round-trip pasando los vectores directamente en memoria:

**`audioTransformer.py` — `detener()` ahora retorna `avg`:**
```python
# Antes
def detener(self) -> str:
    ...
    np.savetxt(PATH_MIC,      avg["norm"])
    np.savetxt(PATH_MIC_ACOV, avg["acov"])
    return PATH_MIC  # retornaba el path, nunca usado

# Después
def detener(self) -> dict:
    ...
    np.savetxt(PATH_MIC,      avg["norm"])
    np.savetxt(PATH_MIC_ACOV, avg["acov"])
    return avg  # retorna los vectores ya calculados
```

**`classifier.py` — `classify()` recibe los vectores como parámetros:**
```python
# Antes
def classify():
    mic_norm = np.loadtxt(PATH_MIC)       # leía del disco
    mic_acov = np.loadtxt(PATH_MIC_ACOV)  # leía del disco

# Después
def classify(mic_norm: np.ndarray, mic_acov: np.ndarray):
    # recibe los vectores directamente, sin tocar disco
```

**`interface.py` — `_detenerGrabacion()` conecta los dos:**
```python
# Antes
transformer.detener()
resultado, ... = classify()

# Después
avg = transformer.detener()
resultado, ... = classify(avg["norm"], avg["acov"])
```

### Por qué se hizo así

El flujo queda completamente **explícito y lineal**: al leer `_detenerGrabacion()` se ve
de dónde vienen los datos del micrófono sin necesidad de rastrear archivos intermedios.
Los archivos `.txt` del micrófono se siguen guardando en disco como respaldo, pero ya
no son el canal de comunicación entre módulos.

---

## ESTRUCTURA DE CARPETAS INVOLUCRADAS

```
FM-Noise-Classifier-Algorithm/
│
├── main.py                          ← PASO 1: Punto de entrada
│
├── src/
│   │
│   ├── interface/
│   │   └── interface.py             ← PASOS 2, 3, 12, 13: GUI y micrófono
│   │
│   ├── utils/
│   │   └── audioTransformer.py      ← PASOS 4, 5, 6, 7: Acumulación y procesamiento
│   │
│   ├── processing/
│   │   └── process.py               ← PASO 5: Transformaciones (Autocovariance, FFT, Norma)
│   │
│   ├── models/
│   │   └── classifier.py            ← PASOS 8, 9, 10, 11: Clasificación
│   │
│   ├── data/
│   │   ├── FM/                      ← Audios de entrenamiento FM
│   │   └── WN/                      ← Audios de entrenamiento WN
│   │
│   ├── dataSet/
│   │   ├── fmVector.txt             ← Vector FM de referencia (creado una vez)
│   │   ├── fmAcov.txt               ← Autocovariance FM de referencia
│   │   ├── wnVector.txt             ← Vector WN de referencia (creado una vez)
│   │   ├── wnAcov.txt               ← Autocovariance WN de referencia
│   │   ├── micProcessed.txt         ← Respaldo del micrófono procesado ← GENERADO PASO 7
│   │   └── micAcov.txt              ← Respaldo autocovariance micrófono ← GENERADO PASO 7
│   │
│   └── loaders/
│       └── loaderDataSet.py         ← Carga y procesa archivos de entrenamiento
```

---

## PARÁMETROS CLAVE

| Parámetro | Valor | Ubicación | Función |
|-----------|-------|-----------|---------|
| Sample Rate (SR) | 44100 Hz | interface.py | Frecuencia de muestreo del micrófono |
| Block Size | 1024 muestras | interface.py | Tamaño de cada bloque transferido |
| Duración Ventana | 2 segundos | audioTransformer.py | Cantidad de audio a procesar de una vez |
| Frames Ventana | 88,200 muestras | audioTransformer.py | SR × 2 segundos = 88,200 |

---

## CASOS DE USO

### Caso 1: Entrenamiento Inicial
```
1. Ejecutar: python -m src.loaders.loaderDataSet
2. Lee archivos de src/data/FM/ y src/data/WN/
3. Procesa como PASOS 5-6 pero para múltiples archivos
4. Genera vectores de referencia en src/dataSet/
5. Guarda: fmVector.txt, fmAcov.txt, wnVector.txt, wnAcov.txt
```

### Caso 2: Clasificación en Tiempo Real
```
1. Usuario ejecuta main.py
2. Interfaz abre (PASO 2)
3. Usuario graba audio (PASOS 3-4)
4. Se procesan ventanas (PASO 5-6)
5. Usuario detiene (PASO 7) → detener() retorna avg
6. classify() recibe avg["norm"] y avg["acov"] como parámetros (PASO 8)
7. Se clasifica (PASOS 9-11)
8. Se muestra resultado (PASO 12-13)
```

---

## MATEMÁTICAS SIMPLIFICADAS

### Autocovariance
**¿Qué es?** Mide cómo las muestras de audio se correlacionan consigo mismas a diferentes desplazamientos.
**Utilidad:** Captura patrones repetitivos en el audio (periodicidad).

### FFT (Transformada Rápida de Fourier)
**¿Qué es?** Descompone el audio en sus frecuencias componentes.
**Utilidad:** Transforma el dominio temporal al dominio de frecuencias.

### Norma (Magnitud)
**¿Qué es?** El valor absoluto de cada componente de frecuencia.
**Utilidad:** Dice cuánta "energía" hay en cada frecuencia.

### Distancia Manhattan (L1)
**¿Qué es?** Suma de valores absolutos de diferencias.
**Fórmula:** `distance = sum(|vector1 - vector2|) / length`
**Utilidad:** Mide similitud entre dos patrones de espectro.

---

## NOTAS IMPORTANTES

1. **Procesamiento en Tiempo Real**: El audio se procesa mientras se captura, no después.
2. **Buffer Deslizante**: Cada 2 segundos se procesa una nueva ventana, pero puede haber solapamiento conceptual.
3. **Vectores de Referencia**: FM y WN se crean UNA SOLA VEZ del entrenamiento y se reutilizan.
4. **Threading**: Se usa `threading.Lock()` para evitar conflictos cuando múltiples procesos acceden al buffer.
5. **Normalización**: Es crucial para que la comparación sea justa sin importar la amplitud del audio.
6. **Entrenamiento requerido**: Antes de correr `main.py` por primera vez en un entorno nuevo, ejecutar `python -m src.loaders.loaderDataSet` para generar los vectores de referencia en `src/dataSet/`.