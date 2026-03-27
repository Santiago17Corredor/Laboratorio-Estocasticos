import queue
import numpy as np
import sounddevice as sd
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.utils.audioTransformer import AudioTransformer
from src.models.classifier import classify

SR         = 44100
BLOCK_SIZE = 1024

audio_queue = queue.Queue()
transformer = AudioTransformer()


def callback(indata, frames, time, status):
    bloque = indata[:, 0].astype(np.float64)
    audio_queue.put(bloque)
    transformer.agregar(bloque)


class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Reconocedor de audio")
        self.root.configure(bg="#111111")
        self.root.resizable(True, True)

        self.stream = None
        self.fig    = None
        self.canvas = None

        self._construirLayout()
        self._construirGraficaEspera()

    def _construirLayout(self):
        self.panel_izq = tk.Frame(self.root, bg="#111111", padx=20, pady=20)
        self.panel_izq.pack(side="left", fill="y")

        self.panel_der = tk.Frame(self.root, bg="#1a1a1a")
        self.panel_der.pack(side="right", fill="both", expand=True)

        self.btn_iniciar = tk.Button(
            self.panel_izq, text="Empezar a grabar",
            command=self._iniciarGrabacion,
            bg="#1e1e1e", fg="#ffffff",
            activebackground="#2a2a2a", activeforeground="#ffffff",
            relief="flat", padx=12, pady=8,
            font=("Consolas", 10), cursor="hand2", width=18
        )
        self.btn_iniciar.pack(anchor="w", pady=(0, 8))

        self.btn_detener = tk.Button(
            self.panel_izq, text="Detener",
            command=self._detenerGrabacion,
            bg="#1e1e1e", fg="#555555",
            activebackground="#2a2a2a", activeforeground="#ffffff",
            relief="flat", padx=12, pady=8,
            font=("Consolas", 10), cursor="hand2", width=18, state="disabled"
        )
        self.btn_detener.pack(anchor="w", pady=(0, 24))

        tk.Label(self.panel_izq, text="Estado de grabación",
                 bg="#111111", fg="#444444",
                 font=("Consolas", 8)).pack(anchor="w")
        self.lbl_grabacion = tk.Label(
            self.panel_izq, text="Detenido",
            bg="#111111", fg="#555555",
            font=("Consolas", 10), wraplength=180, justify="left"
        )
        self.lbl_grabacion.pack(anchor="w", pady=(2, 20))

        tk.Label(self.panel_izq, text="Suficiencia para comparación",
                 bg="#111111", fg="#444444",
                 font=("Consolas", 8)).pack(anchor="w")
        self.lbl_suficiencia = tk.Label(
            self.panel_izq, text="—",
            bg="#111111", fg="#555555",
            font=("Consolas", 10), wraplength=180, justify="left"
        )
        self.lbl_suficiencia.pack(anchor="w", pady=(2, 20))

        tk.Label(self.panel_izq, text="Clasificación",
                 bg="#111111", fg="#444444",
                 font=("Consolas", 8)).pack(anchor="w")
        self.lbl_resultado = tk.Label(
            self.panel_izq, text="—",
            bg="#111111", fg="#555555",
            font=("Consolas", 12, "bold"), wraplength=180, justify="left"
        )
        self.lbl_resultado.pack(anchor="w", pady=(2, 0))
    def _construirGraficaEspera(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 4))
        self._estiloAx(self.ax)
        self.ax.set_title("Esperando grabación...", color="#444444", fontsize=10)
        self.fig.tight_layout()
        self._montarCanvas()

    def _construirGraficaLive(self):
        plt.close(self.fig)
        self.fig, self.ax = plt.subplots(figsize=(9, 4))
        self._estiloAx(self.ax)
        self.ax.set_title("Espectro acumulado — micrófono", color="#888888", fontsize=10)
        self.ax.set_xlabel("Frecuencia (Hz)", color="#555555", fontsize=9)
        self.ax.set_ylabel("Magnitud normalizada", color="#555555", fontsize=9)
        self.ax.set_xlim(0, SR / 2)
        self.ax.set_ylim(0, 1.1)
        self.line_live, = self.ax.plot([], [], color="#ff6f61", linewidth=1.0, label="Micrófono")
        self.ax.legend(facecolor="#1a1a1a", edgecolor="#333333",
                       labelcolor="#cccccc", fontsize=8)
        self.fig.tight_layout()
        self._montarCanvas()

    def _mostrarComparativa(self, resultado, dist_fm, dist_wn,
                             mic_n, fm_n, wn_n, mic_acov, fm_acov, wn_acov):
        plt.close(self.fig)

        self.fig, (ax_acov, ax_spec) = plt.subplots(
            2, 1, figsize=(9, 7), facecolor="#1a1a1a"
        )

        color_titulo = "#4fc3f7" if resultado == "FM" else "#aed581"
        alpha_fm = 0.9 if resultado == "FM" else 0.3
        alpha_wn = 0.9 if resultado == "Ruido Blanco" else 0.3

        self._estiloAx(ax_acov)
        ax_acov.set_title("Autocovarianza", color="#888888", fontsize=10)
        ax_acov.set_xlabel("Lag (muestras)", color="#555555", fontsize=9)
        ax_acov.set_ylabel("Autocovarianza", color="#555555", fontsize=9)

        size_acov = len(mic_acov)
        lags      = np.arange(size_acov)

        view = min(2000, size_acov)
        ax_acov.plot(lags[:view], fm_acov[:view],
                     color="#4fc3f7", linewidth=1.0, alpha=alpha_fm, label="FM")
        ax_acov.plot(lags[:view], wn_acov[:view],
                     color="#aed581", linewidth=1.0, alpha=alpha_wn, label="Ruido Blanco")
        ax_acov.plot(lags[:view], mic_acov[:view],
                     color="#ff6f61", linewidth=1.0, alpha=0.9, label="Micrófono")
        ax_acov.legend(facecolor="#1a1a1a", edgecolor="#333333",
                       labelcolor="#cccccc", fontsize=8)

        self._estiloAx(ax_spec)
        ax_spec.set_title(
            f"Espectro — Resultado: {resultado}  |  d(FM)={dist_fm:.5f}  d(WN)={dist_wn:.5f}",
            color=color_titulo, fontsize=10, fontweight="bold"
        )
        ax_spec.set_xlabel("Frecuencia (Hz)", color="#555555", fontsize=9)
        ax_spec.set_ylabel("Magnitud normalizada", color="#555555", fontsize=9)
        ax_spec.set_xlim(0, 1000)

        size_spec = min(len(mic_n), len(fm_n), len(wn_n))
        freqs     = np.linspace(0, SR / 2, size_spec)

        ax_spec.plot(freqs, fm_n[:size_spec],
                     color="#4fc3f7", linewidth=1.0, alpha=alpha_fm,
                     label=f"FM  (d={dist_fm:.5f})")
        ax_spec.plot(freqs, wn_n[:size_spec],
                     color="#aed581", linewidth=1.0, alpha=alpha_wn,
                     label=f"Ruido Blanco  (d={dist_wn:.5f})")
        ax_spec.plot(freqs, mic_n[:size_spec],
                     color="#ff6f61", linewidth=1.0, alpha=0.9, label="Micrófono")
        ax_spec.legend(facecolor="#1a1a1a", edgecolor="#333333",
                       labelcolor="#cccccc", fontsize=8)

        self.fig.tight_layout(pad=2.5)
        self._montarCanvas()

    def _estiloAx(self, ax):
        self.fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#444444", labelsize=8)
        ax.spines[:].set_color("#2a2a2a")
        ax.grid(True, color="#222222", linewidth=0.5)

    def _montarCanvas(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.panel_der)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)
        self.canvas.draw()

    def _iniciarGrabacion(self):
        transformer.reset()

        self.btn_iniciar.config(state="disabled")
        self.btn_detener.config(state="normal", fg="#ffffff")
        self.lbl_grabacion.config(text="🔴  Grabando...", fg="#e74c3c")
        self.lbl_suficiencia.config(text="Esperando 2 seg...", fg="#888888")
        self.lbl_resultado.config(text="—", fg="#555555")

        self._construirGraficaLive()

        self.stream = sd.InputStream(
            samplerate=SR, blocksize=BLOCK_SIZE,
            channels=1, dtype="float32", callback=callback
        )
        self.stream.start()

        self._ultimo_fragmento = 0
        self._pollFragmento()

    def _pollFragmento(self):
        if self.stream is None:
            return

        n = transformer.cantidadFragmentos()

        if n >= 1:
            self.lbl_suficiencia.config(
                text=f"✅  Listo ({n} fragmento{'s' if n > 1 else ''} de 2 seg)",
                fg="#2ecc71"
            )
        else:
            self.lbl_suficiencia.config(text="Esperando 2 seg...", fg="#888888")

        if n > self._ultimo_fragmento:
            self._ultimo_fragmento = n
            self._actualizarEspectroLive()

        self.root.after(500, self._pollFragmento)

    def _actualizarEspectroLive(self):
        avg  = transformer.promedioActual()
        norm = avg["norm"]
        if len(norm) == 0:
            return

        freqs = np.linspace(0, SR / 2, len(norm))
        max_v = np.max(norm) + 1e-10
        self.line_live.set_data(freqs, norm / max_v)
        self.canvas.draw_idle()

    def _detenerGrabacion(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.btn_detener.config(state="disabled")
        self.lbl_grabacion.config(text="⚙  Clasificando...", fg="#f39c12")
        self.root.update()

        try:
            transformer.detener()
            resultado, dist_fm, dist_wn, mic_n, fm_n, wn_n, mic_acov, fm_acov, wn_acov = classify()

            color = "#4fc3f7" if resultado == "FM" else "#aed581"
            self.lbl_grabacion.config(text="Detenido", fg="#555555")
            self.lbl_resultado.config(text=resultado, fg=color)

            self._mostrarComparativa(resultado, dist_fm, dist_wn,
                                     mic_n, fm_n, wn_n,
                                     mic_acov, fm_acov, wn_acov)
        except RuntimeError as e:
            self.lbl_grabacion.config(text=f"{e}", fg="#e74c3c")
        finally:
            self.btn_iniciar.config(state="normal")


def launch():
    root = tk.Tk()
    App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: root.quit())
    root.mainloop()
    root.destroy()