import streamlit as st
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import google.generativeai as genai

# configuracion del modelo 
api = st.secrets['auth_api']
genai.configure(api_key= api)

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 500,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)

# Enviar esta información a la IA (Gemini) para un análisis detallado
def generate_analysis_from_data(t, y, kp, ki, kd):
    """Genera un análisis del sistema usando IA basado en los datos de la respuesta y el tiempo."""
  
    # Crear el prompt para la IA
    prompt = f"""
    Se ha simulado un sistema de control PID en lazo cerrado con los siguientes parámetros PID:
    - **Kp (Proporcional)**: {kp}
    - **Ki (Integral)**: {ki}
    - **Kd (Derivativo)**: {kd}

    Los vectores de tiempo (t) y respuesta (y) son los siguientes:

    - **Tiempo (t)**: {list(t)}
    - **Respuesta (y)**: {list(y)}

    Analiza estos resultados y proporciona recomendaciones para mejorar el rendimiento del sistema. Considera los siguientes puntos en tu análisis, recuerda eres un ingeniero experto en automatización:
    1. Evalúa cómo estos parámetros afectan el rendimiento general del sistema, ojo si el sistema no alcanza el estado estacionario 1 debes mostrar la diferencia y explicar el porqué no está alcanzado (1 debe llegar debes compararlo con los últimos valores del arreglo y). además
    si el valor llega en estado estacionario 1 indica si hay una respuesta rápida, lenta o agresiva debes mencionarla ya que son muy notorias cuando ocurren, de lo contrario el análisis debe ser cuando el valor en estado estacionario no llega a 1, y recuerda darme la diferencia
    entre el valor 1 y el valor actual en estado estacionario y dame el % de error. 
    2. Sugiere ajustes a los parámetros PID (Kp, Ki, Kd) para mejorar la estabilidad, el tiempo de respuesta y reducir el sobreimpulso.
    En lo posible recomienda valores de (Kp, Ki, Kd) para respuestas rápidas, más lentas y agresivas del sistema, ojo deben estar en el rango propuesto de valores
    3. ¿Qué ajustes serían necesarios si el sistema experimentara más ruido o perturbaciones?
    4. Analiza las implicaciones de cada parámetro para la estabilidad y el comportamiento transitorio del sistema.
    """
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# Interfaz en Streamlit
st.title("⚙️ Análisis de Control PID")
st.write("Ajusta los parámetros y visualiza la respuesta del sistema.")

# Parámetros del Controlador PID
kp = st.slider("Kp (Proporcional)", 0.0, 10.0, 1.0)
ki = st.slider("Ki (Integral)", 0.0, 5.0, 0.0)
kd = st.slider("Kd (Derivativo)", 0.0, 5.0, 0.0)

# Definir numerador y denominador del Controlador PID
num_C = [kd, kp, ki]  # Numerador del controlador
den_C = [1]  # Denominador trivial del controlador

# Definir numerador y denominador de la Planta
num_P = [1]  # Numerador de la planta
den_P = [1, 3, 2, 0]  # Denominador de la planta

# Función de transferencia en lazo abierto G(s) = C(s) * P(s)
num_G = np.polymul(num_C, num_P)  # Multiplicación de numeradores
den_G = np.polymul(den_C, den_P)  # Multiplicación de denominadores

# Función de transferencia en lazo cerrado H(s) = G(s) / (1 + G(s))
num_H = num_G  # El numerador de H(s) es el mismo de G(s)
den_H = np.polyadd(den_G, num_G)  # El denominador es den_G + num_G
system_H = signal.TransferFunction(num_H, den_H)  # Sistema en lazo cerrado

# Crear columnas para mostrar ecuaciones
col1, col2 = st.columns(2)

# Mostrar el Controlador en la columna izquierda
with col1:
    st.subheader("🎛️ Controlador PID")
    latex_C = fr'''
        C(s) = {kd} s^2 + {kp} s + {ki}
    '''
    st.latex(latex_C)

# Mostrar la Planta en la columna derecha
with col2:
    st.subheader("🌱 Planta del Sistema")
    latex_P = r'''
        P(s) = \frac{1}{s^3 + 3s^2 + 2s}
    '''
    st.latex(latex_P)

# Mostrar la función de transferencia en lazo cerrado
st.subheader("🔒 Función de Transferencia en Lazo Cerrado")
latex_H = fr'''
    H(s) = \frac{{G(s)}}{{1 + G(s)}} = 
    \frac{{{kd} s^2 + {kp} s + {ki}}}{{s^3 + {3 + kd} s^2 + {round(2 + kp,2)} s + {ki}}}
'''
st.latex(latex_H)

# Simulación de la respuesta en lazo cerrado
t, y = signal.step(system_H)  # Respuesta en lazo cerrado

# Graficar las respuestas en lazo cerrado
fig, ax = plt.subplots()
ax.plot(t, y, label="Respuesta en lazo cerrado", color="blue")
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Salida")
ax.set_title("Respuesta a una entrada escalón")
ax.legend()
st.pyplot(fig)

#Generar insights 
if st.button("🔍 Generar Análisis desde los Datos"):
    analysis = generate_analysis_from_data(t, y, kp, ki, kd)
    st.subheader("📊 Análisis de la Respuesta del Sistema desde los Datos")
    st.write(analysis)
