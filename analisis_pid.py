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
    
    # Calcular parámetros a partir de la respuesta
    settling_time = t[np.where(np.abs(y - y[-1]) < 0.02 * y[-1])[0][0]]
    rise_time = t[np.where(y >= 0.9 * y[-1])[0][0]]
    overshoot = (max(y) - y[-1]) / y[-1] * 100
    steady_state_error = y[-1] - y[-1]  # Se puede ajustar para otros tipos de error si es necesario
    
    # Crear el prompt para la IA
    prompt = f"""
    Se ha simulado un sistema de control PID en lazo cerrado con los siguientes parámetros PID:
    - **Kp (Proporcional)**: {kp}
    - **Ki (Integral)**: {ki}
    - **Kd (Derivativo)**: {kd}

    Los datos de la respuesta del sistema a un escalón son los siguientes:
    - **Tiempo de Asentamiento**: {settling_time:.2f} segundos
    - **Tiempo de Subida**: {rise_time:.2f} segundos
    - **Sobreimpulso**: {overshoot:.2f}%
    - **Error en Estado Estacionario**: {steady_state_error:.2f}

    Los vectores de tiempo (t) y respuesta (y) son los siguientes:

    - **Tiempo (t)**: {list(t)}
    - **Respuesta (y)**: {list(y)}

    Analiza estos resultados y proporciona recomendaciones para mejorar el rendimiento del sistema. Considera los siguientes puntos en tu análisis:
    1. Evalúa cómo estos parámetros afectan el rendimiento general del sistema.
    2. Sugiere ajustes a los parámetros PID (Kp, Ki, Kd) para mejorar la estabilidad, el tiempo de respuesta y reducir el sobreimpulso.
    3. ¿Qué ajustes serían necesarios si el sistema experimentara más ruido o perturbaciones?
    4. Analiza las implicaciones de cada parámetro para la estabilidad y el comportamiento transitorio del sistema.
    """
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# Función para obtener parámetros clave del gráfico
def analyze_step_response(t, y):
    # Tiempo de asentamiento (settling time) - tiempo donde la respuesta está dentro del 2% del valor final
    settling_time = t[np.where(np.abs(y - y[-1]) < 0.02 * y[-1])[0][0]]
    
    # Tiempo de subida (rise time) - tiempo que tarda en alcanzar el 90% del valor final
    rise_time = t[np.where(y >= 0.9 * y[-1])[0][0]]
    
    # Sobreimpulso (overshoot) - cuánto se excede el valor final en términos porcentuales
    overshoot = (max(y) - y[-1]) / y[-1] * 100
    
    # Error en estado estacionario (si se considera el valor final)
    steady_state_error = y[-1] - y[-1]  # Siempre será cero en un sistema ideal
    return settling_time, rise_time, overshoot, steady_state_error

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

# Análisis de la respuesta
settling_time, rise_time, overshoot, steady_state_error = analyze_step_response(t, y)

# Mostrar el análisis en Streamlit
st.subheader("🔍 Análisis de la Respuesta:")
st.write(f"📊 **Tiempo de Asentamiento (Settling Time):** {settling_time:.2f} segundos")
st.write(f"📊 **Tiempo de Subida (Rise Time):** {rise_time:.2f} segundos")
st.write(f"📊 **Sobreimpulso (Overshoot):** {overshoot:.2f}%")
st.write(f"📊 **Error en Estado Estacionario:** {steady_state_error:.2f}")

#Generar insights 
if st.button("🔍 Generar Análisis desde los Datos"):
    analysis = generate_analysis_from_data(t, y, kp, ki, kd)
    st.subheader("📊 Análisis de la Respuesta del Sistema desde los Datos")
    st.write(analysis)
