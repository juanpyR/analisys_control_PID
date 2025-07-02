import streamlit as st
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from openai import OpenAI

# ingresa tu API
api_model = st.secrets['auth_api']

# configuracion del modelo 
client = OpenAI(
api = api_model ,
base_url="https://api.groq.com/openai/v1"
)
model_name = "gemma2-9b-it"  # modelo de prueba 

def mensaje_IA(prompt):
    messages = [
        {"role": "system", "content": "Eres un experto y un asistente útil para análisis de sistemas PID."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=700,
            temperature=1.0
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al llamar a la API: {e}")
        return None
    
def generate_analysis_from_data(t, y, kp, ki, kd):
    prompt_user = f"""
    Se ha simulado un sistema de control PID en lazo cerrado con los siguientes parámetros PID:
    - **Kp (Proporcional)**: {kp}
    - **Ki (Integral)**: {ki}
    - **Kd (Derivativo)**: {kd}

    Los vectores de tiempo y respuesta son los siguientes:

    - **Tiempo**: {list(t)}
    - **Respuesta**: {list(y)}
Analiza el sistema y responde con un resumen a cada pregunta:
1. ¿Llega a estado estacionario 1? Si no, ¿cuál es la diferencia y el % de error?
2. ¿La respuesta es rápida, lenta o agresiva?
3. ¿Qué ajustes PID recomiendas para una respuesta mejor?
4. ¿Qué harías si hay ruido?
5. ¿Qué efecto tiene cada parámetro?
"""

    text = mensaje_IA(prompt_user)  #respuesta de IA
    return text


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
