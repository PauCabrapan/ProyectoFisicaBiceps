import cv2
import mediapipe as mp
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle 

# Inicialización de MediaPipe Pose
mp_pose = mp.solutions.pose

# Configuración inicial
video_path = 'videoGym.mp4'
if not os.path.exists(video_path):
    print(f"Error: Archivo de video no encontrado en {video_path}")
    exit()

lado = 'derecho'  # 'izquierdo' o 'derecho'
puntos = {
    'izquierdo': {'hombro': 11, 'codo': 13, 'muneca': 15, 'cadera': 23},
    'derecho': {'hombro': 12, 'codo': 14, 'muneca': 16, 'cadera': 24}
}

umbral_visibilidad = 0.2
altura_real = 1.75
longitud_real_antebrazo = 0.227

# Variables para los vectores direccional
vector_history = []
SMOOTHING_WINDOW = 5  # Frames para suavizado de los vectores
VECTOR_SCALE = 100    # Longitud visual del vector en píxeles

# Estructuras de datos
datos_completos = []
datos_visibles = []

# Parámetros para la aceleración dinámica
RECORRIDO_TOTAL_PX = 200  # Recorrido total en píxeles
SMOOTHING_ACCELERATION = True  # Habilitar suavizado de la aceleración
ACCELERATION_VALUES = []  # Historial de aceleraciones para suavizado

# Procesamiento del video
cap = cv2.VideoCapture(video_path)
initial_position = None  # Posición inicial del brazo (se inicializará en el primer frame)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
) as pose:

    frame_id = 0
    h, w = None, None
    landmarks_frame = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            landmarks_frame = landmarks
            idxs = puntos[lado]

            if all(landmarks[i].visibility > umbral_visibilidad for i in idxs.values()):
                hombro = landmarks[idxs['hombro']]
                codo = landmarks[idxs['codo']]
                muneca = landmarks[idxs['muneca']]
                cadera = landmarks[idxs['cadera']]

                xh, yh = int(hombro.x * w), int(hombro.y * h)
                xc, yc = int(codo.x * w), int(codo.y * h)
                xm, ym = int(muneca.x * w), int(muneca.y * h)
                xca, yca = int(cadera.x * w), int(cadera.y * h)

                # Dibujar articulaciones y conexiones
                color = (0, 255, 255) if lado == 'izquierdo' else (255, 100, 100)
                cv2.line(frame, (xh, yh), (xc, yc), color, 5)
                cv2.line(frame, (xc, yc), (xm, ym), color, 5)
                cv2.line(frame, (xh, yh), (xca, yca), color, 5)
                
                for point in [(xh, yh), (xc, yc), (xm, ym), (xca, yca)]:
                    cv2.circle(frame, point, 9, (0, 255, 0), -1)

                # -----------------------------------------------
                # CÁLCULO Y DIBUJO DE LOS VECTORES
                # -----------------------------------------------
                if frame_id > 0 and len(datos_visibles) > 0:
                    # Obtener posición anterior de la muñeca
                    prev_data = datos_visibles[-1]
                    
                    # Calcular vector de movimiento (diferencia entre frames)
                    dx = xm - prev_data['muneca_x']
                    dy = ym - prev_data['muneca_y']
                    
                    # Calcular distancia recorrida
                    distance_moved = np.sqrt(dx**2 + dy**2)
                    
                    # Obtener la posición inicial del brazo (primer frame)
                    if initial_position is None:
                        initial_position = ym  # Asumimos movimiento vertical
                    
                    # Calcular el recorrido total y el progreso actual
                    if initial_position <= ym:
                        displacement = abs(ym - initial_position)
                    else:
                        displacement = abs(initial_position - ym)
                    
                    total_displacement = RECORRIDO_TOTAL_PX
                    progress = displacement / total_displacement
                    
                    # Determinar la aceleración dinámicamente
                    if progress <= 0.5:
                        # Primera mitad del recorrido: Aceleración positiva
                        acceleration = 2.0  # Valor ajustable
                    else:
                        # Segunda mitad del recorrido: Desaceleración
                        if displacement > 0:
                            acceleration = -2.0  # Valor ajustable
                    
                    # Suavizar la aceleración utilizando un promedio móvil
                    if SMOOTHING_ACCELERATION:
                        ACCELERATION_VALUES.append(acceleration)
                        if len(ACCELERATION_VALUES) > SMOOTHING_WINDOW:
                            ACCELERATION_VALUES.pop(0)
                        smoothed_acceleration = np.mean(ACCELERATION_VALUES)
                    else:
                        smoothed_acceleration = acceleration
                    
                    # Actualizar aceleración suavizada
                    aceleracion_actual = smoothed_acceleration
                    
                    # Normalizar los vectores para dirección contrario
                    length_velocity = np.sqrt((dx**2 + dy**2))
                    length_acceleration = np.sqrt((dx**2 + dy**2))
                    
                    if length_velocity > 0 and length_acceleration > 0:
                        # Vector de velocidad (VERDE)
                        opposite_velocity_dx = dx / length_velocity
                        opposite_velocity_dy = dy / length_velocity
                        
                        # Vector de aceleración (ROJO)
                        opposite_acceleration_dx = dx / length_acceleration
                        opposite_acceleration_dy = dy / length_acceleration
                        
                        # Aplicar suavizado al vector direccional
                        vector_history.append((opposite_velocity_dx, opposite_velocity_dy))
                        if len(vector_history) > SMOOTHING_WINDOW:
                            vector_history.pop(0)
                        
                        avg_dx = sum(v[0] for v in vector_history) / len(vector_history)
                        avg_dy = sum(v[1] for v in vector_history) / len(vector_history)
                        
                        # Normalizar el vector promedio
                        avg_length = np.sqrt(avg_dx**2 + avg_dy**2)
                        if avg_length > 0:
                            avg_dx /= avg_length
                            avg_dy /= avg_length
                            
                            # Punto final del vector de velocidad
                            end_point_velocity = (
                                int(xm + avg_dx * VECTOR_SCALE),
                                int(ym + avg_dy * VECTOR_SCALE)
                            )
                            
                            # Punto final del vector de aceleración (más corto)
                            end_point_acceleration = (
                                int(xm + avg_dx * (VECTOR_SCALE * 0.8)),
                                int(ym + avg_dy * (VECTOR_SCALE * 0.8))
                            )
                            
                            # Dibujar vector de velocidad (VERDE)
                            cv2.arrowedLine(
                                frame,
                                (xm, ym),
                                end_point_velocity,
                                (0, 255, 0),
                                3,
                                tipLength=0.3
                            )
                            
                            # Dibujar vector de aceleración (ROJO)
                            cv2.arrowedLine(
                                frame,
                                (xm, ym),
                                end_point_acceleration,
                                (0, 0, 255),
                                3,
                                tipLength=0.3
                            )
    
                    # Mostrar texto para velocidad y aceleración
                    cv2.putText(
                        frame,
                        f"Velocidad: {math.hypot(dx, dy):.2f} px/frame",
                        (xm + 20, ym + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,  # Aumentamos el tamaño de la fuente
                        (255, 255, 255),
                        3  # Aumentamos el grosor del texto
                    )
                    cv2.putText(
                        frame,
                        f"Aceleración: {aceleracion_actual:.2f} m/s2",
                        (xm + 20, ym + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,  # Aumentamos el tamaño de la fuente
                        (255, 255, 255),
                        3  # Aumentamos el grosor del texto
                    )

                # Almacenar datos visibles
                datos_visibles.append({
                    'frame': frame_id,
                    'hombro_x': xh, 'hombro_y': yh,
                    'codo_x': xc, 'codo_y': yc,
                    'muneca_x': xm, 'muneca_y': ym,
                    'cadera_x': xca, 'cadera_y': yca
                })

            # Almacenar todos los datos
            fila_completa = {'frame': frame_id}
            for i, lm in enumerate(landmarks):
                fila_completa[f'P{i}_x'] = int(lm.x * w)
                fila_completa[f'P{i}_y'] = int(lm.y * h)
                fila_completa[f'P{i}_vis'] = lm.visibility
            datos_completos.append(fila_completa)

        # Redimensionar y mostrar frame
        alto_deseado = 720
        proporcion = alto_deseado / frame.shape[0]
        ancho_escalado = int(frame.shape[1] * proporcion)
        resized = cv2.resize(frame, (ancho_escalado, alto_deseado))

        cv2.imshow("Seguimiento del brazo con vectores de velocidad y aceleración", resized)
        if cv2.waitKey(10) & 0xFF == ord('e'):
            break

        frame_id += 1

cap.release()
cv2.destroyAllWindows()

# Post-procesamiento de datos
df_completo = pd.DataFrame(datos_completos)
df_visibles = pd.DataFrame(datos_visibles)

if landmarks_frame:
    codo = landmarks_frame[puntos[lado]['codo']]
    muneca = landmarks_frame[puntos[lado]['muneca']]
    antebrazo_px = math.sqrt((codo.x - muneca.x)**2 + (codo.y - muneca.y)**2) * h
    factor = longitud_real_antebrazo / antebrazo_px
    print(f"\nFactor de conversión (m/px) usando antebrazo: {factor:.6f}")

    def convertir_a_metros(df, factor):
        df_m = df.copy()
        for col in df.columns:
            if '_x' in col or '_y' in col:
                df_m[col] = df[col] * factor
        return df_m

    df_visibles_metros = convertir_a_metros(df_visibles, factor)

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 1 else 30
    dt = 1 / fps

    df_derivadas = df_visibles_metros.copy()
    for col in df_visibles_metros.columns:
        if '_x' in col or '_y' in col:
            df_derivadas[f'{col}_vel'] = df_visibles_metros[col].diff() / dt
            df_derivadas[f'{col}_acc'] = df_derivadas[f'{col}_vel'].diff() / dt

    # Suavizado de datos
    ventana = 5
    df_visibles_metros_suav = df_visibles_metros.copy()
    df_derivadas_suav = df_derivadas.copy()
    
    for col in df_visibles_metros.columns:
        if col != 'frame':
            df_visibles_metros_suav[col] = df_visibles_metros[col].rolling(
                window=ventana, center=True).mean()

    for col in df_derivadas.columns:
        if col != 'frame':
            df_derivadas_suav[col] = df_derivadas[col].rolling(
                window=ventana, center=True).mean()

    # Crear DataFrame unificado
    articulaciones = ['hombro', 'codo', 'muneca', 'cadera']
    columnas_unificado = ['frame']
    
    for articulacion in articulaciones:
        columnas_unificado.extend([
            f'{articulacion}_x', f'{articulacion}_y',
            f'{articulacion}_x_vel', f'{articulacion}_y_vel',
            f'{articulacion}_x_acc', f'{articulacion}_y_acc'
        ])
    
    columnas_existentes = [col for col in columnas_unificado if col in df_derivadas_suav.columns]
    df_unificado = df_derivadas_suav[columnas_existentes].copy()
    
    # Guardar resultados
    carpeta_datos = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datos')
    os.makedirs(carpeta_datos, exist_ok=True)
    
    ruta_pickle = os.path.join(carpeta_datos, 'datos_articulaciones.pkl')
    df_unificado.to_pickle(ruta_pickle)
    
    ruta_csv = os.path.join(carpeta_datos, 'datos_articulaciones.csv')
    df_unificado.to_csv(ruta_csv, index=False)

    print(f"\nResultados guardados en {carpeta_datos}:")
    print(f"- datos_articulaciones.pkl (DataFrame completo)")
    print(f"- datos_articulaciones.csv (Versión CSV)")

    # Generación de gráficos (opcional)
    puntos_seguir = ['hombro', 'codo', 'muneca', 'cadera']
    
    #graficos para X
    for punto in puntos_seguir:
        fig_x, axs_x = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig_x.suptitle(f'Movimiento de {punto.capitalize()} en eje X')
        
        axs_x[0].plot(df_unificado['frame'], df_unificado[f'{punto}_x'], 
                     label='Posición', color='blue')
        axs_x[0].set_ylabel('Posición (m)')
        axs_x[0].legend()
        
        axs_x[1].plot(df_unificado['frame'], df_unificado[f'{punto}_x_vel'], 
                     label='Velocidad', color='green')
        axs_x[1].set_ylabel('Velocidad (m/s)')
        axs_x[1].legend()
        
        axs_x[2].plot(df_unificado['frame'], df_unificado[f'{punto}_x_acc'], 
                     label='Aceleración', color='red')
        axs_x[2].set_xlabel('Frame')
        axs_x[2].set_ylabel('Aceleración (m/s2)')
        axs_x[2].legend()
        
        plt.tight_layout()
        ruta_grafico = os.path.join(carpeta_datos, f'movimiento_{punto}_X.png')
        plt.savefig(ruta_grafico)
        plt.close()

         # Gráfico para Y
        fig_y, axs_y = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig_y.suptitle(f'{punto.capitalize()} movimiento en Y')

        axs_y[0].plot(df_visibles_metros_suav['frame'], df_visibles_metros_suav[f'{punto}_y'], label='Posición Y (m)', color='purple')
        axs_y[0].set_ylabel('Posición (m)')
        axs_y[0].legend()

        axs_y[1].plot(df_visibles_metros['frame'], df_derivadas_suav[f'{punto}_y_vel'], label='Velocidad Y (m/s)', color='orange')
        axs_y[1].set_ylabel('Velocidad (m/s)')
        axs_y[1].legend()

        axs_y[2].plot(df_visibles_metros['frame'], df_derivadas_suav[f'{punto}_y_acc'], label='Aceleración Y (m/s²)', color='brown')
        axs_y[2].set_xlabel('Frame')
        axs_y[2].set_ylabel('Aceleración (m/s²)')
        axs_y[2].legend()

        plt.tight_layout()
        ruta_grafico_y = os.path.join(carpeta_datos, f'grafico_{punto}_Y.png')
        plt.savefig(ruta_grafico_y)
        plt.close()

print("\nProceso completado exitosamente!")