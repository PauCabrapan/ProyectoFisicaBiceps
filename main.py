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

umbral_visibilidad = 0.1
altura_real = 1.75
longitud_real_antebrazo = 0.227

# Variables para visualización
VECTOR_SCALE = 150 #Agrande un pooco los vectores
MIN_VEL_THRESHOLD = 0.001  # Umbral mínimo para dibujar el vector de velocidad
MIN_ACC_THRESHOLD = 0.005  # Umbral mínimo para dibujar el vector de aceleración
ventana = 7  # Definimos ventana aquí para usarla en el suavizado en tiempo real

# Estructuras de datos
datos_completos = []
datos_visibles = []
previous_position = None
previous_velocity = [0, 0]
factor = None
smoothed_acc_x = 0  # Aceleración suavizada en x
smoothed_acc_y = 0  # Aceleración suavizada en y

# Procesamiento del video en una sola pasada
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 1 else 30
dt = 1 / fps

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
    first_visible_frame = True

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

                # Calcular factor de conversión en el primer frame visible
                if first_visible_frame:
                    try:
                        codo = landmarks_frame[puntos[lado]['codo']]
                        muneca = landmarks_frame[puntos[lado]['muneca']]
                        antebrazo_px = math.sqrt((codo.x - muneca.x)**2 + (codo.y - muneca.y)**2) * h
                        if antebrazo_px > 10:
                            factor = longitud_real_antebrazo / antebrazo_px
                            print(f"Factor de conversión (m/px) usando antebrazo: {factor:.6f}")
                        else:
                            print("Longitud del antebrazo detectada es inválida. Usando factor por defecto de 0.001")
                            factor = 0.001  # Valor por defecto si falla
                        first_visible_frame = False
                    except (KeyError, AttributeError, TypeError) as e:
                        print(f"Error al calcular el factor de conversión: {e}. Usando factor por defecto de 0.001")
                        factor = 0.001  # Valor por defecto si falla

                # Calcular velocidad y aceleración en tiempo real
                vel_x = vel_y = acc_x = acc_y = 0
                if previous_position is not None and factor is not None:
                    dx = (xm - previous_position[0]) * factor
                    dy = (ym - previous_position[1]) * factor
                    vel_x = dx / dt  # Velocidad en m/s
                    vel_y = dy / dt
                    acc_x = (vel_x - previous_velocity[0]) / dt if frame_id > 0 else 0
                    acc_y = (vel_y - previous_velocity[1]) / dt if frame_id > 0 else 0
                    previous_velocity = [vel_x, vel_y]

                    # Suavizado simple de la aceleración (promedio móvil básico)
                    smoothed_acc_x = (smoothed_acc_x * (ventana - 1) + acc_x) / ventana if frame_id > 0 else acc_x
                    smoothed_acc_y = (smoothed_acc_y * (ventana - 1) + acc_y) / ventana if frame_id > 0 else acc_y

                previous_position = (xm, ym)

                print(f"Frame {frame_id}: Posición (xm, ym) = ({xm}, {ym}), Velocidad (x, y) = ({vel_x:.6f}, {vel_y:.6f}), Aceleración suavizada (x, y) = ({smoothed_acc_x:.6f}, {smoothed_acc_y:.6f})")

                # Normalizar y dibujar vector de velocidad (VERDE)
                vel_length = math.sqrt(vel_x**2 + vel_y**2)
                if vel_length > MIN_VEL_THRESHOLD:
                    norm_vel_x = vel_x / vel_length
                    norm_vel_y = vel_y / vel_length
                    end_point_velocity = (
                        int(xm + norm_vel_x * VECTOR_SCALE),
                        int(ym + norm_vel_y * VECTOR_SCALE)
                    )
                    end_point_velocity = (
                        max(0, min(end_point_velocity[0], w)),
                        max(0, min(end_point_velocity[1], h))
                    )
                    cv2.arrowedLine(
                        frame,
                        (xm, ym),
                        end_point_velocity,
                        (0, 255, 0),
                        3,
                        tipLength=0.3
                    )
                    print(f"Dibujando vector de velocidad en ({xm}, {ym}) -> ({end_point_velocity[0]}, {end_point_velocity[1]})")

                # Normalizar y dibujar vector de aceleración suavizada (ROJO)
                acc_length = math.sqrt(smoothed_acc_x**2 + smoothed_acc_y**2)
                if acc_length > MIN_ACC_THRESHOLD:
                    norm_acc_x = smoothed_acc_x / acc_length
                    norm_acc_y = smoothed_acc_y / acc_length
                    end_point_acceleration = (
                        int(xm + norm_acc_x * (VECTOR_SCALE * 0.8)),
                        int(ym + norm_acc_y * (VECTOR_SCALE * 0.8))
                    )
                    end_point_acceleration = (
                        max(0, min(end_point_acceleration[0], w)),
                        max(0, min(end_point_acceleration[1], h))
                    )
                    cv2.arrowedLine(
                        frame,
                        (xm, ym),
                        end_point_acceleration,
                        (0, 0, 255),
                        3,
                        tipLength=0.3
                    )
                    print(f"Dibujando vector de aceleración suavizada en ({xm}, {ym}) -> ({end_point_acceleration[0]}, {end_point_acceleration[1]})")

                # Mostrar texto siempre
                cv2.putText(
                    frame,
                    f"Velocidad: {vel_length:.6f} m/s",
                    (xm + 20, ym + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    3
                )
                cv2.putText(
                    frame,
                    f"Aceleración: {acc_length:.6f} m/s2",
                    (xm + 20, ym + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    3
                )
                 #Diagrama de cuerpo libre: 
                
                # Primero se calcula el centro de masa y longitudes de las flechas
                cm_x = int((xc + xm) / 2) #Aproximacion del centro de masa del antebrazo al punto medio entre la articulación del codo y la muñeca
                cm_y = int((yc + ym) / 2)
                mg_px_len = VECTOR_SCALE // 2 #Longitudes en píxeles de cada flecha (fuerza peso, muscular y reacción)
                fm_px_len = VECTOR_SCALE // 2
                R_px_len  = VECTOR_SCALE // 2

                # Realiza la flecha Peso mg → flecha hacia abajo desde el centro de masa
                end_mg = (cm_x, cm_y + mg_px_len) 
                cv2.arrowedLine(frame, (cm_x,cm_y), end_mg, (0,255,255), 3, tipLength=0.3)
                cv2.putText(frame, 'mg',
                            (cm_x-10, cm_y+mg_px_len+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                # Realiza la flecha de la Fuerza muscular Fm desde el codo hacia el hombro
                vec_mx, vec_my = xh-xc, yh-yc
                mag_m = math.hypot(vec_mx, vec_my)
                if mag_m > 0:
                    ux, uy = vec_mx/mag_m, vec_my/mag_m
                    end_fm = (xc + int(ux*fm_px_len), yc + int(uy*fm_px_len))
                    cv2.arrowedLine(frame, (xc,yc), end_fm, (255,255,0), 3, tipLength=0.3)
                    cv2.putText(frame, 'Fm',
                                (end_fm[0]+5, end_fm[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

                    # Realiza la flecha de Reacción articular R = –(Fm + mg)
                    mg_vec = np.array([0, mg_px_len])
                    fm_vec = np.array([ux*fm_px_len, uy*fm_px_len])
                    R_vec  = -(fm_vec + mg_vec)
                    mag_R  = np.linalg.norm(R_vec)
                    if mag_R > 0:
                        Rx, Ry = R_vec/mag_R
                        end_R = (xc + int(Rx*R_px_len), yc + int(Ry*R_px_len))
                        cv2.arrowedLine(frame, (xc,yc), end_R, (200,200,200), 3, tipLength=0.3)
                        cv2.putText(frame, 'R',
                                    (end_R[0]+5, end_R[1]+5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)

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
            if results.pose_landmarks:
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

        cv2.imshow("Seguimiento del brazo con vectores", resized)
        if cv2.waitKey(10) & 0xFF == ord('e'):
            break

        frame_id += 1

cap.release()
cv2.destroyAllWindows()

# Post-procesamiento de datos
df_completo = pd.DataFrame(datos_completos)
df_visibles = pd.DataFrame(datos_visibles)

if factor is not None:
    def convertir_a_metros(df, factor):
        df_m = df.copy()
        for col in df.columns:
            if '_x' in col or '_y' in col:
                df_m[col] = df[col] * factor
        return df_m

    df_visibles_metros = convertir_a_metros(df_visibles, factor)

    print(f"Total de frames visibles: {len(df_visibles_metros)}")
    if len(df_visibles_metros) < 2:
        print("No hay suficientes datos para calcular derivadas.")
    else:
        df_derivadas = df_visibles_metros.copy()
        for col in df_visibles_metros.columns:
            if '_x' in col or '_y' in col:
                df_derivadas[f'{col}_vel'] = df_visibles_metros[col].diff() / dt
                df_derivadas[f'{col}_acc'] = df_derivadas[f'{col}_vel'].diff() / dt

        df_derivadas.fillna(0, inplace=True)

        ventana = 3
        df_visibles_metros_suav = df_visibles_metros.copy()
        df_derivadas_suav = df_derivadas.copy()
        
        for col in df_visibles_metros.columns:
            if col != 'frame':
                df_visibles_metros_suav[col] = df_visibles_metros[col].rolling(
                    window=ventana, center=True, min_periods=1).mean()

        for col in df_derivadas.columns:
            if col != 'frame':
                df_derivadas_suav[col] = df_derivadas[col].rolling(
                    window=ventana, center=True, min_periods=1).mean()

        df_derivadas_suav.fillna(0, inplace=True)

        # Agregar columnas en coordenadas polares
        articulaciones = ['hombro', 'codo', 'muneca', 'cadera']
        for articulacion in articulaciones:
            # Posición
            df_derivadas[f'{articulacion}_mag'] = np.sqrt(
                df_derivadas[f'{articulacion}_x']**2 + df_derivadas[f'{articulacion}_y']**2
            )
            df_derivadas[f'{articulacion}_ang'] = np.arctan2(
                df_derivadas[f'{articulacion}_y'], df_derivadas[f'{articulacion}_x']
            )
            df_derivadas_suav[f'{articulacion}_mag'] = np.sqrt(
                df_derivadas_suav[f'{articulacion}_x']**2 + df_derivadas_suav[f'{articulacion}_y']**2
            )
            df_derivadas_suav[f'{articulacion}_ang'] = np.arctan2(
                df_derivadas_suav[f'{articulacion}_y'], df_derivadas_suav[f'{articulacion}_x']
            )
            # Velocidad
            df_derivadas[f'{articulacion}_vel_mag'] = np.sqrt(
                df_derivadas[f'{articulacion}_x_vel']**2 + df_derivadas[f'{articulacion}_y_vel']**2
            )
            df_derivadas[f'{articulacion}_vel_ang'] = np.arctan2(
                df_derivadas[f'{articulacion}_y_vel'], df_derivadas[f'{articulacion}_x_vel']
            )
            df_derivadas_suav[f'{articulacion}_vel_mag'] = np.sqrt(
                df_derivadas_suav[f'{articulacion}_x_vel']**2 + df_derivadas_suav[f'{articulacion}_y_vel']**2
            )
            df_derivadas_suav[f'{articulacion}_vel_ang'] = np.arctan2(
                df_derivadas_suav[f'{articulacion}_y_vel'], df_derivadas_suav[f'{articulacion}_x_vel']
            )
            # Aceleración
            df_derivadas[f'{articulacion}_acc_mag'] = np.sqrt(
                df_derivadas[f'{articulacion}_x_acc']**2 + df_derivadas[f'{articulacion}_y_acc']**2
            )
            df_derivadas[f'{articulacion}_acc_ang'] = np.arctan2(
                df_derivadas[f'{articulacion}_y_acc'], df_derivadas[f'{articulacion}_x_acc']
            )
            df_derivadas_suav[f'{articulacion}_acc_mag'] = np.sqrt(
                df_derivadas_suav[f'{articulacion}_x_acc']**2 + df_derivadas_suav[f'{articulacion}_y_acc']**2
            )
            df_derivadas_suav[f'{articulacion}_acc_ang'] = np.arctan2(
                df_derivadas_suav[f'{articulacion}_y_acc'], df_derivadas_suav[f'{articulacion}_x_acc']
            )

        # Guardar resultados
        articulaciones = ['hombro', 'codo', 'muneca', 'cadera']
        columnas_unificado = ['frame']
        for articulacion in articulaciones:
            columnas_unificado.extend([
                f'{articulacion}_x', f'{articulacion}_y',
                f'{articulacion}_x_vel', f'{articulacion}_y_vel',
                f'{articulacion}_x_acc', f'{articulacion}_y_acc',
                f'{articulacion}_mag', f'{articulacion}_ang',
                f'{articulacion}_vel_mag', f'{articulacion}_vel_ang',
                f'{articulacion}_acc_mag', f'{articulacion}_acc_ang'
            ])
        
        columnas_existentes = [col for col in columnas_unificado if col in df_derivadas_suav.columns]
        df_unificado = df_derivadas_suav[columnas_existentes].copy()
        
        carpeta_datos = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datos')
        os.makedirs(carpeta_datos, exist_ok=True)
        
        ruta_pickle = os.path.join(carpeta_datos, 'datos_articulaciones.pkl')
        df_unificado.to_pickle(ruta_pickle)
        
        ruta_csv = os.path.join(carpeta_datos, 'datos_articulaciones.csv')
        df_unificado.to_csv(ruta_csv, index=False)

        print(f"\nResultados guardados en {carpeta_datos}:")
        print(f"- datos_articulaciones.pkl (DataFrame completo)")
        print(f"- datos_articulaciones.csv (Versión CSV)")

        # Generación de gráficos (suavizados y no suavizados)
        puntos_seguir = ['hombro', 'codo', 'muneca', 'cadera']
        for punto in puntos_seguir:
            # Gráficos para eje X (no suavizados)
            fig_x, axs_x = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig_x.suptitle(f'Movimiento de {punto.capitalize()} en eje X (No Suavizado)')
            axs_x[0].plot(df_derivadas['frame'], df_derivadas[f'{punto}_x'], label='Posición', color='blue')
            axs_x[0].set_ylabel('Posición (m)')
            axs_x[0].legend()
            axs_x[1].plot(df_derivadas['frame'], df_derivadas[f'{punto}_x_vel'], label='Velocidad', color='green')
            axs_x[1].set_ylabel('Velocidad (m/s)')
            axs_x[1].legend()
            axs_x[2].plot(df_derivadas['frame'], df_derivadas[f'{punto}_x_acc'], label='Aceleración', color='red')
            axs_x[2].set_xlabel('Frame')
            axs_x[2].set_ylabel('Aceleración (m/s²)')
            axs_x[2].legend()
            plt.tight_layout()
            ruta_grafico_x = os.path.join(carpeta_datos, f'movimiento_{punto}_X_no_suavizado.png')
            plt.savefig(ruta_grafico_x)
            plt.close()

            # Gráficos para eje X (suavizados)
            fig_x_suav, axs_x_suav = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig_x_suav.suptitle(f'Movimiento de {punto.capitalize()} en eje X (Suavizado)')
            axs_x_suav[0].plot(df_unificado['frame'], df_unificado[f'{punto}_x'], label='Posición', color='blue')
            axs_x_suav[0].set_ylabel('Posición (m)')
            axs_x_suav[0].legend()
            axs_x_suav[1].plot(df_unificado['frame'], df_unificado[f'{punto}_x_vel'], label='Velocidad', color='green')
            axs_x_suav[1].set_ylabel('Velocidad (m/s)')
            axs_x_suav[1].legend()
            axs_x_suav[2].plot(df_unificado['frame'], df_unificado[f'{punto}_x_acc'], label='Aceleración', color='red')
            axs_x_suav[2].set_xlabel('Frame')
            axs_x_suav[2].set_ylabel('Aceleración (m/s²)')
            axs_x_suav[2].legend()
            plt.tight_layout()
            ruta_grafico_x_suav = os.path.join(carpeta_datos, f'movimiento_{punto}_X_suavizado.png')
            plt.savefig(ruta_grafico_x_suav)
            plt.close()

            # Gráficos para eje Y (no suavizados)
            fig_y, axs_y = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig_y.suptitle(f'Movimiento de {punto.capitalize()} en eje Y (No Suavizado)')
            axs_y[0].plot(df_derivadas['frame'], df_derivadas[f'{punto}_y'], label='Posición', color='purple')
            axs_y[0].set_ylabel('Posición (m)')
            axs_y[0].legend()
            axs_y[1].plot(df_derivadas['frame'], df_derivadas[f'{punto}_y_vel'], label='Velocidad', color='orange')
            axs_y[1].set_ylabel('Velocidad (m/s)')
            axs_y[1].legend()
            axs_y[2].plot(df_derivadas['frame'], df_derivadas[f'{punto}_y_acc'], label='Aceleración', color='brown')
            axs_y[2].set_xlabel('Frame')
            axs_y[2].set_ylabel('Aceleración (m/s²)')
            axs_y[2].legend()
            plt.tight_layout()
            ruta_grafico_y = os.path.join(carpeta_datos, f'movimiento_{punto}_Y_no_suavizado.png')
            plt.savefig(ruta_grafico_y)
            plt.close()

            # Gráficos para eje Y (suavizados)
            fig_y_suav, axs_y_suav = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig_y_suav.suptitle(f'Movimiento de {punto.capitalize()} en eje Y (Suavizado)')
            axs_y_suav[0].plot(df_unificado['frame'], df_unificado[f'{punto}_y'], label='Posición', color='purple')
            axs_y_suav[0].set_ylabel('Posición (m)')
            axs_y_suav[0].legend()
            axs_y_suav[1].plot(df_unificado['frame'], df_unificado[f'{punto}_y_vel'], label='Velocidad', color='orange')
            axs_y_suav[1].set_ylabel('Velocidad (m/s)')
            axs_y_suav[1].legend()
            axs_y_suav[2].plot(df_unificado['frame'], df_unificado[f'{punto}_y_acc'], label='Aceleración', color='brown')
            axs_y_suav[2].set_xlabel('Frame')
            axs_y_suav[2].set_ylabel('Aceleración (m/s²)')
            axs_y_suav[2].legend()
            plt.tight_layout()
            ruta_grafico_y_suav = os.path.join(carpeta_datos, f'movimiento_{punto}_Y_suavizado.png')
            plt.savefig(ruta_grafico_y_suav)
            plt.close()

            # Gráficos cartesianos para polares (no suavizados)
            fig_vel_ang, ax_vel_ang = plt.subplots(figsize=(10, 6))
            fig_vel_ang.suptitle(f'Velocidad Angular de {punto.capitalize()} (No Suavizado)')
            ax_vel_ang.plot(df_derivadas['frame'], df_derivadas[f'{punto}_vel_ang'], label='Velocidad Angular', color='green')
            ax_vel_ang.set_xlabel('Frame')
            ax_vel_ang.set_ylabel('Ángulo (radianes)')
            ax_vel_ang.legend()
            plt.tight_layout()
            ruta_grafico_vel_ang = os.path.join(carpeta_datos, f'movimiento_{punto}_vel_ang_no_suavizado.png')
            plt.savefig(ruta_grafico_vel_ang)
            plt.close()

            fig_acc_ang, ax_acc_ang = plt.subplots(figsize=(10, 6))
            fig_acc_ang.suptitle(f'Aceleración Angular de {punto.capitalize()} (No Suavizado)')
            ax_acc_ang.plot(df_derivadas['frame'], df_derivadas[f'{punto}_acc_ang'], label='Aceleración Angular', color='red')
            ax_acc_ang.set_xlabel('Frame')
            ax_acc_ang.set_ylabel('Ángulo (radianes)')
            ax_acc_ang.legend()
            plt.tight_layout()
            ruta_grafico_acc_ang = os.path.join(carpeta_datos, f'movimiento_{punto}_acc_ang_no_suavizado.png')
            plt.savefig(ruta_grafico_acc_ang)
            plt.close()

            # Gráficos cartesianos para polares (suavizados)
            fig_vel_ang_suav, ax_vel_ang_suav = plt.subplots(figsize=(10, 6))
            fig_vel_ang_suav.suptitle(f'Velocidad Angular de {punto.capitalize()} (Suavizado)')
            ax_vel_ang_suav.plot(df_unificado['frame'], df_unificado[f'{punto}_vel_ang'], label='Velocidad Angular', color='green')
            ax_vel_ang_suav.set_xlabel('Frame')
            ax_vel_ang_suav.set_ylabel('Ángulo (radianes)')
            ax_vel_ang_suav.legend()
            plt.tight_layout()
            ruta_grafico_vel_ang_suav = os.path.join(carpeta_datos, f'movimiento_{punto}_vel_ang_suavizado.png')
            plt.savefig(ruta_grafico_vel_ang_suav)
            plt.close()

            fig_acc_ang_suav, ax_acc_ang_suav = plt.subplots(figsize=(10, 6))
            fig_acc_ang_suav.suptitle(f'Aceleración Angular de {punto.capitalize()} (Suavizado)')
            ax_acc_ang_suav.plot(df_unificado['frame'], df_unificado[f'{punto}_acc_ang'], label='Aceleración Angular', color='red')
            ax_acc_ang_suav.set_xlabel('Frame')
            ax_acc_ang_suav.set_ylabel('Ángulo (radianes)')
            ax_acc_ang_suav.legend()
            plt.tight_layout()
            ruta_grafico_acc_ang_suav = os.path.join(carpeta_datos, f'movimiento_{punto}_acc_ang_suavizado.png')
            plt.savefig(ruta_grafico_acc_ang_suav)
            plt.close()

        print("\nProceso completado exitosamente!")
else:
    print("No se pudo calcular un factor de conversión válido.")
