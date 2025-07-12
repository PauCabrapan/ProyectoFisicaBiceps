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
position_buffer_x = []  # Buffer para posiciones x
position_buffer_y = []  # Buffer para posiciones y

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

    work_mus = 0.0
    previous_theta = None


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

                # Suavizado de posiciones usando un buffer
                if factor is not None:
                    position_buffer_x.append(xm * factor)
                    position_buffer_y.append(ym * factor)
                    if len(position_buffer_x) > ventana:
                        position_buffer_x.pop(0)
                        position_buffer_y.pop(0)
                    smoothed_x = np.mean(position_buffer_x) if position_buffer_x else xm * factor
                    smoothed_y = np.mean(position_buffer_y) if position_buffer_y else ym * factor
                else:
                    smoothed_x = xm * 0.001  # Factor por defecto si no se calculó
                    smoothed_y = ym * 0.001


                # Calcular velocidad y aceleración a partir de posiciones suavizadas
                vel_x = vel_y = acc_x = acc_y = 0
                if previous_position is not None and factor is not None:
                    dx = (smoothed_x - previous_position[0])
                    dy = (smoothed_y - previous_position[1])
                    vel_x = dx / dt  # Velocidad en m/s
                    vel_y = dy / dt
                    acc_x = (vel_x - previous_velocity[0]) / dt if frame_id > 0 else 0
                    acc_y = (vel_y - previous_velocity[1]) / dt if frame_id > 0 else 0
                    previous_velocity = [vel_x, vel_y]

                    # Suavizado simple de la aceleración (promedio móvil básico)
                    smoothed_acc_x = (smoothed_acc_x * (ventana - 1) + acc_x) / ventana if frame_id > 0 else acc_x
                    smoothed_acc_y = (smoothed_acc_y * (ventana - 1) + acc_y) / ventana if frame_id > 0 else acc_y

                previous_position = (smoothed_x, smoothed_y)

                print(f"Frame {frame_id}: Posición suavizada (x, y) = ({smoothed_x:.6f}, {smoothed_y:.6f}), Velocidad (x, y) = ({vel_x:.6f}, {vel_y:.6f}), Aceleración suavizada (x, y) = ({smoothed_acc_x:.6f}, {smoothed_acc_y:.6f})")

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
                insertion_dist_m = 0.04              # 4 cm en metros
                insertion_px = insertion_dist_m / factor  # distancia en píxeles
                vec_mx, vec_my = xh - xc, yh - yc
                mag_m = math.hypot(vec_mx, vec_my)
                ux, uy = vec_mx/mag_m, vec_my/mag_m if mag_m>0 else (0,0)
                vec_cw_x, vec_cw_y = xm - xc, ym - yc
                mag_cw = math.hypot(vec_cw_x, vec_cw_y)
                ux_ins, uy_ins = vec_cw_x/mag_cw, vec_cw_y/mag_cw if mag_cw>0 else (0,0)
                origin_fm = (
                    int(xc + ux_ins * insertion_px),
                    int(yc + uy_ins * insertion_px)
                )
                end_fm = (
                    int(origin_fm[0] + ux * fm_px_len),
                    int(origin_fm[1] + uy * fm_px_len)
                )
                cv2.arrowedLine(frame, origin_fm, end_fm,
                                (255,255,0), 3, tipLength=0.3)
                cv2.putText(frame, 'Fm',
                            (end_fm[0] + 5, end_fm[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                
                # Cálculo de la magnitud de Fm en cada frame 

                peso_usuario = 74.0           
                m_ant        = 0.02 * peso_usuario  # masa antebrazo
                m_manc       = 2.5                 # kg, tu mancuerna
                g            = 9.81                # m/s²

                # Distancias en metros (factor = m/px)
                r_d = math.hypot(xm-xc, ym-yc) * factor   # codo→muñeca
                r_g = r_d / 2.0                           # codo→CM antebrazo
                r_m = 0.04                                # 4 cm = brazo palanca bíceps

                # Momento gravitatorio
                tau_ant  = m_ant  * g * r_g
                tau_manc = m_manc * g * r_d

                # Fuerza muscular estática
                Fm_mag = (tau_ant + tau_manc) / r_m
                # --- Ángulo actual del antebrazo (codo→muñeca) ---

                # Convertimos a metros para ser coherentes, pero el factor se cancela en la dirección.
                dx = (xm - xc) * factor
                dy = (ym - yc) * factor
                theta = math.atan2(dy, dx)   # radianes

                # Calculo del trabajo
                if previous_theta is not None:
                    dtheta = theta - previous_theta
                    # Torque muscular en N·m
                    torque_muscular = Fm_mag * r_m  
                    # dW = τ · dθ
                    work_mus += torque_muscular * abs(dtheta)   


                    eta = 0.25                           # eficiencia muscular
                    E_metab_J   = work_mus / eta        # J metabolicos
                    kcal_burned = E_metab_J / 4184.0    # kcal

                    # Dibujar en pantalla
                    cv2.putText(frame,
                        f"Cal: {kcal_burned:.4f} kcal",
                        (1400, text_h*2 + 150),            # justo debajo de W
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.5,
                        (255,255,255),
                        4,
                        cv2.LINE_AA)
                    
                previous_theta = theta

                # Imprime el calculo del trabajo por pantalla
                h_frame, w_frame, _ = frame.shape
                cv2.putText(frame,
                            f"W = {work_mus:.1f} J",
                            (10, h_frame - 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            3.0,
                            (255,255,255),
                            4,
                            cv2.LINE_AA)
                


                h_frame, w_frame, _ = frame.shape


                text = f"Fm = {Fm_mag:.1f} N"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 3.0
                thickness = 5

                (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)

                pos_x = 10
                pos_y = h_frame - 10

                cv2.putText(
                    frame,
                    text,
                    (pos_x, pos_y),
                    font,
                    scale,
                    (255,255,255),
                    thickness,
                    cv2.LINE_AA
                )
                
                # Realiza flecha del peso de la mancuerna 
                
                masa_manc = 2.5       
                g = 9.81              
                fuerza_manc = masa_manc * g  
                peso_db_px = VECTOR_SCALE // 2
                origin_db = (xm, ym)
                end_db = (xm, ym + peso_db_px)
                cv2.arrowedLine(frame, origin_db, end_db,
                                (255, 0, 0), 3, tipLength=0.3)
                cv2.putText(frame, 'Pm',
                            (end_db[0] - 10, end_db[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                

                
                    
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

        # Calcular energías al final del bucle
        m_total = m_ant + m_manc  # Masa total (antebrazo + mancuerna)
        h_ref = 0.605 #ym * factor if frame_id == 546 and factor is not None else yc * 0.001  # Altura de referencia (inicial)
        h = (h_ref - smoothed_y) if frame_id > 0 and factor is not None else 0  # Altura relativa en metros
        Ep = m_total * g * h if factor is not None else 0  # Energía potencial en jules
        I = ((1/3) * m_ant * (r_d ** 2) +  m_manc * ((r_d) ** 2)) if factor is not None else 0  # Momento de inercia total

        # Almacenar datos para gráficos
        if 'energy_data' not in locals():
            energy_data = {'frame': [], 'Ep': [], 'Ek': []}
        energy_data['frame'].append(frame_id)
        energy_data['Ep'].append(Ep)

        # Actualizar energía cinética con theta (usamos el theta calculado antes)
        if previous_theta is not None and frame_id > 0 and factor is not None:
            omega = theta / dt  # Velocidad angular
            Ek = 0.5 * I * (omega ** 2)  # Energía cinética en julios
            energy_data['Ek'].append(Ek)  # Agregamos Ek aquí
        else:
            energy_data['Ek'].append(0)  # Si no hay theta previo, ponemos 0
        
        previous_r_d = r_d
        
        if previous_r_d is not None and previous_theta is not None and factor is not None:
            # velocidad radial ṙ
            r_dot = (r_d - previous_r_d) / dt
            # velocidad angular θ̇
            omega = (theta ) / dt
            # Ek según la nueva fórmula
            Ek = 0.5 * m_total * (r_dot**2 + (r_d * omega)**2)
            print(f"Frame {frame_id}: r={r_d:.4f} m, ṙ={r_dot:.4f} m/s, θ̇={omega:.4f} rad/s, Ek={Ek:.4f} J")
        else:
            Ek = 0.0
            print(f"Frame {frame_id}: sin previos, ṙ/θ̇→Ek=0")


print(f"Trabajo total hecho por el bíceps: {work_mus:.2f} J")

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

            #Gráfico de Energía Potencial Gravitatoria (suavizado)
            for punto in ['muneca']:  # Solo muneca para simplificar, podés agregar otros
                fig_ep, ax_ep = plt.subplots(figsize=(10, 6))
                fig_ep.suptitle(f'Energía Potencial Gravitatoria de {punto.capitalize()} (Suavizado)')
                ax_ep.plot(df_unificado['frame'], [energy_data['Ep'][i] for i in range(len(df_unificado)) if i < len(energy_data['Ep'])], label='Energía Potencial (J)', color='blue')
                ax_ep.set_xlabel('Frame')
                ax_ep.set_ylabel('Energía Potencial (J)')
                ax_ep.legend()
                plt.tight_layout()
                ruta_grafico_ep = os.path.join(carpeta_datos, f'movimiento_{punto}_energia_potencial_suavizado.png')
                plt.savefig(ruta_grafico_ep)
                plt.close()

            # Gráfico de Energía Cinética (suavizado)
            for punto in ['muneca']:  # Solo muneca para simplificar, podés agregar otros
                """fig_ek, ax_ek = plt.subplots(figsize=(10, 6))
                fig_ek.suptitle(f'Energía Cinética de {punto.capitalize()} (Suavizado)')
                ek_values = [energy_data['Ek'][i] for i in range(len(df_unificado)) if i < len(energy_data['Ek'])]
                ax_ek.plot(df_unificado['frame'], ek_values, label='Energía Cinética (J)', color='red')"""
                frames_ek = energy_data['frame'][1:]
                ek_values = energy_data['Ek'][1:]
                fig_ek, ax_ek = plt.subplots(figsize=(10, 6))
                ax_ek.set_title('Energía Cinética de Muneca (Suavizado)')
                ax_ek.plot(frames_ek, ek_values, label='Energía Cinética (J)')
                ax_ek.set_xlabel('Frame')
                ax_ek.set_ylabel('Energía Cinética (J)')
                ax_ek.legend()
                plt.tight_layout()
                ruta_grafico_ek = os.path.join(carpeta_datos, f'movimiento_{punto}_energia_cinetica_suavizado.png')
                plt.savefig(ruta_grafico_ek)
                plt.close()

        print("\nProceso completado exitosamente!")
else:
    print("No se pudo calcular un factor de conversión válido.")
    print("No se pudo calcular un factor de conversión válido.")