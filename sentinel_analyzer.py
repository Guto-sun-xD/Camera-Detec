# -*- coding: utf-8 -*-
"""
Sentinel Analyzer - Hybrid Security Video Analysis System
=========================================================
VERSÃO 2.0

Este script implementa uma arquitetura de vigilância híbrida e inteligente,
projetada como um protótipo robusto para empresas de segurança.

FUNCIONALIDADES:
-   Arquitetura Híbrida: Usa um gatilho leve de detecção de movimento para
    iniciar uma análise profunda sob demanda, economizando recursos.
-   Análise Profunda: Utiliza YOLOv7-tiny para detecção, rastreamento de
    objetos com "Ghost Tracking" para oclusões e lógica causal para gerar
    "Smart Alerts" (ex: "Pessoa deixou uma mochila").
-   Detecção de Cor Aprimorada: Lógica de cor recalibrada para identificar
    melhor os tons do dia a dia (incluindo azuis escuros) e pode registrar
    até duas cores dominantes se houver ambiguidade.
-   Banco de Dados Estruturado: Cada evento e os objetos detectados dentro
    dele (com suas classes e cores) são salvos em um banco de dados SQLite,
    permitindo buscas detalhadas posteriores com um script separado.
"""
import cv2
import numpy as np
import datetime
import os
import sys
import imutils
import time
import math
import sqlite3
from collections import Counter, defaultdict, deque

# ==============================================================================
# --- CONFIGURAÇÕES GERAIS E PARÂMETROS DE AJUSTE ---
# ==============================================================================

# --- CONFIGURAÇÕES DE PASTAS E ARQUIVOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
EVIDENCIAS_DIR = os.path.join(OUTPUT_DIR, "evidence_clips")
DATABASE_PATH = os.path.join(OUTPUT_DIR, "sentinel_events.db")

# --- VÍDEO DE ENTRADA ---
# Para usar a webcam, coloque 0. Para um arquivo, coloque o caminho.
# Exemplo: "meu_video.mp4" ou 0
VIDEO_INPUT = 0

# --- CONFIGURAÇÕES DO MODELO YOLO ---
MODEL_DIR = os.path.join(BASE_DIR, "yolo_models")
YOLO_CFG = os.path.join(MODEL_DIR, "yolov7-tiny.cfg")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov7-tiny.weights")
YOLO_NAMES = os.path.join(MODEL_DIR, "coco.names")

# --- PARÂMETROS DO GATILHO DE MOVIMENTO ---
MOTION_DETECTION_FRAME_WIDTH = 500
MIN_MOTION_AREA = 800
EVENT_COOLDOWN_SECONDS = 5.0

# --- PARÂMETROS DE INFERÊNCIA E ANÁLISE ---
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# --- PARÂMETROS DO MOTOR DE RASTREAMENTO E CAUSALIDADE ---
MIN_OVERLAP_FRAMES = 12
MAX_MISSING_FRAMES = 40
MAX_REID_DISTANCE = 90
MANIPULABLE_CLASSES = ['backpack', 'suitcase', 'handbag', 'laptop', 'bottle', 'remote', 'cell phone', 'book']
MAJOR_CLASSES = ['person', 'car', 'truck', 'bus', 'motorcycle']

# ==============================================================================
# --- FUNÇÕES AUXILIARES E DE LÓGICA ---
# ==============================================================================

def get_dominant_color(image_roi):
    """
    Calcula a(s) cor(es) dominante(s) de uma ROI (Região de Interesse).
    - Regras de HSV ajustadas para melhor detectar cores escuras (ex: azul-marinho).
    - Retorna uma lista com até duas cores se a segunda for relevante.
    """
    if image_roi is None or image_roi.size == 0 or image_roi.shape[0] < 5 or image_roi.shape[1] < 5:
        return ["N/A"]
    try:
        # Redimensiona para performance e converte para um formato adequado para o K-Means
        pixels = cv2.resize(image_roi, (50, 50), interpolation=cv2.INTER_AREA).reshape(-1, 3)
        pixels = np.float32(pixels[np.any(pixels != [0, 0, 0], axis=1)])
        if len(pixels) < 10: return ["N/A"]
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        counts = Counter(labels.flatten())
        sorted_counts = counts.most_common()
        
        detected_colors = []

        def classify_hsv(h, s, v):
            """Classifica um pixel HSV em um nome de cor."""
            if v > 200 and s < 30: return "Branco"
            if v < 50: return "Preto"
            if s < 35: return "Cinza"
            if (h < 6) or (h > 170): return "Vermelho"
            if h < 24: return "Laranja"
            if h < 40: return "Amarelo"
            if h < 78: return "Verde"
            if h < 135: return "Azul"
            if h < 158: return "Roxo"
            return "Outra"

        # Processa a cor mais comum
        if sorted_counts:
            dominant_bgr = centers[sorted_counts[0][0]]
            h, s, v = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            main_color = classify_hsv(h, s, v)
            if main_color != "Outra":
                detected_colors.append(main_color)

        # Processa a segunda cor mais comum (se for relevante)
        if len(sorted_counts) > 1:
            count1 = sorted_counts[0][1]
            count2 = sorted_counts[1][1]
            # Se a segunda cor tiver pelo menos 60% da frequência da primeira, também é relevante
            if count2 >= count1 * 0.6:
                second_bgr = centers[sorted_counts[1][0]]
                h, s, v = cv2.cvtColor(np.uint8([[second_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                second_color = classify_hsv(h, s, v)
                if second_color != "Outra" and second_color not in detected_colors:
                    detected_colors.append(second_color)
        
        return detected_colors if detected_colors else ["N/A"]

    except Exception:
        return ["N/A"]

def calculate_iou(box1, box2):
    x_a, y_a = max(box1[0], box2[0]), max(box1[1], box2[1])
    x_b, y_b = min(box1[0] + box1[2], box2[0] + box2[2]), min(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box1_area, box2_area = box1[2] * box1[3], box2[2] * box2[3]
    union_area = float(box1_area + box2_area - inter_area)
    return inter_area / union_area if union_area > 0 else 0

def get_box_center(box):
    return (box[0] + box[2] // 2, box[1] + box[3] // 2)

def calculate_distance(center1, center2):
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

# ==============================================================================
# --- FUNÇÕES DE BANCO DE DADOS (SQLite) ---
# ==============================================================================

def init_database(db_path):
    """Inicializa o BD com tabelas para eventos E objetos detectados."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Tabela principal de eventos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                video_filename TEXT NOT NULL UNIQUE,
                smart_alert TEXT NOT NULL
            )
        """)
        # Tabela para objetos, com referência ao evento
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detected_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                object_class TEXT NOT NULL,
                dominant_colors TEXT,
                FOREIGN KEY (event_id) REFERENCES events (id)
            )
        """)
        conn.commit()
        conn.close()
        print(f"[BD] Banco de dados inicializado em: {db_path}")
    except sqlite3.Error as e:
        print(f"[BD] Erro ao inicializar o banco de dados: {e}")
        sys.exit(1)

def insert_event_and_objects(db_path, timestamp, video_filename, smart_alert, tracked_objects):
    """Insere um evento e todos os seus objetos rastreados no BD."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO events (timestamp, video_filename, smart_alert) VALUES (?, ?, ?)",
                       (timestamp, video_filename, smart_alert))
        event_id = cursor.lastrowid

        objects_to_insert = []
        for data in tracked_objects.values():
            if data['first_frame'] != -1 and data['colors']:
                obj_class = data['class']
                top_colors = [color for color, _ in data['colors'].most_common(2)]
                colors_str = ",".join(top_colors)
                objects_to_insert.append((event_id, obj_class, colors_str))
        
        if objects_to_insert:
            cursor.executemany("INSERT INTO detected_objects (event_id, object_class, dominant_colors) VALUES (?, ?, ?)",
                               objects_to_insert)

        conn.commit()
        conn.close()
        print(f"[BD] Evento {event_id} e {len(objects_to_insert)} objetos registrados com sucesso.")
    except sqlite3.IntegrityError:
         print(f"[BD] AVISO: Evento com nome de arquivo '{video_filename}' já existe. Ignorando inserção.")
    except sqlite3.Error as e:
        print(f"[BD] Erro ao inserir evento no banco de dados: {e}")

# ==============================================================================
# --- NÚCLEO DE ANÁLISE PROFUNDA (MOTOR DE INTELIGÊNCIA) ---
# ==============================================================================

def generate_story_summary(object_tracker, final_frame_objects):
    """Gera uma descrição textual (Smart Alert) baseada nos dados rastreados."""
    if not object_tracker:
        return "Nenhum objeto rastreado com persistência na cena."

    stories = []
    people = {tid: data for tid, data in object_tracker.items() if data['class'] == 'person'}

    for p_id, p_data in people.items():
        person_disappeared = p_id not in final_frame_objects
        person_color = p_data['colors'].most_common(1)[0][0] if p_data['colors'] else 'cor não identificada'

        if p_data['interaction_frames'] >= MIN_OVERLAP_FRAMES and p_data['last_interacted_obj_id']:
            obj_id = p_data['last_interacted_obj_id']
            if obj_id in object_tracker:
                obj_data = object_tracker[obj_id]
                obj_disappeared = obj_id not in final_frame_objects
                obj_class = obj_data['class'].capitalize()
                obj_color = obj_data['colors'].most_common(1)[0][0] if obj_data['colors'] else 'cor não identificada'
                
                if person_disappeared and not obj_disappeared:
                    stories.append(f"ALERTA DE ABANDONO: Pessoa {person_color} DEIXOU um(a) {obj_class} {obj_color} e saiu.")
                    p_data['processed'] = True 
                elif person_disappeared and obj_disappeared:
                    stories.append(f"ALERTA DE REMOÇÃO: Pessoa {person_color} REMOVEU um(a) {obj_class} {obj_color} da cena.")
                    p_data['processed'] = True

    for tid, data in object_tracker.items():
        if data.get('processed'): continue 
        
        # CORREÇÃO: Um objeto apareceu se ele foi detectado em qualquer frame válido (não -1)
        obj_appeared = data['first_frame'] != -1
        
        obj_disappeared = tid not in final_frame_objects
        obj_class = data['class'].capitalize()
        top_colors = [color for color, _ in data['colors'].most_common(2)]
        color_str = ",".join(top_colors) if top_colors else 'N/A'
        
        if obj_appeared and obj_disappeared:
            stories.append(f"Um(a) {obj_class} ({color_str}) entrou e saiu da cena.")
        elif obj_appeared and not obj_disappeared:
            stories.append(f"Um(a) {obj_class} ({color_str}) entrou e permaneceu na cena.")

    if not stories:
        return "Atividade detectada, mas sem eventos de interesse conclusivos."

    return "Resumo da Cena: " + " | ".join(stories)

def analyze_event_scene(scene_frames, net, classes):
    """Executa a análise profunda em uma cena capturada."""
    print(f"[Análise] Processando cena com {len(scene_frames)} frames...")
    (H, W) = scene_frames[0].shape[:2]
    
    object_tracker = defaultdict(lambda: {
        'class': '', 'colors': Counter(), 'last_box': (0,0,0,0), 
        'first_frame': -1, 'last_seen_frame': 0, 'interaction_frames': 0,
        'last_interacted_obj_id': None
    })
    occlusion_tracker = {}
    last_frame_detections = {}
    current_tracker_id = 0
    annotated_frames = []

    for frame_idx, frame in enumerate(scene_frames):
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(output_layers)

        boxes, confidences, classIDs = [], [], []
        for output in outs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > CONFIDENCE_THRESHOLD:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        current_frame_detections = {}
        
        if len(indexes) > 0:
            detections = [(boxes[i], classes[classIDs[i]]) for i in indexes.flatten()]
            
            for box, label in detections:
                if label not in MAJOR_CLASSES and label not in MANIPULABLE_CLASSES:
                    continue

                center = get_box_center(box)
                best_match_id = -1

                min_dist = MAX_REID_DISTANCE
                for ghost_id, data in occlusion_tracker.items():
                    if data['class'] == label:
                        dist = calculate_distance(center, get_box_center(data['last_box']))
                        if dist < min_dist:
                            min_dist = dist
                            best_match_id = ghost_id
                
                if best_match_id != -1:
                    track_id = best_match_id
                    del occlusion_tracker[best_match_id]
                else:
                    best_iou = 0.2
                    for prev_id, data in last_frame_detections.items():
                        if data['label'] == label and prev_id not in current_frame_detections:
                            iou = calculate_iou(box, data['box'])
                            if iou > best_iou:
                                best_iou = iou
                                best_match_id = prev_id
                    
                    if best_match_id != -1:
                        track_id = best_match_id
                    else:
                        current_tracker_id += 1
                        track_id = current_tracker_id
                        object_tracker[track_id]['first_frame'] = frame_idx
                
                current_frame_detections[track_id] = {'box': box, 'label': label}

        for track_id, data in current_frame_detections.items():
            box, label = data['box'], data['label']
            (x, y, w, h) = box
            roi = frame[y:y+h, x:x+w]
            color_names = get_dominant_color(roi)
            
            for color in color_names:
                if color != "N/A":
                    object_tracker[track_id]['colors'][color] += 1
            
            object_tracker[track_id].update({'class': label, 'last_box': box, 'last_seen_frame': frame_idx})
            
            color_text = ",".join(color_names)
            text = f"{label.capitalize()} ID:{track_id} ({color_text})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        people_in_frame = {tid: data for tid, data in current_frame_detections.items() if data['label'] == 'person'}
        objects_in_frame = {tid: data for tid, data in current_frame_detections.items() if data['label'] in MANIPULABLE_CLASSES}
        
        for p_id, p_data in people_in_frame.items():
            interaction_found = False
            for obj_id, obj_data in objects_in_frame.items():
                if calculate_iou(p_data['box'], obj_data['box']) > 0.1:
                    object_tracker[p_id]['interaction_frames'] += 1
                    object_tracker[p_id]['last_interacted_obj_id'] = obj_id
                    interaction_found = True
                    break
            if not interaction_found:
                 object_tracker[p_id]['interaction_frames'] = max(0, object_tracker[p_id]['interaction_frames'] - 1)

        for prev_id, data in last_frame_detections.items():
            if prev_id not in current_frame_detections:
                occlusion_tracker[prev_id] = {'class': data['label'], 'last_box': data['box'], 'missing_frames': 1}
        
        for ghost_id, data in list(occlusion_tracker.items()):
            data['missing_frames'] += 1
            if data['missing_frames'] > MAX_MISSING_FRAMES:
                del occlusion_tracker[ghost_id]

        last_frame_detections = current_frame_detections
        annotated_frames.append(frame)

    final_frame_objects = {tid for tid, data in object_tracker.items() if data['last_seen_frame'] == len(scene_frames) - 1}
    story = generate_story_summary(object_tracker, final_frame_objects)
    
    print(f"[Análise] Concluída. Resultado: {story}")
    return annotated_frames, story, object_tracker

# ==============================================================================
# --- FUNÇÃO PRINCIPAL E LOOP DE EXECUÇÃO ---
# ==============================================================================

# ==============================================================================
# --- FUNÇÃO PRINCIPAL E LOOP DE EXECUÇÃO ---
# ==============================================================================

def main():
    for path in [OUTPUT_DIR, EVIDENCIAS_DIR, YOLO_CFG, YOLO_WEIGHTS, YOLO_NAMES]:
        if not os.path.exists(path) and '.' not in os.path.basename(path):
            os.makedirs(path, exist_ok=True)
        elif not os.path.exists(path):
            print(f"🚨 ERRO FATAL: Arquivo necessário não encontrado: {path}")
            sys.exit(1)

    init_database(DATABASE_PATH)
    
    with open(YOLO_NAMES, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]
    NET = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
    
    try:
        NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("INFO: Backend CUDA (GPU) selecionado para a rede YOLO.")
    except:
        print("INFO: Backend CUDA não disponível. Usando CPU.")

    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"ERRO: Não foi possível abrir a fonte de vídeo: {VIDEO_INPUT}")
        sys.exit(1)

    (H, W) = (None, None)
    previous_frame_gray = None
    last_motion_time = time.time()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    scene_buffer = deque(maxlen=int(fps * 30))
    
    SYSTEM_STATUS = "OCIOSO"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro de captura.")
            break

        if W is None or H is None: (H, W) = frame.shape[:2]

        motion_detected = False
        gray_frame = cv2.cvtColor(imutils.resize(frame, width=MOTION_DETECTION_FRAME_WIDTH), cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        
        if previous_frame_gray is None:
            previous_frame_gray = gray_frame
            continue

        frame_delta = cv2.absdiff(previous_frame_gray, gray_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = imutils.grab_contours(cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
        
        for c in contours:
            if cv2.contourArea(c) > MIN_MOTION_AREA:
                motion_detected = True
                break
        previous_frame_gray = gray_frame

        if motion_detected:
            if SYSTEM_STATUS == "OCIOSO":
                SYSTEM_STATUS = "GRAVANDO EVENTO"
                scene_buffer.clear()
            scene_buffer.append(frame.copy())
            last_motion_time = time.time()
        elif SYSTEM_STATUS == "GRAVANDO EVENTO":
            if time.time() - last_motion_time > EVENT_COOLDOWN_SECONDS:
                SYSTEM_STATUS = "ANALISANDO CENA"
                
                if scene_buffer:
                    annotated_frames, smart_alert, tracked_objects = analyze_event_scene(list(scene_buffer), NET, CLASSES)
                    
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                    # 🔧 CORREÇÃO: Adiciona milissegundos ao nome do arquivo para garantir que seja único
                    file_timestamp = now.strftime("%Y%m%d_%H%M%S_%f") 
                    video_filename = f"EVENT_{file_timestamp}.mp4"
                    video_path = os.path.join(EVIDENCIAS_DIR, video_filename)
                    
                    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
                    for i, af in enumerate(annotated_frames):
                        if i >= len(annotated_frames) - int(fps * 3):
                           cv2.putText(af, "SMART ALERT:", (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                           cv2.putText(af, smart_alert, (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        out.write(af)
                    out.release()
                    print(f"Evidência de vídeo salva em: {video_path}")
                    
                    insert_event_and_objects(DATABASE_PATH, timestamp, video_filename, smart_alert, tracked_objects)
                
                scene_buffer.clear()
                SYSTEM_STATUS = "OCIOSO"

        display_frame = frame.copy()
        status_colors = {"OCIOSO": (0, 255, 0), "GRAVANDO EVENTO": (0, 165, 255), "ANALISANDO CENA": (0, 0, 255)}
        status_text = f"Status: {SYSTEM_STATUS}"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_colors[SYSTEM_STATUS], 2)
        cv2.imshow("Sentinel Analyzer - Hybrid Mode", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Análise encerrada.")


if __name__ == "__main__":
    try:
        import imutils
    except ImportError:
        print("\n--- BIBLIOTECA FALTANDO: 'imutils' é necessária. ---\n")
        print("--> Por favor, instale com: pip install imutils\n")
        sys.exit(1)
    main()