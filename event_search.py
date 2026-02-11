# -*- coding: utf-8 -*-
"""
Sentinel Event Search Tool
==========================
Ferramenta de linha de comando para pesquisar eventos de segurança gravados
pelo sentinel_analyzer.py no banco de dados SQLite.
"""
import sqlite3
import os

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
DATABASE_PATH = os.path.join(OUTPUT_DIR, "sentinel_events.db")
EVIDENCIAS_DIR = os.path.join(OUTPUT_DIR, "evidence_clips")

def search_events(db_path, color_query=None, object_query=None):
    """
    Busca eventos no banco de dados com base na cor e/ou tipo de objeto.
    """
    if not os.path.exists(db_path):
        print(f"Erro: Arquivo de banco de dados não encontrado em '{db_path}'.")
        print("Execute o 'sentinel_analyzer.py' primeiro para gerar eventos.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # A consulta base une as duas tabelas
        query = """
            SELECT DISTINCT
                e.id,
                e.timestamp,
                e.smart_alert,
                e.video_filename
            FROM events e
            JOIN detected_objects o ON e.id = o.event_id
        """
        
        conditions = []
        params = []
        
        if color_query:
            conditions.append("o.dominant_colors LIKE ?")
            params.append(f"%{color_query}%")
        
        if object_query:
            conditions.append("o.object_class LIKE ?")
            params.append(f"%{object_query}%")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY e.timestamp DESC"

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return results

    except sqlite3.Error as e:
        print(f"Erro ao consultar o banco de dados: {e}")
        return []

def main():
    """Loop principal da interface de busca."""
    print("========================================")
    print("--- Ferramenta de Busca de Eventos ---")
    print("========================================")
    
    while True:
        print("\nDigite os critérios de busca (deixe em branco para ignorar):")
        
        color_input = input("  > Cor do objeto (ex: Azul, Vermelho): ").strip().capitalize()
        object_input = input("  > Tipo de objeto (ex: person, car, backpack): ").strip().lower()
        
        if not color_input and not object_input:
            print("\nPor favor, forneça pelo menos um critério de busca.")
            continue
            
        print("\nBuscando eventos...")
        found_events = search_events(DATABASE_PATH, color_query=color_input, object_query=object_input)
        
        if not found_events:
            print("\n----------------------------------------")
            print("Nenhum evento encontrado com esses critérios.")
            print("----------------------------------------")
        else:
            print(f"\n----------------------------------------")
            print(f"Encontrados {len(found_events)} evento(s):")
            print("----------------------------------------")
            for event in found_events:
                event_id, timestamp, alert, filename = event
                print(f"  ID do Evento: {event_id}")
                print(f"  Data/Hora:    {timestamp}")
                print(f"  Smart Alert:  {alert}")
                print(f"  Arquivo:      {os.path.join(EVIDENCIAS_DIR, filename)}")
                print("----------------------------------------")

        another_search = input("\nDeseja fazer outra busca? (s/n): ").strip().lower()
        if another_search != 's':
            break

    print("\nEncerrando a ferramenta de busca.")


if __name__ == "__main__":
    main()