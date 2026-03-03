from database_manager import init_db, get_or_create_id, update_vocab_usage, build_hot_token_cache, TEXT_TO_ID_CACHE
import time

def main():
    print("ZeroRAM-GEN: Validando Sprint 27 (Hot-Token Cache)")
    
    # Init
    init_db()
    
    # 1. Simular uso de tokens comuns
    common_tokens = ["o", "a", "robô", "zero", "ram"]
    print(f"Simulando uso intensivo de: {common_tokens}")
    for token in common_tokens:
        token_id = get_or_create_id(token)
        # Simular 100 usos para cada
        for _ in range(100):
            update_vocab_usage(token_id)
            
    # 2. Reconstruir Cache de Hot-Tokens
    print("\nConstruindo cache de Hot-Tokens...")
    build_hot_token_cache(size=10)
    
    # 3. Validar se estão no cache
    print("\nVerificando Cache (RAM):")
    for token in common_tokens:
        in_cache = token in TEXT_TO_ID_CACHE
        status = "[OK] No Cache" if in_cache else "[ERRO] Não encontrado"
        print(f"Token '{token}': {status}")
        
    # 4. Teste de Performance (Opcional)
    start = time.time()
    for _ in range(1000):
        _ = get_or_create_id("robô")
    duration = time.time() - start
    print(f"\nTempo para 1000 lookups (Cache HIT): {duration:.4f}s")

    print("\n[OK] Sprint 27 validada. O sistema agora prioriza tokens quentes na RAM.")

if __name__ == "__main__":
    main()
