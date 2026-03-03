from tensor_manager import load_model_checkpoint, save_model_checkpoint, sync_model_to_vocab
from compress_checkpoint import compress_checkpoint_v11
from database_manager import init_db

def final_fix_v12():
    print("--- Dyson Final Fix: Sincronização e Compressão ---")
    init_db()
    
    # 1. Carregar o checkpoint PT que estava com tamanho errado
    # (Ou recarregar do v1.1_pt e syncar)
    if load_model_checkpoint("v1.1_pt"):
        print("[SYNC] Sincronizando dimensões...")
        sync_model_to_vocab()
        save_model_checkpoint("v1.1_pt_synced")
        
        # 2. Comprimir o sincado
        # Vou modificar compress_checkpoint_v11 para aceitar o nome do ckpt
        # Mas por agora apenas renomeio v1.1_pt_synced para v1.1_pt temporariamente
        import shutil
        import os
        ckpt_dir = os.path.join("checkpoints", "v1.1_pt")
        synced_dir = os.path.join("checkpoints", "v1.1_pt_synced")
        
        # Backup do quebrado
        shutil.move(ckpt_dir, os.path.join("checkpoints", "v1.1_pt_broken"))
        shutil.copytree(synced_dir, ckpt_dir)
        
        # Chamar compressão (ele olha para os pesos atuais em weights/)
        print("[COMPRESS] Iniciando compressão INT4...")
        compress_checkpoint_v11()
    else:
        print("[ERR] Não foi possível carregar v1.1_pt")

if __name__ == "__main__":
    final_fix_v12()
