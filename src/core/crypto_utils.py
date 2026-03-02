from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
import base64

class CryptoUtils:
    @staticmethod
    def generate_key_pair() -> tuple[str, str]:
        private_key = ec.generate_private_key(ec.SECP256K1())
        public_key = private_key.public_key()
        
        # Serialize Private Key
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Serialize Public Key
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return public_bytes.decode('utf-8'), private_bytes.decode('utf-8')

    @staticmethod
    def normalize_public_key(pem_string: str) -> str:
        # Remove headers e footers para enviar via API
        lines = pem_string.strip().split("\n")
        data_lines = [line for line in lines if not line.startswith("-----")]
        return "".join(data_lines)
