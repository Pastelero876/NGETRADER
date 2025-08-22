from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict

from cryptography.fernet import Fernet
import os
import hmac
import hashlib
import time
import json


class SecretStore:
    """Almacena/recupera secretos cifrados en disco usando Fernet.

    Clave KMS simple en `data/secret.key`. Archivo de secretos: `data/secrets.enc` (JSON minimal).
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path("data")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.key_path = self.base_dir / "secret.key"
        self.store_path = self.base_dir / "secrets.enc"
        self._fernet = Fernet(self._get_or_create_key())
        self._cache: Dict[str, str] = {}

    def _get_or_create_key(self) -> bytes:
        if self.key_path.exists():
            return self.key_path.read_bytes().strip()
        key = Fernet.generate_key()
        self.key_path.write_bytes(key)
        return key

    def save(self, secrets: Dict[str, str]) -> None:
        # mezclar con existentes
        current = {}
        try:
            current = self.load()
        except Exception:
            current = {}
        current.update(secrets)
        from nge_trader.config.settings import Settings as _S
        backend = _S().secret_backend
        if backend == "env":
            # Guardar en .env (fallback) – no recomendado para prod
            from pathlib import Path as _P
            lines = [f"{k}={v}" for k, v in current.items()]
            (_P(".env")).write_text("\n".join(lines), encoding="utf-8")
        elif backend == "vault":
            try:
                from nge_trader.config.settings import Settings as _SS
                s = _SS()
                import requests as _rq
                if not (s.vault_addr and s.vault_token and s.vault_kv_path):
                    raise RuntimeError("Faltan variables de Vault")
                url = f"{s.vault_addr.rstrip('/')}/v1/{s.vault_kv_path.lstrip('/')}"
                headers = {"X-Vault-Token": s.vault_token}
                payload = {"data": current}
                resp = _rq.post(url, headers=headers, data=json.dumps(payload), timeout=10)
                resp.raise_for_status()
            except Exception as exc:
                raise RuntimeError(f"Error guardando en Vault: {exc}")
        else:
            payload = json.dumps(current, ensure_ascii=False).encode("utf-8")
            token = self._fernet.encrypt(payload)
            self.store_path.write_bytes(token)
        self._cache = current.copy()

    def load(self) -> Dict[str, str]:
        from nge_trader.config.settings import Settings as _S
        backend = _S().secret_backend
        # Backend Vault
        if backend == "vault":
            try:
                s = _S()
                import requests as _rq
                if not (s.vault_addr and s.vault_token and s.vault_kv_path):
                    return {}
                url = f"{s.vault_addr.rstrip('/')}/v1/{s.vault_kv_path.lstrip('/')}"
                headers = {"X-Vault-Token": s.vault_token}
                resp = _rq.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                jd = resp.json()
                # KV v2 -> {data: {data: {...}}}; KV v1 -> {data: {...}}
                payload = jd.get("data", {})
                if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], dict):
                    payload = payload["data"]
                obj = {str(k): str(v) for k, v in (payload or {}).items()}
                self._cache = obj.copy()
                return obj
            except Exception:
                return {}
        # Backend ENV (.env)
        if backend == "env":
            try:
                p = Path(".env")
                if not p.exists():
                    return {}
                lines = p.read_text(encoding="utf-8").splitlines()
                out: Dict[str, str] = {}
                for ln in lines:
                    if not ln or ln.strip().startswith("#") or "=" not in ln:
                        continue
                    k, v = ln.split("=", 1)
                    out[k.strip()] = v.strip()
                self._cache = out.copy()
                return out
            except Exception:
                return {}
        # Backend archivo (Fernet)
        if not self.store_path.exists():
            return {}
        token = self.store_path.read_bytes()
        data = self._fernet.decrypt(token)
        obj = {k: v for k, v in json.loads(data.decode("utf-8")).items()}
        self._cache = obj.copy()
        return obj

    # Roles (stub sencillo con basedir)
    def has_role(self, role: str) -> bool:
        role = role.lower().strip()
        # En una implementación real, se validaría contra identidad/ACL
        # Aquí usamos un archivo marcador: data/role_operator.flag
        flag = self.base_dir / f"role_{role}.flag"
        return flag.exists()

    # ===== 2FA (TOTP) =====
    def get_totp_secret(self) -> str | None:
        try:
            if not self._cache:
                self.load()
            return self._cache.get("TOTP_SECRET")
        except Exception:
            return None

    def set_totp_secret(self, secret_b32: str) -> None:
        if not secret_b32:
            raise ValueError("Se requiere secreto base32")
        self.save({"TOTP_SECRET": secret_b32})

    def generate_totp_secret(self, length: int = 20) -> str:
        import base64 as _b64
        raw = os.urandom(int(length))
        b32 = _b64.b32encode(raw).decode("utf-8").rstrip("=")
        self.set_totp_secret(b32)
        return b32

    def provisioning_uri(self, account_name: str, issuer: str) -> str:
        secret = self.get_totp_secret() or self.generate_totp_secret()
        from urllib.parse import quote
        label = f"{issuer}:{account_name}"
        params = f"secret={secret}&issuer={quote(issuer)}&algorithm=SHA1&digits=6&period=30"
        return f"otpauth://totp/{quote(label)}?{params}"

    @staticmethod
    def _b32_to_bytes(b32: str) -> bytes:
        import base64 as _b64
        # Normalizar padding
        s = b32.strip().replace(" ", "").upper()
        pad = "=" * ((8 - len(s) % 8) % 8)
        return _b64.b32decode(s + pad)

    def totp_code(self, secret_b32: str, for_time: int | None = None, period: int = 30, digits: int = 6) -> str:
        ts = int(for_time if for_time is not None else time.time())
        counter = ts // int(period)
        key = self._b32_to_bytes(secret_b32)
        msg = counter.to_bytes(8, "big")
        h = hmac.new(key, msg, hashlib.sha1).digest()
        o = h[-1] & 0x0F
        code = (int.from_bytes(h[o:o+4], "big") & 0x7FFFFFFF) % (10 ** digits)
        return str(code).zfill(digits)

    def verify_totp(self, code: str, secret_b32: str | None = None, period: int = 30, digits: int = 6, window: int = 1) -> bool:
        secret = secret_b32 or self.get_totp_secret()
        if not secret:
            return False
        try:
            now = int(time.time())
            for offset in range(-window, window + 1):
                cand = self.totp_code(secret, for_time=now + offset * period, period=period, digits=digits)
                if hmac.compare_digest(cand, str(code)):
                    return True
            return False
        except Exception:
            return False

    # Rotación de clave (re-encripta la store con nueva clave Fernet)
    def rotate_key(self) -> None:
        data = self.load()
        new_key = Fernet.generate_key()
        self.key_path.write_bytes(new_key)
        self._fernet = Fernet(new_key)
        self.save(data)


