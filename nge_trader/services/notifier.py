from __future__ import annotations

import requests

from nge_trader.config.settings import Settings


class Notifier:
    """Envia notificaciones a Telegram si está configurado."""

    def __init__(self) -> None:
        self.settings = Settings()

    def send(self, message: str) -> None:
        token = self.settings.telegram_bot_token
        chat_id = self.settings.telegram_chat_id
        if not (token and chat_id):
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
        except Exception:
            pass

    def send_alert_if_threshold(self, key: str, value: float, threshold: float, above: bool = True) -> None:
        try:
            ok = (value > threshold) if above else (value < threshold)
            if ok:
                self.send(f"Alerta: {key}={(value):.4f} umbral={'>' if above else '<'} {threshold}")
        except Exception:
            pass

    def send_email(self, subject: str, body: str) -> None:
        try:
            if not (self.settings.smtp_host and self.settings.smtp_user and self.settings.smtp_password and self.settings.smtp_to):
                return
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(body, _charset="utf-8")
            msg["Subject"] = subject
            msg["From"] = self.settings.smtp_user
            msg["To"] = self.settings.smtp_to
            server = smtplib.SMTP(self.settings.smtp_host, int(self.settings.smtp_port or 587), timeout=10)
            try:
                server.starttls()
            except Exception:
                pass
            server.login(self.settings.smtp_user, self.settings.smtp_password)
            server.sendmail(self.settings.smtp_user, [self.settings.smtp_to], msg.as_string())
            server.quit()
        except Exception:
            pass

    def send_document(self, file_path: str, caption: str | None = None) -> None:
        token = self.settings.telegram_bot_token
        chat_id = self.settings.telegram_chat_id
        if not (token and chat_id):
            return
        url = f"https://api.telegram.org/bot{token}/sendDocument"
        try:
            with open(file_path, "rb") as f:
                files = {"document": (file_path, f)}
                data = {"chat_id": chat_id}
                if caption:
                    data["caption"] = caption
                requests.post(url, data=data, files=files, timeout=30)
        except Exception:
            pass

    def send_email_with_attachment(self, subject: str, body: str, file_path: str) -> None:
        try:
            if not (self.settings.smtp_host and self.settings.smtp_user and self.settings.smtp_password and self.settings.smtp_to):
                return
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from email.mime.base import MIMEBase
            from email import encoders
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self.settings.smtp_user
            msg['To'] = self.settings.smtp_to
            msg.attach(MIMEText(body, _charset='utf-8'))
            with open(file_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{file_path}"')
                msg.attach(part)
            server = smtplib.SMTP(self.settings.smtp_host, int(self.settings.smtp_port or 587), timeout=10)
            try:
                server.starttls()
            except Exception:
                pass
            server.login(self.settings.smtp_user, self.settings.smtp_password)
            server.sendmail(self.settings.smtp_user, [self.settings.smtp_to], msg.as_string())
            server.quit()
        except Exception:
            pass


