from __future__ import annotations

import socket
import struct
import time


def get_ntp_time(server: str = "pool.ntp.org", timeout: float = 2.0) -> float:
    """Devuelve tiempo NTP en segundos desde epoch (UTC)."""
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(timeout)
    msg = b"\x1b" + 47 * b"\0"
    addr = (server, 123)
    t0 = time.time()
    client.sendto(msg, addr)
    data, _ = client.recvfrom(1024)
    t1 = time.time()
    if data:
        unpacked = struct.unpack("!12I", data[:48])
        # Transmit Timestamp (seconds since 1900)
        tx_ts = unpacked[10] + float(unpacked[11]) / 2**32
        ntp_epoch = 2208988800  # 1970-01-01 offset
        # Cristaliza al promedio del RTT
        rtt = (t1 - t0) / 2.0
        try:
            from nge_trader.services.metrics import observe_latency
            observe_latency("ntp_poll_latency_ms", (t1 - t0) * 1000.0)
        except Exception:
            pass
        return tx_ts - ntp_epoch + rtt
    raise RuntimeError("Sin respuesta NTP")


def estimate_skew_ms(server: str = "pool.ntp.org") -> float:
    try:
        ntp_ts = get_ntp_time(server=server)
        local_ts = time.time()
        skew = float((local_ts - ntp_ts) * 1000.0)
        try:
            from nge_trader.services.metrics import set_metric
            set_metric("time_skew_ms", skew)
        except Exception:
            pass
        return skew
    except Exception:
        try:
            from nge_trader.services.metrics import inc_metric
            inc_metric("ntp_poll_failures_total", 1.0)
        except Exception:
            pass
        return 0.0


