"""
Module: wincaps_bridge.py
Description: Ultra-low latency asynchronous TCP/UDP teleoperation bridge for
             DENSO RC8 / WINCAPS III virtual controller and NVIDIA Isaac Sim.
Technical Highlights:
  1. Asynchronous non-blocking background connection worker (80ms connection timeout) - ZERO FPS drops!
  2. TCP_NODELAY flag enabled to bypass Nagle's algorithm for instant packet delivery.
  3. PACScript Input #1 compliant serialization format.
  4. Real-time dual UDP broadcast to Isaac Sim (ports 5008 / 5005).
  5. Deterministic rate limiting (40-60 Hz).
"""

import sys
import socket
import time
import logging
import threading
from typing import Optional, Tuple, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger = logging.getLogger("WincapsBridge")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(handler)


class WincapsBridge:
    """
    Direct TCP/UDP telemetry bridge for DENSO RC8 (WINCAPS III) and NVIDIA Isaac Sim.
    Executes asynchronously to prevent camera acquisition stalls, maintaining a stable 60 FPS.
    Automatically reconnects in background when WINCAPS enters playback mode.
    """

    def __init__(
        self,
        host: str = "192.168.1.131",
        port: int = 49152,
        send_rate_hz: float = 60.0,
        auto_reconnect: bool = True,
        reconnect_interval_sec: float = 0.25,
    ):
        self.host = host
        self.port = port
        self.target_interval: float = 1.0 / send_rate_hz
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval_sec = reconnect_interval_sec

        # Real-time UDP socket for NVIDIA Isaac Sim digital twin synchronization
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.isaac_udp_target: Tuple[str, int] = ("127.0.0.1", 5005)

        self.sock: Optional[socket.socket] = None
        self.is_connected: bool = False
        self.last_send_time: float = 0.0
        self.last_connect_attempt: float = 0.0

        # Background asynchronous connection state & locks
        self._is_connecting: bool = False
        self._conn_lock = threading.RLock()
        self._sock_lock = threading.RLock()

        # Telemetry metrics
        self.packets_sent: int = 0
        self.actual_send_rate: float = 0.0
        self._rate_calc_time: float = time.perf_counter()
        self._rate_packet_count: int = 0

    def _get_target_candidates(self) -> List[Tuple[str, int]]:
        """Return candidate endpoints for DENSO RC8 Ethernet Server (default port 49152)."""
        candidates = [
            ("192.168.1.131", 49152),
            ("127.0.0.1", 49152),
        ]
        if (self.host, self.port) not in candidates:
            candidates.insert(0, (self.host, self.port))
        return candidates

    def connect(self, timeout_sec: float = 0.15) -> bool:
        """Trigger background connection and poll up to timeout_sec for immediate verification."""
        self.start_background_connect()
        start = time.perf_counter()
        while time.perf_counter() - start < timeout_sec:
            if self.is_connected:
                return True
            time.sleep(0.01)
        return self.is_connected

    def start_background_connect(self) -> None:
        """Launch background worker thread to establish TCP connection with WINCAPS III."""
        with self._conn_lock:
            if self._is_connecting or self.is_connected:
                return
            self._is_connecting = True
            t = threading.Thread(target=self._async_connect_worker, daemon=True)
            t.start()

    def _async_connect_worker(self) -> None:
        """Asynchronous connection worker (<80ms connect latency) preventing main loop stalls."""
        self.last_connect_attempt = time.perf_counter()
        candidates = self._get_target_candidates()

        for target_host, target_port in candidates:
            if not self.auto_reconnect or self.is_connected:
                break
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
                s.settimeout(0.08)  # 80ms fast timeout

                s.connect((target_host, target_port))
                s.settimeout(0.01)

                with self._sock_lock:
                    if self.sock is not None:
                        try:
                            self.sock.close()
                        except Exception:
                            pass
                    self.sock = s
                    self.host = target_host
                    self.port = target_port
                    self.is_connected = True

                logger.info(f"==> [CONNECTED] WINCAPS III RC8 ({self.host}:{self.port})!")
                with self._conn_lock:
                    self._is_connecting = False
                return
            except (socket.error, ConnectionRefusedError, TimeoutError, OSError):
                try:
                    s.close()
                except Exception:
                    pass
                continue

        with self._conn_lock:
            self._is_connecting = False

    def send_joints(
        self,
        j1: float,
        j2: float,
        j3: float,
        j4: float,
        j5: float,
        j6: float,
        gripper: int,
        fk_x: float = 0.0,
        fk_y: float = 0.0,
        fk_z: float = 0.0,
        force: bool = False,
    ) -> bool:
        """
        Dispatch 6-DOF joint control telemetry to Isaac Sim and DENSO RC8:
        - Isaac Sim (UDP 5008 / 5005): JOINT,j1,j2,j3,j4,j5,j6,gripper
        - WINCAPS RC8 (TCP): j1,j2,j3,j4,j5,j6,gripper
        """
        now = time.perf_counter()
        if not force and (now - self.last_send_time < self.target_interval):
            return False

        # 1. Zero-latency UDP broadcast to NVIDIA Isaac Sim
        udp_payload = f"JOINT,{j1:.2f},{j2:.2f},{j3:.2f},{j4:.2f},{j5:.2f},{j6:.2f},{gripper}\r\n"
        packet_bytes = udp_payload.encode("ascii")

        if self.udp_sock is None:
            try:
                self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            except Exception:
                pass

        if self.udp_sock is not None:
            try:
                self.udp_sock.sendto(packet_bytes, ("127.0.0.1", 5008))
                self.udp_sock.sendto(packet_bytes, ("127.0.0.1", 5005))
                self.last_send_time = now
                self.packets_sent += 1
                self._rate_packet_count += 1
            except Exception:
                pass

        if now - self._rate_calc_time >= 1.0:
            dt = now - self._rate_calc_time
            self.actual_send_rate = self._rate_packet_count / dt
            self._rate_packet_count = 0
            self._rate_calc_time = now

        # 2. TCP packet dispatch to WINCAPS RC8
        tcp_payload = f"{j1:.2f},{j2:.2f},{j3:.2f},{j4:.2f},{j5:.2f},{j6:.2f},{gripper}\r\n"
        tcp_bytes = tcp_payload.encode("ascii")

        if not self.is_connected:
            if self.auto_reconnect and (now - self.last_connect_attempt >= 3.0):
                self.start_background_connect()
            return True

        try:
            with self._sock_lock:
                if self.sock is not None and self.is_connected:
                    self.sock.sendall(tcp_bytes)
                    self.last_send_time = now
                    return True
        except (socket.error, BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"Connection lost to RC8 controller: {e}")
            self.close()
            self.start_background_connect()
            return False

        return False

    def send_pose(
        self,
        x: float,
        y: float,
        z: float,
        rx: float,
        ry: float,
        rz: float,
        gripper: int,
        force: bool = False,
    ) -> bool:
        """
        Dispatch Cartesian pose and gripper telemetry.
        Fully non-blocking: returns immediately if socket is unavailable without degrading vision FPS.
        """
        now = time.perf_counter()

        # Enforce rate limiting
        if not force and (now - self.last_send_time < self.target_interval):
            return False

        # Prepare payload: X,Y,Z,Rx,Ry,Rz,Gripper + CRLF
        payload = f"{x:.1f},{y:.1f},{z:.1f},{rx:.1f},{ry:.1f},{rz:.1f},{gripper}\r\n"
        packet_bytes = payload.encode("ascii")

        # 1. Zero-latency UDP broadcast to NVIDIA Isaac Sim
        if self.udp_sock is None:
            try:
                self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            except Exception:
                pass

        if self.udp_sock is not None:
            try:
                self.udp_sock.sendto(packet_bytes, ("127.0.0.1", 5008))
                self.udp_sock.sendto(packet_bytes, ("127.0.0.1", 5005))
                self.last_send_time = now
                self.packets_sent += 1
                self._rate_packet_count += 1
            except Exception:
                pass

        # Calculate throughput every second
        if now - self._rate_calc_time >= 1.0:
            dt = now - self._rate_calc_time
            self.actual_send_rate = self._rate_packet_count / dt
            self._rate_packet_count = 0
            self._rate_calc_time = now

        # 2. Reconnect trigger if TCP connection to WINCAPS RC8 dropped
        if not self.is_connected:
            if self.auto_reconnect and (now - self.last_connect_attempt >= 3.0):
                self.start_background_connect()
            return True

        try:
            with self._sock_lock:
                if self.sock is not None and self.is_connected:
                    self.sock.sendall(packet_bytes)
                    self.last_send_time = now
                    return True
        except (socket.error, BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"Connection lost to RC8 controller: {e}")
            self.close()
            self.start_background_connect()
            return False

        return False

    def close(self) -> None:
        """Close TCP socket connection with WINCAPS RC8 while keeping UDP alive for Isaac Sim."""
        with self._sock_lock:
            if self.sock is not None:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None
            self.is_connected = False

    def disconnect_all(self) -> None:
        """Close all sockets (TCP and UDP) during application shutdown."""
        self.close()
        if self.udp_sock is not None:
            try:
                self.udp_sock.close()
            except Exception:
                pass
            self.udp_sock = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DexDenso Teleoperation Bridge")
    parser.add_argument("--host", type=str, default="192.168.1.131")
    parser.add_argument("--port", type=int, default=49152)
    parser.add_argument("--rate", type=float, default=40.0)
    args = parser.parse_args()

    print(f"Initializing WincapsBridge to {args.host}:{args.port}...")
    bridge = WincapsBridge(host=args.host, port=args.port, send_rate_hz=args.rate)
    success = bridge.connect(timeout_sec=0.5)

    if success:
        print(f"Connected successfully to {bridge.host}:{bridge.port}!")
        for i in range(20):
            bridge.send_pose(425.0, 0.0, 325.0, 180.0, 0.0, 0.0, 0)
            time.sleep(0.025)
        print(f"Transmitted {bridge.packets_sent} packets successfully!")
    else:
        print(f"Could not connect to {args.host}:{args.port} (RC8 or PACScript may be offline).")
    bridge.close()
