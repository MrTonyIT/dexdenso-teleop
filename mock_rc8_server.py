"""
Module: mock_rc8_server.py
Description: Mock server simulating the DENSO RC8 controller (WINCAPS III) TCP socket.
Features:
  - Listens on TCP port 5000 (or user-specified port).
  - Parses CRLF-terminated telemetry packets (X, Y, Z, Rx, Ry, Rz, Grip) matching PACScript Input #1.
  - Measures real-time packet throughput (Hz) and displays active gripper state.
"""

import socket
import time
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def run_mock_server(host: str = "127.0.0.1", port: int = 5000):
    """Run mock TCP server for standalone testing without physical robot hardware."""
    print("=" * 65)
    print(f"  [MOCK RC8 SERVER] Starting Mock Controller Server on {host}:{port}...")
    print("=" * 65)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)

    print(f"[MOCK RC8] Awaiting connection from DexDenso Teleoperation Bridge...")

    try:
        while True:
            client_sock, addr = server_sock.accept()
            print(f"[MOCK RC8] ==> Client connected successfully from: {addr}")

            buffer = ""
            packet_count = 0
            start_time = time.perf_counter()
            last_rate_time = start_time

            try:
                while True:
                    data = client_sock.recv(1024)
                    if not data:
                        print("[MOCK RC8] Client disconnected gracefully.")
                        break

                    buffer += data.decode("ascii", errors="ignore")

                    # Parse packets terminated by \r\n identical to PACScript Input #1 format
                    while "\r\n" in buffer:
                        line, buffer = buffer.split("\r\n", 1)
                        if not line.strip():
                            continue

                        tokens = line.strip().split(",")
                        if len(tokens) == 7:
                            x, y, z, rx, ry, rz, grip = (
                                float(tokens[0]),
                                float(tokens[1]),
                                float(tokens[2]),
                                float(tokens[3]),
                                float(tokens[4]),
                                float(tokens[5]),
                                int(tokens[6]),
                            )

                            packet_count += 1
                            now = time.perf_counter()

                            if now - last_rate_time >= 1.0:
                                rate = packet_count / (now - last_rate_time)
                                grip_str = "CLOSED (IO 128 ON)" if grip == 1 else "OPEN (IO 128 OFF)"
                                print(
                                    f"[RC8 MOCK @ {rate:4.1f} Hz] "
                                    f"Pos: X={x:5.1f} Y={y:5.1f} Z={z:5.1f} mm | "
                                    f"Rot: ({rx:.0f}, {ry:.0f}, {rz:.0f}) | "
                                    f"Gripper: {grip_str}"
                                )
                                packet_count = 0
                                last_rate_time = now

            except ConnectionResetError:
                print("[MOCK RC8] Connection reset by client.")
            finally:
                client_sock.close()

    except KeyboardInterrupt:
        print("\n[MOCK RC8] Server shutdown requested by user.")
    finally:
        server_sock.close()


if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_mock_server(port=port)
