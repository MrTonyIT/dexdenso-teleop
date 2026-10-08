"""
Module: test_integration_network.py
Description: Integration test verifying socket bridge communication between
             WincapsBridge and Mock RC8 Server at 50 Hz with PACScript Input #1 validation.
"""

import time
import socket
import threading
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from wincaps_bridge import WincapsBridge


def test_network_bridge():
    """Verify high-throughput socket communication and telemetry packet parsing."""
    print("Initiating Socket Bridge integration test suite...")
    received_packets = []
    server_running = True

    ready_event = threading.Event()
    server_port = [0]

    def server_thread():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        server_port[0] = srv.getsockname()[1]
        srv.listen(1)
        ready_event.set()
        srv.settimeout(2.0)
        try:
            conn, addr = srv.accept()
            buf = ""
            while server_running:
                try:
                    data = conn.recv(1024)
                    if not data:
                        break
                    buf += data.decode("ascii")
                    while "\r\n" in buf:
                        line, buf = buf.split("\r\n", 1)
                        if line.strip():
                            received_packets.append(line.strip())
                except socket.timeout:
                    continue
            conn.close()
        except socket.timeout:
            pass
        finally:
            srv.close()

    th = threading.Thread(target=server_thread, daemon=True)
    th.start()
    ready_event.wait(timeout=2.0)
    time.sleep(0.05)

    bridge = WincapsBridge(host="127.0.0.1", port=server_port[0], send_rate_hz=50.0)
    assert bridge.connect(), "Failed to connect to mock RC8 server"

    # Send 25 telemetry cycles at 50 Hz interval (20ms)
    start_t = time.perf_counter()
    for i in range(25):
        bridge.send_pose(
            x=425.0 + i,
            y=-10.0 + i * 0.5,
            z=300.0,
            rx=180.0,
            ry=0.0,
            rz=0.0,
            gripper=(1 if i % 2 == 0 else 0),
            force=False
        )
        time.sleep(0.02)

    total_time = time.perf_counter() - start_t
    print(f"Dispatched 25 packets in {total_time:.3f}s. Received: {len(received_packets)}")

    bridge.close()
    server_running = False
    th.join(timeout=1.0)

    assert len(received_packets) >= 20, f"Insufficient packet delivery rate: {len(received_packets)}"
    # Validate payload integrity of initial packet
    first_toks = received_packets[0].split(",")
    assert len(first_toks) == 7, f"Packet does not contain 7 tokens: {received_packets[0]}"
    assert float(first_toks[0]) == 425.0
    assert float(first_toks[3]) == 180.0

    print("===> NETWORK INTEGRATION TESTS PASSED (100%) <===")


if __name__ == "__main__":
    test_network_bridge()
