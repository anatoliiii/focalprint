#!/usr/bin/env python3
"""Grab raw frames from a FocalTech FT9201 USB sensor (2808:c652).

Protocol, as observed from the vendor library:
  frame   = 02 | len (u16 BE) | payload | xor(len_hi, len_lo, payload)
  reply payload starts with 04 (OK)
  80 05 -> id, 80 03 -> width?, 80 04 -> height?, 82 78 -> setup,
  80 02 -> finger status (00 none, 01 present), 81 -> image (10245-byte reply).

Stop fprintd first (`sudo systemctl stop fprintd`), then run as root.
Frames are saved only to the directory given on the command line; nothing
is sent anywhere. Prints a per-frame summary that is safe to share."""
import argparse, functools, operator, os, sys, time

import usb.core, usb.util

VID, PID = 0x2808, 0xC652
EP_OUT, EP_IN = 0x03, 0x81


def frame(payload: bytes) -> bytes:
    n = len(payload).to_bytes(2, 'big')
    return b'\x02' + n + payload + bytes([functools.reduce(operator.xor, n + payload, 0)])


def unframe(raw: bytes) -> bytes:
    if len(raw) < 4 or raw[0] != 0x02:
        raise ValueError(f'bad frame start: {raw[:8].hex()}')
    n = int.from_bytes(raw[1:3], 'big')
    body, chk = raw[3:3 + n], raw[3 + n]
    if len(body) != n:
        raise ValueError(f'short frame: want {n}, got {len(body)}')
    if functools.reduce(operator.xor, raw[1:3] + body, 0) != chk:
        raise ValueError('xor mismatch')
    return body


class Sensor:
    def __init__(self):
        self.dev = usb.core.find(idVendor=VID, idProduct=PID)
        if self.dev is None:
            sys.exit('sensor 2808:c652 not found')
        if self.dev.is_kernel_driver_active(0):
            self.dev.detach_kernel_driver(0)
        usb.util.claim_interface(self.dev, 0)

    def cmd(self, payload: bytes, read=64, timeout=1000) -> bytes:
        self.dev.write(EP_OUT, frame(payload), timeout)
        reply = unframe(bytes(self.dev.read(EP_IN, read, timeout)))
        if reply[:1] != b'\x04':
            raise RuntimeError(f'cmd {payload.hex()} -> {reply.hex()}')
        return reply[1:]

    def close(self):
        usb.util.release_interface(self.dev, 0)
        usb.util.dispose_resources(self.dev)


def summary(img: bytes) -> str:
    """Stats that reveal the pixel format without revealing the print."""
    out = [f'{len(img)} bytes, head {img[:8].hex()}']
    if len(img) % 2 == 0:
        for order in ('little', 'big'):
            px = [int.from_bytes(img[i:i + 2], order) for i in range(0, len(img), 2)]
            rough = sum(abs(a - b) for a, b in zip(px, px[1:])) / (len(px) - 1)
            out.append(f'u16 {order:6s}: min {min(px):5d} max {max(px):5d} '
                       f'mean {sum(px) / len(px):8.1f} mean|dx| {rough:8.1f}')
    px = list(img)
    rough = sum(abs(a - b) for a, b in zip(px, px[1:])) / (len(px) - 1)
    out.append(f'u8          : min {min(px):5d} max {max(px):5d} mean {sum(px) / len(px):8.1f} mean|dx| {rough:8.1f}')
    return '\n    '.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outdir', help='where raw frames are written (e.g. /dev/shm/ft9201)')
    ap.add_argument('-n', type=int, default=3, help='number of touches to grab')
    args = ap.parse_args()
    os.makedirs(args.outdir, mode=0o700, exist_ok=True)

    s = Sensor()
    try:
        ident = s.cmd(b'\x80\x05')
        w, h = s.cmd(b'\x80\x03')[0], s.cmd(b'\x80\x04')[0]
        s.cmd(b'\x82\x78')
        print(f'id {ident.hex()}  geometry {w}x{h}')
        for k in range(args.n):
            print(f'[{k + 1}/{args.n}] touch the sensor...', flush=True)
            while s.cmd(b'\x80\x02')[:1] != b'\x01':
                time.sleep(0.02)
            img = s.cmd(b'\x81', read=22534, timeout=2000)
            path = os.path.join(args.outdir, f'frame{k:02d}.bin')
            with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'wb') as f:
                f.write(img)
            print(f'    saved {path}\n    {summary(img)}', flush=True)
            while s.cmd(b'\x80\x02')[:1] == b'\x01':
                time.sleep(0.02)
    finally:
        s.close()


if __name__ == '__main__':
    main()
