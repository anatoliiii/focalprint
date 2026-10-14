# Sensor protocols

Both sensors deliver raw 64×80 frames; all matching happens on the host.
Provenance of every fact below: [clean-room.md](clean-room.md).

## FT9368 over SPI (ACPI `FTE4800`)

Bus: SPI with **`SPI_CS_HIGH`** (the sensor ignores an active-low chip
select despite what ACPI declares). Reset is an ACPI GpioIo line; the touch
interrupt is edge-triggered.

Every exchange is one full-duplex transfer:

```text
tx: cmd_hi cmd_lo len_hi len_lo  [3 turnaround bytes if len > 0]  [len filler bytes]
rx: ……………………… header (4, or 7 if len > 0) ………………………  payload × len
```

| Command | Payload | Meaning |
|---|---|---|
| `FF 00` | 0 | wake from doze |
| `91 80` | 32 | status block |
| `90 80` | 5120 | latched frame, 64×80, 8 bits per pixel |
| `90 80` | 6 | short read: only clears the touch notification |

Status block (32 bytes):

| Bytes | Meaning |
|---|---|
| 1, 2 | `11 11` = finger in contact; byte 2 = `22` while busy (retry) |
| 19–20 | chip id, big-endian, `0x9368` |
| 21 | application firmware (0x11, 0x13, 0x14 seen) |
| 0–1 | `FF FF` = no sensor on the bus |

Behaviour:

- **Doze.** ~100 ms after the last command the sensor sleeps and answers with
  one repeated filler byte. Send `FF 00`, wait ~10 ms, probe the status, and
  repeat until it stops answering filler.
- **Notification latch.** A touch latches a notification that keeps the
  interrupt line high, so a later touch raises no new edge. A short `90 80`
  read of 6 bytes clears it; do that before waiting for a touch.
- **Touch.** The interrupt is only a hint; confirm with the status block.
- **Release.** While a finger rests, the contact flag pulses rather than
  staying set. Treat ~700 ms without contact as a release.

Kernel interface (`/dev/focaltech-fp`, single open, ABI 1, ioctl magic `0xF4`):

| ioctl | Argument | Effect |
|---|---|---|
| `_IOR(0xF4, 0, Info)` | `{abi, max_transfer, speed_hz, mode}` | static facts |
| `_IO(0xF4, 1)` | — | pulse the reset line |
| `_IOW(0xF4, 2, Transfer)` | `{tx, rx, len, reserved}` | one SPI transfer, ≤ 8192 bytes |
| `_IOWR(0xF4, 3, WaitIrq)` | `{seen, timeout_ms, reserved}` | wait until the interrupt counter passes `seen`; returns the counter (`ETIMEDOUT` on timeout) |

The module wipes its transfer buffers after use.

## FT9201 over USB (`2808:c652`)

Interface 0 (vendor class), bulk OUT `0x03`, bulk IN `0x81`, 64-byte
packets, full speed. No kernel module is needed: usbfs is enough.

Requests and replies share one frame format:

```text
02 | len (u16 big-endian) | payload × len | xor(len_hi, len_lo, payload…)
```

A reply payload starts with a status byte, `04` = OK.

| Request | Reply payload | Meaning |
|---|---|---|
| `80 05` | `04 62 93` | id |
| `80 03` | `04 40` | width (64) |
| `80 04` | `04 50` | height (80) |
| `82 78` | `04` | set-up step sent by the original software; meaning unknown |
| `80 02` | `04 00` / `04 01` | frame ready? |
| `81` | `04` + 10240 bytes | frame, 64×80 × 16 bits |

Start-up is the first four requests in that order. Then poll `80 02` every
20 ms; on `01`, read the frame with `81` (request ≥ 10245 bytes in whole
64-byte packets; the reply ends with a short packet). While a finger rests,
a new frame becomes ready every 0.1–0.4 s; a release is ~700 ms without one.

Open questions, pending hardware tests: pixel byte order (little-endian
assumed) and pixel polarity. The driver contrast-stretches the 16-bit samples
(2nd–98th percentile) to 8 bits. Until confirmed, both can be switched without
rebuilding through `FOCALPRINT_FT9201_PIXELS` (comma-separated `le`/`be`,
`invert`); see [fedora-ft9201.md](fedora-ft9201.md#check-the-sensor).
