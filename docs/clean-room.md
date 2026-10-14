# How the sensor protocols were recovered

This project talks to two FocalTech fingerprint sensors without any vendor
driver, library, SDK or matcher. This document records where every piece of
protocol knowledge came from, and the rules followed, so that the code can be
audited and extended the same way.

## Rules

1. **No vendor code was read.** No proprietary driver, library or firmware was
   disassembled, decompiled, debugged, patched or string-dumped. No vendor SDK,
   header or datasheet under NDA was used.
2. **Only observable behaviour.** Protocol facts come from what a device
   visibly does: bytes on the bus, ACPI tables published by the laptop
   firmware, the USB descriptors a device reports, and the responses of the
   sensor to our own requests.
3. **Open sources only, with attribution.** Where an existing open-source
   driver informed the work, it is named below with its licence.
4. **No vendor files in this repository.** No firmware images, vendor
   binaries, captured traffic or biometric data are committed (see
   `.gitignore`). Sensor firmware stays whatever the device already runs.
5. **Every claim is testable.** Each recovered exchange is either covered by a
   unit test against recorded bytes or was verified on hardware, as noted.

A strict clean room separates the people who observe from the people who
write code, and has the observers hand over only a written specification.
This was a small project: the same people observed and implemented. The
guarantee here is narrower but still meaningful: the implementation was
derived only from the observations and open sources listed below, never from
vendor code.

## FTE4800 / FT9368 (SPI, TECNO MEGABOOK T1)

| Knowledge | Source |
|---|---|
| ACPI device `FTE4800` on SPI, reset GPIO and interrupt line | The laptop's own ACPI tables (`_CRS`), read with standard Linux tools |
| Command framing, wake / status / frame commands, status block layout, 64×80 frame | The open-source FocalTech driver in [libfprint-fte3600](https://github.com/SamSeven777/libfprint-fte3600) (LGPL-2.1-or-later), whose authors describe it as an independent clean-room implementation. This project reimplements those facts in Rust; no code was copied |
| Chip select is active **high** although ACPI says active low | Our experiment: the sensor answers only with `SPI_CS_HIGH` |
| Sensor dozes ~100 ms after the last command and then clocks out a repeated filler byte | Our experiments on firmware 0x11 and 0x14 |
| A latched touch notification holds the edge-triggered interrupt high until a short frame read clears it | Our experiments (interrupt counter vs. status block) |
| While a finger rests, the contact flag pulses instead of staying set; a release is ~700 ms without contact | Our experiments |
| Status byte `0x22` means busy and is transient | Our experiments |

Verified on hardware: identification, wake, capture via interrupt, release
detection, enrolment and verification through the daemon.

## FT9201 (USB `2808:c652`, TECNO MegaBook K16S)

| Knowledge | Source |
|---|---|
| Vendor-class interface 0, bulk OUT `0x03`, bulk IN `0x81`, 64-byte packets | USB descriptors reported by the device |
| Frame format `02 | len (u16 BE) | payload | xor(len, payload)`, status byte `04` | Black-box capture of USB traffic (Linux `usbmon`) while the proprietary library was used normally by the laptop's owner |
| Start-up `80 05`, `80 03`, `80 04`, `82 78`; poll `80 02`; image `81` | Same capture: request/response pairs and their timing |
| Geometry 64×80, image reply 10245 bytes = framing + status + 64·80·2 | Answers to `80 03` / `80 04` and the reply length in the capture |
| Vendor control requests (`0x34`, `0x22`, `0x3a`, `0x64`) are not needed | Same capture: the device stalls every one of them |

The capture recorded only bus traffic. The proprietary library itself was
never inspected. `usbmon` does not store the payload of transfers larger than
16 KiB, so pixel contents were not observed this way. They are read directly
with `tools/ft9201-grab.py`, which replays the observed protocol.

Status: protocol implemented and unit-tested byte for byte against the
capture, and verified on hardware (TECNO MegaBook K16S, Fedora): 16-bit pixels
are little-endian, and capture, release detection, enrolment and verification
work through `focalprintd`.

## Adding a sensor the same way

1. Record what the laptop firmware publishes: `acpidump` / `lsusb -v`.
2. Record traffic while the existing driver works normally. For USB:
   `modprobe usbmon`, then `tshark -i usbmonN -w trace.pcapng` (or Wireshark),
   performing one enrolment and one verification. Keep each capture to one
   person's consenting fingers and do not publish it: image transfers contain
   biometric data.
3. Write down the protocol from the trace alone, then implement it against a
   scripted transport (see the `Script` mocks in `crates/focalprint-sensor/src`) with the
   recorded bytes as tests.
4. Never open the vendor binary to "just check" something; ask the device
   instead.
