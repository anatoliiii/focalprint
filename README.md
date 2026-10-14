# focalprint

**English** · [Русский](README.ru.md)

Open fingerprint stack in Rust for FocalTech sensors found in TECNO laptops,
written without vendor drivers, libraries or matchers
([how](docs/clean-room.md)).

| Sensor | Bus | Laptop | Status |
|---|---|---|---|
| FTE4800 / FT9368 | SPI (ACPI) | TECNO MEGABOOK T1 | works: capture, enrol, verify, `sudo`/lock screen via fprintd API |
| FT9201 | USB `2808:c652` | TECNO MegaBook K16S | works: capture, enrol, verify, unlock; verified on a K16S under Fedora ([Fedora guide](docs/fedora-ft9201.md)) |

It replaces libfprint + fprintd entirely: `focalprintd` serves fprintd's
`net.reactivated.Fprint` D-Bus API, so `pam_fprintd`, hyprlock, GNOME and
KDE keep working unchanged.

> **Beta.** Keep a password login. Matching quality is still being measured
> (see [Matcher](#matcher)).

## Layout

| Path | What |
|---|---|
| `kernel/` | Rust kernel module for the SPI sensor: `/dev/focaltech-fp`, raw transfers, reset, interrupt waits ([ABI](docs/protocols.md#ft9368-over-spi-acpi-fte4800)) |
| `crates/focalprint-sensor` | Sensor protocols: FT9368 over the module, FT9201 over usbfs (`nusb`), common `Sensor` trait |
| `crates/focalprintd` | Daemon with fprintd's D-Bus API, encrypted template store |
| `crates/focalprint-net` | Neural matcher (FpNet ONNX via `tract`) and dense alignment scoring |
| `crates/focalprint-dataset` | Opt-in encrypted research collection |
| `crates/focalprint` | Diagnostics: `info`, `capture`, `identify`, `neural`, `dataset` … |
| `crates/focalprint-match` | Earlier classical matcher (keypoints + RANSAC), kept for comparison |
| `research/` | Python/PyTorch: FpNet training, evaluation, synthetic fingerprints, distillation, ONNX export |
| `models/` | Ready-to-use FpNet trained without private frames, download from the [release](https://github.com/anatoliiii/focalprint/releases/tag/v0.1.0) ([provenance, data sources and quality](models/README.md)) |
| `tools/` | `dev-switch` (hand the sensor to this stack and back), kernel toolchain helper, FT9201 frame grabber |

## Privacy and security

Fingerprint frames are biometric data and are treated as secrets:

- frames live only in the daemon's memory, which is locked (`mlockall`) and
  not dumpable, and are wiped on drop (`zeroize`), including USB/SPI buffers;
- frames and logs never mix: `Image` has a `Debug` that prints no pixels;
- templates are encrypted with XChaCha20-Poly1305 under a key sealed to the
  TPM by `systemd-creds` (`LoadCredentialEncrypted=`), bound to user and
  finger as associated data;
- the research collection is off unless enabled with
  `focalprint dataset enable`, and is encrypted the same way;
- `focalprint dump` writes only to `/dev/shm` and is meant for development.

No model trained on anyone's fingerprints is published here. The shipped
`models/public/fpnet-distill-kd.onnx` never saw a real person's frame: it
learnt from public datasets and synthetics, distilled from a privately
trained teacher whose targets were computed on public images only
([models/README.md](models/README.md)).

## Building

Userspace (stable Rust ≥ 1.90):

```sh
cargo build --release            # FT9368 + FT9201
cargo build --release --no-default-features   # without USB support
cargo test --workspace
```

Kernel module (SPI sensor only). Out-of-tree Rust modules must be built with
exactly the `rustc` that built the running kernel; `tools/kernel-toolchain`
fetches the matching Arch Linux `rust` package and `bindgen` into
`~/.cache/focalprint/toolchain`:

```sh
make -C kernel
```

The USB sensor needs no module.

## Running (beta)

```sh
sudo tools/dev-switch daemon models/public/fpnet-distill-kd.onnx   # SPI: load module, start focalprintd
sudo tools/dev-switch c                            # back to the previous stack
focalprint --backend usb capture                   # FT9201 terminal preview
```

The daemon refuses to start without its TPM-sealed key, which systemd hands
over through `LoadCredentialEncrypted=`. Create it once with
`sudo focalprint setup --model models/public/fpnet-distill-kd.onnx`.
`dev-switch` targets the MEGABOOK T1; for the FT9201 follow
[docs/fedora-ft9201.md](docs/fedora-ft9201.md) (the steps apply to other
distributions too). Autostart: `tools/focalprint-beta.service.in`.

## Matcher

The sensor window is about 4.5 × 6 mm, so a touch holds only a handful of
minutiae and classical matchers fail. FpNet is a small CNN that produces
dense descriptors; two touches are aligned by RANSAC on mutual descriptor
matches and scored by descriptor agreement over the shared skin.

The daemon needs an ONNX model; `fpnet-distill-kd.onnx` from the
[release](https://github.com/anatoliiii/focalprint/releases/tag/v0.1.0) is
ready to use (put it into `models/public/`; start with `--threshold 0.41`, see
[models/README.md](models/README.md)). Train your own with
`research/train.py` on public datasets; `research/distill.py` transfers a
privately trained model into one that never saw private frames, and
`research/export_onnx.py` exports a checkpoint for the daemon. Evaluation
protocol and results: `research/eval2.py`.

**Adaptive template.** A confident match (score ≥ `--adapt-score`, 0.5)
that is not a near-copy of a stored touch joins the finger's template, up
to 20 touches beyond the enrolment ones, replacing the most similar
adaptive touch when full. Enrolment touches are never replaced, so a lucky
impostor touch cannot take a template over. Templates thus grow to cover
the whole pad, which removes most rejections caused by touching with skin
the enrolment missed.

## Licence

| Part | Licence |
|---|---|
| Userspace: `crates/`, `research/`, `tools/`, `packaging/` | [LGPL-2.1-or-later](LICENSES/LGPL-2.1-or-later.txt) (`LICENSE`) |
| Kernel module: `kernel/` | [GPL-2.0-only](LICENSES/GPL-2.0-only.txt), like Linux itself |
| Model weights (release assets) | [CC-BY-4.0](LICENSES/CC-BY-4.0.txt) |

The FT9368 protocol knowledge builds on the open-source
[libfprint-fte3600](https://github.com/SamSeven777/libfprint-fte3600)
driver (LGPL-2.1-or-later); no code was copied, see
[docs/clean-room.md](docs/clean-room.md).
Public fingerprint datasets used for training are not redistributed and keep
their own terms; sources and terms are listed in
[models/README.md](models/README.md#training-data).
