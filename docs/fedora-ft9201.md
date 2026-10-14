# Fedora with the FT9201 USB sensor (`2808:c652`)

For laptops with the USB FocalTech FT9201, e.g. TECNO MegaBook K16S. No
kernel module is needed. The USB protocol was reverse-engineered from
captures and is unit-tested, but **has not yet run on real hardware**, so
check the sensor first (step 2) before handing it to fprintd.

Keep a password login: everything below can be undone with one command
(see [Rolling back](#rolling-back)).

```sh
lsusb -d 2808:c652        # must list the sensor
systemd-creds has-tpm2    # the key protecting templates is sealed to the TPM
```

## 1. Build

Needs Rust ≥ 1.90 (`rustc --version`; if Fedora's is older, use rustup).

```sh
sudo dnf install cargo rust gcc git fprintd fprintd-pam
git clone <this repository> focalprint && cd focalprint
cargo build --release
```

## 2. Check the sensor

fprintd must not hold the sensor while testing:

```sh
sudo systemctl stop fprintd
sudo ./target/release/focalprint --backend usb info
sudo ./target/release/focalprint --backend usb capture
```

`info` should report `FT9201 (USB), id 0x6293`. `capture` previews each touch
in the terminal. A good frame looks like the FT9368's: **dark ridges, light
valleys, light where there is no finger.** Two things are still unconfirmed
on this sensor and can be switched without rebuilding:

| What you see | Try |
|---|---|
| noise, no ridge pattern | `FOCALPRINT_FT9201_PIXELS=be` (16-bit byte order) |
| ridges light, background dark | `FOCALPRINT_FT9201_PIXELS=invert` |
| both | `FOCALPRINT_FT9201_PIXELS=be,invert` |

```sh
sudo env FOCALPRINT_FT9201_PIXELS=be ./target/release/focalprint --backend usb capture
```

Then test the matcher with the shipped model. Fingers are enrolled in
memory only; nothing is stored:

```sh
sudo ./target/release/focalprint --backend usb neural \
    --model models/public/fpnet-distill-kd.onnx --fingers 2 --threshold 0.41
```

Each verification line shows the scores of both fingers, e.g.
`MATCH A [A=0.701 B=0.243]`. The model was trained for the FT9368 and has not
been measured on this sensor: if genuine touches mostly score well above the
other finger's score, it works; choose a threshold clearly above the highest
other-finger score you see (more in [models/README.md](../models/README.md)).

## 3. Install

From the repository root, after `cargo build --release`:

```sh
sudo env PREFIX=/usr/local SYSTEMD_DIR=/etc/systemd/system \
    packaging/install-tree.sh / /usr/local/libexec/focalprint 0.1.0
sudo restorecon -R /usr/local/bin/focalprint /usr/local/libexec/focalprint
sudo /usr/local/libexec/focalprint/post-install 0.1.0 no
```

This installs `focalprint`, the daemon `/usr/local/libexec/focalprint/focalprintd`,
and a drop-in `/etc/systemd/system/fprintd.service.d/focalprint.conf` that
makes `fprintd.service` run `focalprintd` instead of the stock daemon.
`pam_fprintd`, `fprintd-*` and GNOME/KDE settings talk to it unchanged.

Create the TPM-sealed key and install the model:

```sh
sudo focalprint setup --model models/public/fpnet-distill-kd.onnx
```

Set the threshold from step 2 (and the pixel setting, if you needed one):

```sh
sudo systemctl edit fprintd
```

```ini
[Service]
ExecStart=
ExecStart=/usr/local/libexec/focalprint/focalprintd --threshold 0.41
# only if step 2 needed it:
Environment=FOCALPRINT_FT9201_PIXELS=be
```

```sh
sudo systemctl restart fprintd
journalctl -u fprintd -b     # "sensor ready sensor=FT9201 (USB) …"
fprintd-enroll               # 20 touches: centre, edges, tip, turned
fprintd-verify
```

For `sudo`, login and the lock screen:

```sh
sudo authselect enable-feature with-fingerprint
```

## Adaptive template

Every confident match (score ≥ 0.5, `--adapt-score`) that shows skin not yet
in the template is added to it, up to 20 touches per finger beyond the 20
from enrolment. Rejections caused by touching with a part of the finger
the enrolment missed fade out over the first days of use. The journal shows
`template adapted … touches=N`.

## Reporting

`journalctl -u fprintd` holds scores and timings only, never images, and is
safe to share. Frames are biometric data: do not send them. If the protocol
misbehaves, `tools/ft9201-grab.py` (needs `python3-pyusb`) prints a per-frame
summary that is safe to share.

## SELinux

The daemon runs as `unconfined_service_t` (a `bin_t` binary started by
systemd), so SELinux should not get in the way. If it does:
`sudo ausearch -m avc -ts recent`.

## Rolling back

```sh
sudo systemctl revert fprintd     # drops both drop-ins: stock fprintd again
sudo systemctl restart fprintd
```

Templates and the key stay in `/var/lib/focalprint` until you delete it.
