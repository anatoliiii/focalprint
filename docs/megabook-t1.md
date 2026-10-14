# TECNO MEGABOOK T1, early revisions (SPI sensor FTE4800)

Early MEGABOOK T1 units carry a FocalTech FT9368 behind the ACPI device
`FTE4800` on the SPI bus. Check yours:

```sh
ls /sys/bus/acpi/devices/FTE4800:00     # exists → this guide applies
lsusb -d 2808:                          # a USB FocalTech sensor instead → no module needed
```

Unlike the USB sensors, this one needs a kernel module, `focaltech_spi_fp`.
It is written in Rust, and that comes with a hard requirement.

## Requirement: a Rust-enabled kernel and its exact rustc

An out-of-tree Rust module can only be built:

1. against a kernel built with **`CONFIG_RUST=y`**, and
2. with **exactly the `rustc` (and `bindgen`) that built that kernel**: the
   same version *and the same build*. A rustup toolchain of the same version
   number does not work; the module would fail to link against the kernel's
   Rust metadata.

Check your kernel:

```sh
grep -E 'CONFIG_RUST=|RUSTC_VERSION_TEXT|BINDGEN_VERSION_TEXT' \
    /usr/lib/modules/$(uname -r)/build/.config
```

| Distribution | Status |
|---|---|
| Arch Linux (`linux`) | works: built with `CONFIG_RUST=y`; the matching toolchain is installed automatically (below) |
| Other distributions | works only if the kernel has `CONFIG_RUST=y` and the distribution's `rustc` and `bindgen` packages are exactly the versions printed above |
| Kernels without `CONFIG_RUST` | the module cannot be built; DKMS skips them |

`tools/kernel-toolchain` (installed with the module sources) picks the
compiler:

- if `/usr/bin/rustc --version` and `/usr/bin/bindgen --version` print the
  very strings recorded in the kernel config, it uses them;
- on Arch, otherwise, it downloads the matching `rust` and `rust-bindgen`
  packages from the Arch Linux Archive, verifies their signatures with
  `pacman-key`, and caches them in `/var/cache/focalprint/toolchain` (about
  320 MB per toolchain version). This needs network access when a new kernel
  arrives.

## Installing

The `focalprint` package detects the sensor and builds the module with DKMS:

- **Arch** (`pacman -U`), **Fedora** (`dnf install`): automatically when the
  sensor is present and `dkms` is installed (install `dkms` and the kernel
  headers, e.g. `linux-headers`, first);
- **Debian/Ubuntu** (`apt install ./focalprint_*.deb`): asks during
  installation;
- **tarball** (`sudo ./install.sh`): asks.

Later, by hand:

```sh
sudo /usr/lib/focalprint/dkms-helper enable VERSION     # Arch
sudo /usr/libexec/focalprint/dkms-helper enable VERSION # Debian, Ubuntu, Fedora
```

Then:

```sh
sudo focalprint setup --model /path/to/fpnet.onnx
sudo systemctl restart fprintd
fprintd-enroll
```

## After a kernel update

DKMS rebuilds the module for each new kernel. If that fails, typically
because the new kernel was built by a `rustc` that cannot be found, the
sensor stops working until it is fixed; password login is unaffected. See
`dkms status` and `/var/lib/dkms/focaltech-spi-fp/*/build/make.log`.

## Rolling back

```sh
sudo /usr/lib/focalprint/dkms-helper disable    # or /usr/libexec/focalprint/…
```

Removing the `focalprint` package does the same and gives fprintd back its
stock daemon. Enrolled fingers and the key stay in `/var/lib/focalprint`
until you delete that directory.
