#!/bin/sh
# Lays out a built focalprint tree for packaging; shared by every format.
#   install-tree.sh DESTDIR LIBEXECDIR VERSION
# Distribution packages use PREFIX=/usr (default) and the vendor systemd
# directory; the tarball sets PREFIX=/usr/local, SYSTEMD_DIR=/etc/systemd/system.
# Expects release binaries in $CARGO_TARGET_DIR/release (default target/).
set -eu
dest=$1
libexec=$2
ver=$3
prefix=${PREFIX:-/usr}
systemd_dir=${SYSTEMD_DIR:-/usr/lib/systemd/system}
datadir=$prefix/share
top=$(cd "$(dirname "$0")/.." && pwd)
common=$top/packaging/common
bin=${CARGO_TARGET_DIR:-$top/target}/release

install -Dm0755 "$bin/focalprint" "$dest$prefix/bin/focalprint"
install -Dm0755 "$bin/focalprintd" "$dest$libexec/focalprintd"
for f in detect-sensor dkms-helper post-install pre-remove; do
    sed "s|@DATADIR@|$datadir|g; s|@LIBEXECDIR@|$libexec|g" "$common/$f" |
        install -Dm0755 /dev/stdin "$dest$libexec/$f"
done
install -Dm0755 "$top/tools/migrate-from-fte4800" "$dest$libexec/migrate-from-fte4800"

sed "s|@LIBEXECDIR@|$libexec|g" "$common/fprintd-focalprint.conf.in" |
    install -Dm0644 /dev/stdin "$dest$systemd_dir/fprintd.service.d/focalprint.conf"

# Kernel module sources, registered with DKMS only on machines with the SPI
# sensor (see post-install). Not under /usr/src on purpose: some dkms
# packages build everything found there automatically.
dkms=$dest$datadir/focalprint/dkms-source
for f in Kbuild Makefile focaltech_spi_fp.rs reset.rs sensor.rs spi.rs uapi.rs \
         sys/mod.rs sys/wrapper.h sys/bindgen_parameters; do
    install -Dm0644 "$top/kernel/$f" "$dkms/$f"
done
install -Dm0755 "$top/tools/kernel-toolchain" "$dkms/kernel-toolchain"
sed "s|@VERSION@|$ver|g" "$common/dkms.conf.in" | install -Dm0644 /dev/stdin "$dkms/dkms.conf"

for f in README.md docs/clean-room.md docs/protocols.md docs/megabook-t1.md docs/fedora-ft9201.md; do
    install -Dm0644 "$top/$f" "$dest$datadir/doc/focalprint/$(basename "$f")"
done
