# SPDX-License-Identifier: GPL-2.0-only
# Out-of-tree build of the focaltech_spi_fp Rust module with the kernel's own rustc.
KDIR      ?= /usr/lib/modules/$(shell uname -r)/build
TOOLCHAIN_SCRIPT ?= $(CURDIR)/../tools/kernel-toolchain
TOOLCHAIN ?= $(shell $(TOOLCHAIN_SCRIPT) path $(KDIR))

KMAKE = $(MAKE) -C $(KDIR) M=$(CURDIR) \
	RUSTC="env LD_LIBRARY_PATH=$(TOOLCHAIN)/usr/lib $(TOOLCHAIN)/usr/bin/rustc" \
	BINDGEN="$(TOOLCHAIN)/usr/bin/bindgen"

modules clean:
	$(KMAKE) $@

.PHONY: modules clean
