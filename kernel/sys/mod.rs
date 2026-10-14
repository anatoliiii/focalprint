// SPDX-License-Identifier: GPL-2.0-only
//! Raw bindings to kernel C APIs that the `kernel` crate does not wrap yet.
//!
//! Generated at build time from this kernel's headers (see `Kbuild`), so
//! struct layouts always match the running kernel.

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    dead_code,
    unsafe_op_in_unsafe_fn,
    clippy::all,
    clippy::pedantic,
    clippy::undocumented_unsafe_blocks,
    unreachable_pub,
    missing_docs,
    improper_ctypes
)]

use kernel::ffi;

// Blocklisted like in the kernel's own bindings: these are pointer-sized.
type __kernel_size_t = usize;
type __kernel_ssize_t = isize;
type __kernel_ptrdiff_t = isize;

include!("generated.rs");
