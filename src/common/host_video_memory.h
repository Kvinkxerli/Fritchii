// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Common {

/// The dedicated video memory of the host's largest hardware GPU in bytes, 0 when unknown (not
/// Windows, or DXGI failed). Usable before any renderer exists; read once and cached.
[[nodiscard]] u64 HostDedicatedVideoMemory();

/// Whether the host's GPU is a small card (under 5 GiB of dedicated memory, a GTX 1650): Switch 2
/// titles then get the defaults that keep them inside its memory. False when unknown.
[[nodiscard]] bool HostHasSmallVideoMemory();

} // namespace Common
