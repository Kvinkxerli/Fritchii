// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/host_video_memory.h"

#ifdef _WIN32
#include <windows.h>

#include <dxgi.h>
#endif

namespace Common {

u64 HostDedicatedVideoMemory() {
#ifdef _WIN32
    static const u64 bytes = []() -> u64 {
        // Loaded by hand, as the Vulkan allocator does: no link dependency on dxgi.lib.
        const HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
        if (dxgi == nullptr) {
            return 0;
        }
        using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
        const auto create = reinterpret_cast<CreateFactory>(
            reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
        IDXGIFactory1* factory{};
        if (create == nullptr ||
            FAILED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
            return 0;
        }
        u64 largest = 0;
        IDXGIAdapter1* adapter{};
        for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND;
             ++index) {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter->GetDesc1(&desc)) &&
                (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
                desc.DedicatedVideoMemory > largest) {
                largest = desc.DedicatedVideoMemory;
            }
            adapter->Release();
        }
        factory->Release();
        return largest;
    }();
    return bytes;
#else
    return 0;
#endif
}

bool HostHasSmallVideoMemory() {
    const u64 bytes = HostDedicatedVideoMemory();
    return bytes != 0 && bytes < (u64{5} << 30);
}

} // namespace Common
