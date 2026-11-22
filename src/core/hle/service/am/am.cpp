// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/settings.h"
#include "core/hle/service/am/am.h"
#include "core/hle/service/am/button_poller.h"
#include "core/hle/service/am/event_observer.h"
#include "core/hle/service/am/service/all_system_applet_proxies_service.h"
#include "core/hle/service/am/service/application_proxy_service.h"
#include "core/hle/service/am/window_system.h"
#include "core/hle/service/server_manager.h"

namespace Service::AM {

void LoopProcess(Core::System& system) {
    WindowSystem window_system(system);
    ButtonPoller button_poller(system, window_system);
    EventObserver event_observer(system, window_system);

    auto server_manager = std::make_unique<ServerManager>(system);

    server_manager->RegisterNamedService(
        "appletAE", std::make_shared<IAllSystemAppletProxiesService>(system, window_system));
    server_manager->RegisterNamedService(
        "appletOE", std::make_shared<IApplicationProxyService>(system, window_system));
    // Serve Switch 2 graphics (nx2:gsv, vi2, vdecsvc, ...) on extra threads: vi2's frame
    // presents block until the frame's fence and vblank (~14 ms at 60 fps), so other requests
    // queued behind them stall without extra threads.
    if (Settings::CurrentProgramIsSwitch2()) {
        server_manager->StartAdditionalHostThreads("AM:Host", 2);
    }
    ServerManager::RunServer(std::move(server_manager));
}

} // namespace Service::AM
