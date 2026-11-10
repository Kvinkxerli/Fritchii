// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>
#include <boost/container/deque.hpp>

#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "video_core/renderer_vulkan/vk_host_pacer.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Core::Frontend {
class EmuWindow;
} // namespace Core::Frontend

namespace Vulkan {

class Device;
class Scheduler;
class Swapchain;

struct Frame {
    u32 width;
    u32 height;
    u32 index;
    vk::Image image;
    vk::ImageView image_view;
    vk::ImageView storage_view;
    vk::Framebuffer framebuffer;
    vk::CommandBuffer cmdbuf;
    vk::Semaphore render_ready;
    vk::Fence present_done;
    bool storage_capable{};
    u64 render_tick{};                               ///< Scheduler tick that renders the frame
    std::chrono::steady_clock::time_point queued_at; ///< When it reached the present queue
};

class PresentManager {
public:
    PresentManager(const vk::Instance& instance,
                   Core::Frontend::EmuWindow& render_window,
                   const Device& device,
                   MemoryAllocator& memory_allocator,
                   Scheduler& scheduler,
                   Swapchain& swapchain,
                   vk::SurfaceKHR& surface);
    ~PresentManager();

    /// Returns the last used presentation frame
    Frame* GetRenderFrame();

    /// Pushes a frame for presentation
    void Present(Frame* frame);

    /// Recreates the present frame to match the provided parameters
    void RecreateFrame(Frame* frame, u32 width, u32 height, VkFormat image_view_format,
                       VkRenderPass rd, bool storage);

    [[nodiscard]] bool NeedsStorage(const Frame* frame, bool required) const;

    /// Waits for the present thread to finish presenting all queued frames.
    void WaitPresent();

    /// How many additional frames can be queued without stalling the render thread
    [[nodiscard]] size_t MaxExtraFrames() const;

    [[nodiscard]] std::size_t SwapchainImageCount() const { return swapchain_image_count; }
    [[nodiscard]] VkFormat SwapchainImageFormat() const { return swapchain_image_format; }

private:
    void PresentThread(std::stop_token token);

    /// Host frame pacer (vk_host_pacer.h): collects when earlier frames were shown and holds
    /// `frame` until its target. Returns whether it was held.
    bool PaceFrame(Frame* frame, std::stop_token token, double* out_fence_wait_ms = nullptr,
                   double* out_present_sleep_ms = nullptr);

    /// Waits for each present id until it is shown (VK_KHR_present_wait).
    void PresentWaitThread(std::stop_token token);

    void ReportShown(u64 id, std::chrono::steady_clock::time_point shown, bool lost);

    void DrainShown();

    void CopyToSwapchain(Frame* frame);

    void CopyToSwapchainImpl(Frame* frame);

    void RecreateSwapchain(Frame* frame);

    void DiscardFrame(Frame* frame);

    void SetImageCount();

private:
    const vk::Instance& instance;
    Core::Frontend::EmuWindow& render_window;
    const Device& device;
    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;
    Swapchain& swapchain;
    vk::SurfaceKHR& surface;
    vk::CommandPool cmdpool;
    std::vector<Frame> frames;
    boost::container::deque<Frame*> present_queue;
    boost::container::deque<Frame*> free_queue;
    std::condition_variable_any frame_cv;
    std::condition_variable free_cv;
    std::mutex swapchain_mutex;
    std::mutex queue_mutex;
    std::mutex free_mutex;
    std::jthread present_thread;
    bool blit_supported;
    bool storage_supported;
    bool use_present_thread;
    std::atomic<std::size_t> image_count{};
    std::atomic<std::size_t> swapchain_image_count{};
    std::atomic<VkFormat> swapchain_image_format{};

    // Host frame pacer, used when the device supports VK_KHR_present_wait (present thread only).
    struct PendingWait {
        u64 id;
        VkSwapchainKHR swapchain;
        std::chrono::steady_clock::time_point submitted;
    };
    struct ShownReport {
        u64 id;
        std::chrono::steady_clock::time_point shown;
        bool lost;
    };
    bool present_wait_enabled{};
    HostPacer pacer;
    u64 last_present_id{};
    std::mutex present_wait_mutex; ///< Held while waiting on a swapchain; recreation takes it
    std::mutex wait_queue_mutex;
    std::condition_variable_any wait_queue_cv;
    std::deque<PendingWait> wait_queue;
    std::mutex shown_mutex;
    std::condition_variable shown_cv;
    std::vector<ShownReport> shown_reports;
    std::jthread present_wait_thread;
};

} // namespace Vulkan
