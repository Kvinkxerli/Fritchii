// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// The numbered experiments' defaults (VideoCore::Nx2Debug::LoadExperiments), kept out of
// nx2_debug_control.h so a default flip rebuilds this one file. Compiled into common: common
// (thread.cpp) and tools that link only common read Experiment() too. What each experiment
// does: docs/fritchii/switches.md.

#include <atomic>
#include <cstdlib>
#include <string>

#include "common/host_video_memory.h"
#include "video_core/nx2_debug_control.h"

namespace VideoCore::Nx2Debug {

void LoadExperiments() {
    static const bool loaded = [] {
        // Kept experiments: on unless their variable says otherwise. EXP8 = 15: Ocarina of Time
        // walking through Hyrule Field, fresh launches: 1% low 30.3 -> 35.1 fps, frames over
        // 25 ms 68 -> 38 of ~1650.
        g_experiments[8].store(15, std::memory_order_relaxed);
        // EXP9 = 1: native uniform bank pages compared only when they may have changed (verified
        // with EXP9=2: 0 missed in 120 s walking). Idle 54.5 -> 57.7 fps, walking 56.6 -> 58.9.
        // EXP9 = 3 (2026-11-08): the same without the once-a-frame compare of every page, which
        // never found a change ("frame check N / 0", "missed without the frame check 0"). Castle
        // Town, one session, 4 x 35 s each: uniform buffers 135.6 -> 97.9 ms/s on the split's back,
        // 40.8 -> 42.3 fps; flash watch on the Castle Town route 267 reports vs 273 with EXP9=1.
        g_experiments[9].store(3, std::memory_order_relaxed);
        // EXP10 = 1: macro DMA reads wait only for the dispatches, uploads, copies and releases
        // that may have written them (EDEN_NX2_DMA_SNAPSHOT=10 verify: 0 differed in 2 x 90 s
        // walking). GPU front idle 81 -> 105 ms/s, waits for the back 368 -> 338 ms/s.
        g_experiments[10].store(1, std::memory_order_relaxed);
        // EXP14: unchanged macro register writes dropped before FrontWrite's other checks.
        // Castle Town, profile toggle: front macro 250 -> 236 ms/s, push 77 -> 70 ms/s.
        g_experiments[14].store(1, std::memory_order_relaxed);
        // EXP18 = 6 (exact): a buffer touches its LRU node once
        // per frame; a second touch in the same frame was already a no-op in LruCache::Touch
        // (its tick is not older), so only the node's cache miss goes. Bit 2: storage bindings
        // reuse their last FindBuffer and range result while binding_epoch and the map serial
        // are unchanged (2026-11-19, AMD one follower with ESPLIT at the fountain: queued entries
        // 2.4 -> 2.05 us, draw tails 3.42 -> 3.37 us, picture unchanged). Bit 1 stays off.
        // EDEN_NX2_EXP18=4 is the old default.
        g_experiments[18].store(6, std::memory_order_relaxed);
        // EXP21 = 1: a texture whose handle word is a storage buffer's address gets no texture
        // (VideoCommon::HandleIsStoragePointer). Ocarina of Time's Zora's River waterfall
        // flickered every other frame in bursts without it.
        g_experiments[21].store(1, std::memory_order_relaxed);
        // EXP29 = 6: the exact parts of the ported micro bundle (TickWork's lock-free check, native
        // compute inline uploads not recorded); 1 (memory usage reuse) changes what the caches'
        // GC sees and stays off.
        g_experiments[29].store(6, std::memory_order_relaxed);
        // EXP34 = 1: ACQUIRE_PRECHECK=2 no longer answers met over a write to the word still
        // queued for the back (a reset there let a channel run before its producer). A strict
        // tightening: only met answers become the normal path.
        g_experiments[34].store(1, std::memory_order_relaxed);
        // EXP35 = 1: guest writes of never-GPU-written cached pages collected for the back instead
        // of OnCPUWrite under both caches' mutexes (see the EXP35 notes in docs/fritchii/switches.md).
        // EXP35 = 2 (2026-11-08): every guest write collected (guest-core write lock spinning 0.7% ->
        // 0, standing 45.3 -> 46.5 fps, route flash reports 253 = camera noise).
        g_experiments[35].store(2, std::memory_order_relaxed);
        // EXP42 = 1 and EXP43 = 3 (2026-11-08, moving-NPC dips): pooled VkImages for deleted
        // render targets, and drifted per-frame targets relocated in place (the newest of several
        // predecessors). Castle Town live toggles: EXP43 0/3 46.3 -> 49.3 fps, EXP42 0/1 on top
        // 44.7 -> 47.9 (each won all 4 pairs); 21 loads 0 washed out; owner playtest: no visual
        // bugs, rarely under 40 fps.
        g_experiments[42].store(1, std::memory_order_relaxed);
        g_experiments[43].store(3, std::memory_order_relaxed);
        // EXP56 = 1 (2026-11-09): a cleared depth target larger than its colour targets is not
        // uploaded from guest memory first (TextureCache::JoinImages, overwritten_by_clear).
        // Castle Town live toggle 41.3 -> 43.8 fps, clear zone 54.7 -> 36.3 ms/s; screenshots
        // off/on identical.
        g_experiments[56].store(1, std::memory_order_relaxed);
        // EXP58 = 1 (2026-11-10): guarded render targets that a newer target of another shape
        // superseded are deleted once 60 frames old (TextureCache::SweepColdImages). Castle Town
        // 150 s A/B: device memory 7.8-10.5 -> 5.2-5.7 GB, collector evictions 83-1915 a window
        // -> 0, worst frames 140-180 -> 30-48 ms, moving median 42.2 -> 46.9 fps; skeleton fight
        // 5.1-5.5 GB instead of 10.9 with 0 evictions; two owner playtests without glitches.
        g_experiments[58].store(1, std::memory_order_relaxed);
        // EXP61 = 1 (2026-11-10): FlushCommands submits nothing when nothing was recorded since
        // the last submit. Castle Town live toggle: empty submits 4,600 -> 145 a second, 43.0 ->
        // 45.6 fps (3 of 4 pairs, the 4th a tie), GPU thread split drain 432 -> 396 ms/s.
        g_experiments[61].store(1, std::memory_order_relaxed);
        // EXP64 = 16 (2026-11-11, NPC-pass dips): a cleared target retires its slot's old
        // same-format images untouched for 16 frames. Night market live toggle 0/16: peak texture
        // cache memory 5,972 -> 5,482 MiB, binds over 1 ms 30 -> 6 a 2 min (over 10 ms 1 -> 0),
        // 53.4 -> 54.6 fps; 0 retired images wanted again within 16 frames (at 1: 29, at 8: 2).
        g_experiments[64].store(16, std::memory_order_relaxed);
        // EXP68 = 1 (2026-11-11, NPC-pass dips): a depth clear with a scissor of the real width
        // goes to the exact image (one per NPC a frame, not two plus an upload and alias copies;
        // fable52_answer_npcdips). Night market toggle: images made 483 -> 379, texture upload
        // 18.8 -> 14.1 ms/s, about +1 fps; owner playtest: no visual glitches, 50-55 fps except
        // when a dog enters from behind the camera. Bit 4 (no pool) stays off: binds over 1 ms x3.
        g_experiments[68].store(1, std::memory_order_relaxed);
        // EXP70 = 2 (2026-11-11, display stalls): with no free render frame for 2 ms the renderer
        // drops a queued frame instead of stalling emulation. Night market with a simulated 150 ms
        // display stall every 10 s (EXP71): off, every stall froze emulation 65-82 ms; on, one
        // 50 ms hitch in a minute (21 frames dropped). No drops in a minute of normal play.
        g_experiments[70].store(2, std::memory_order_relaxed);
        // EXP69 = 600 (2026-11-11): Preflushable marks reset every 600 frames. Night market
        // toggle: buffer downloads 81 -> 53 MB/s, back fence 45.9 -> 38.1 ms/s, +0.6 fps. Owner
        // playtest with EXP70 (market + Stalchild fight): constant 60 outside the boss wave
        // (40-45 there), no flashes or glitches.
        g_experiments[69].store(600, std::memory_order_relaxed);
        // EXP69 = 4 and EXP81 = 4 (2026-11-12, fable52_answer_fences2 (a); EXP81 now fed by a
        // lock-free read bitmap, so it works with EXP86): pages nobody read for 4 frames stop being
        // downloaded at fences. Owner skeleton fight A/B: downloads 412-719 -> 162-203 MB per 600
        // frames, unread landed pages 77-83% -> 27-30%, fence thread buffer mutex held 383 -> 83
        // ms / 5 s, worst hitch 788 -> 409 ms, frames over 40 ms 4.5 -> 4.1 per 1000; owner: not
        // less smooth.
        g_experiments[69].store(4, std::memory_order_relaxed);
        // EXP81 16, not 4 (fable52_answer_fight3 5): with 4 the game's read-backs doubled (50 -> 116
        // a second, guest waiting 64 -> 117 ms/s); a longer memory keeps pages it reads every few frames.
        g_experiments[81].store(16, std::memory_order_relaxed);
        // EXP72 = 1 (2026-11-11): a preflushed read whose bytes are all in a submitted fence download
        // waits for it. Night market toggle: reader waits 683 -> 479, wait time 515 -> 299 ms. Owner
        // Stalchild fight playtest: readers waited for the host GPU ~280-350 -> 40-70 ms/s, 0
        // timeouts, no flicker or glitches (the boss wave still dips: front waits, not reads).
        g_experiments[72].store(1, std::memory_order_relaxed);
        // EXP73 = 1 (2026-11-12): no report flush round trip after a drained macro read with no
        // report listed over the bytes. Owner fight toggle with EXP30: macro dma flush 34.5 -> 20.4
        // ms/s, front waits 442 -> 416 ms/s, no flashes; it only drops empty work.
        g_experiments[73].store(1, std::memory_order_relaxed);
        // EXP24 = 4 (2026-11-12): OoT's present poll sleeps at most 250 us and ends when the
        // event's value word reads 1. The guest timeline put that poll on the night market's
        // critical path (1.9 ms a frame, 5.2 at p90); 3 alone gave 57.8 -> 58.4 fps. Only the
        // verified site. With EXP75 = 600 (precise guest timers, +0.9 fps at the night market):
        // together they made the GPU side the frame's critical path, so its cuts now show
        // (OOT_MACROS +2.1 with them on). Owner playtest of all three: dogs no longer hitch,
        // the chicken much better, no glitches; owner asked for them as defaults.
        g_experiments[24].store(4, std::memory_order_relaxed);
        g_experiments[75].store(600, std::memory_order_relaxed);
        // EXP77 = 1 and EXP78 = 1 (2026-11-12): kill frames re-made GPU-drawn targets the cold
        // sweep had just taken and filled them from memory nobody wrote. Owner Stalchild fight with
        // both on vs off: frames over 22 ms 2.54 -> 2.06 a second, over 28 ms 0.39 -> 0.30, up to
        // 500 uploads (1.5 GB) skipped per 600 frames, device memory steady ~5.5 GB, no glitches
        // seen; owner asked for them as defaults.
        g_experiments[77].store(1, std::memory_order_relaxed);
        g_experiments[78].store(1, std::memory_order_relaxed);
        // EXP82 = 1800 and EXP66 = 1024 (2026-11-12, overnight): proven image pool entries stay
        // 1,800 frames and the pool may hold 1 GiB. Daytime market toggle (orbit, uncapped, zones
        // on): clear 59.6 -> 43.0 ms/s, back busy 7.88 -> 7.45 us per draw, 0.1% low 30.8 ->
        // 35.4; pool 1,000-1,500 taken per 600 frames, held up to ~716 MiB. The pool still
        // empties on device memory pressure (TickFrame).
        g_experiments[82].store(1800, std::memory_order_relaxed);
        g_experiments[66].store(1024, std::memory_order_relaxed);
        // EXP86 = 1 (2026-11-12, overnight): guest reads of GPU-written pages answered from the
        // lock-free page mirrors. Verify mode (2) at the daytime market: ~17,000 reads per 5 s
        // checked, 0 mismatches. Toggle (orbit, uncapped, 180 s warm, 6 cycles): guest lock waits
        // 65.0 -> 3.7 ms/s median, 65.0 -> 66.5 fps, 1% low 44.0 -> 44.9, 0 broken mirrors.
        g_experiments[86].store(1, std::memory_order_relaxed);
        // EXP88 = 3 (2026-11-12, overnight): ported parts a-c (Fable's plan). Verify
        // runs: part c skipped 100% of ~800,000 flush ranges a second with 0 that would change,
        // parts a+b 0 differ over ~8 million direct writes. Daytime market toggle (orbit,
        // uncapped): back 7.89 -> 7.43 us per draw, flush work 33.6 -> 16.9 ms/s, 59.8 -> 61.8
        // fps in the same session, 0.1% low 29.2 -> 32.8; picture normal.
        g_experiments[88].store(3, std::memory_order_relaxed);
        // EXP85 = 3 (2026-11-12, overnight): inline uploads and WriteMemory skip the buffer lock on
        // never-GPU-written pages. Alone (EXP86 on) it was inside the noise; with EXP88 = 3 at the
        // daytime market (orbit, uncapped): back 8.09 -> 7.69 us per draw, method 175.4 -> 163.2
        // ms/s, 58.7 -> 60.8 fps same session, 1% low 38.8 -> 40.4; picture normal.
        g_experiments[85].store(3, std::memory_order_relaxed);
        // EXP79 = 1 (2026-11-12, overnight): guest cores wait for work in
        // hardware. Judged earlier in runs that mixed the market's two phases (no gain); inside the
        // heavy phase only (orbit, uncapped, 8 x 25 s): 56.5 -> 58.6 fps, 1% low 39.2 -> 40.4,
        // back 8.04 -> 7.81 us per draw. Timing only, no rendering change.
        g_experiments[79].store(1, std::memory_order_relaxed);
        // EXP90 = 1 (2026-11-12, overnight; Fable after88 2): an image's LRU node is touched once a
        // frame. Exact (the node ignored same-tick touches anyway). Heavy phase toggle: 92% of
        // touches skipped, back 8.30 -> 8.02 us per draw, 56.0 -> 57.1 fps, 1% low 38.1 -> 39.6.
        g_experiments[90].store(1, std::memory_order_relaxed);
        // EXP99 = 1 (2026-11-12; fable52_answer_chicken 1): evicted image pool entries (OoT's
        // per-NPC depth targets, the chicken hitch) are destroyed in the back's idle wait instead
        // of at the frame's end. Heavy phase toggle: other 143.4 -> 114.2 ms/s, back 7.92 -> 7.72
        // us per draw, 0.1% low 15.6 -> 23.0; every eviction went while idle ("here" 0).
        g_experiments[99].store(1, std::memory_order_relaxed);
        // EXP112 = 1 (2026-11-13): a guest read of GPU-written memory takes the whole GPU-written
        // run around it (up to 256 KiB) in the same read-back. Great fairy fountain, flipped every
        // 20 s: 52.3 -> 71.0 fps, 1% / 0.1% low 40.0 / 34.2 -> 46.0 / 38.8; read-backs that went
        // to the back 16.6 -> 3.1 a frame, the game's waiting 7.5 ms a frame -> under 2.
        g_experiments[112].store(1, std::memory_order_relaxed);
        // EXP105 = 64 (2026-11-13): the command list GPU-written check (read-only, logs only, never
        // fired in OoT) runs on 1 entry in 64. At every entry it cost ~8 ms/s of the back and was
        // 236,000 of the memory stage's 273,000 level entries a second at the market (5b shadow).
        g_experiments[105].store(64, std::memory_order_relaxed);
        // EXP121 = 1 (2026-11-13, owner OK): the texture collector records the download of a
        // GPU-written image it evicts and deletes the image once the GPU is past the copy, instead
        // of waiting for the GPU each time (2-3 ms stalls in fight frames). Verified with EXP121=2:
        // 481 landings compared, 0 mismatches. Fight 62.5 -> 64.2 and 62.9 -> 64.7 fps.
        g_experiments[121].store(1, std::memory_order_relaxed);
        // EXP120 = 2 (2026-11-13, owner OK): register writes, linear uploads and blit launches are
        // noted for the macro DMA snapshots by the bytes they really write, not as unranged
        // writers. Market: 1.89 M reads skipped their drain, 0 differed; 1% low 37.7 -> 39.2.
        g_experiments[120].store(2, std::memory_order_relaxed);
        // EXP122 = 1 (2026-11-13): the writer page watch (2,048 pages) answers snapshot checks
        // without the ring scan; verified at the pot room (4.39 M answers, 0 mismatches) and the
        // "past the scan" drains stop while it is on.
        g_experiments[122].store(1, std::memory_order_relaxed);
        // EXP124 = 1 (2026-11-13): a compute macro's inline upload words go as one entry per batch
        // and one upload call per safe 4 KiB piece. Market 0/1 (4 x 30 s): 56.2 -> 58.4 fps, 1% low
        // 38.8 -> 40.0, the back's method zone -11 ms/s; EXP124=2 check: no piece disagreed.
        g_experiments[124].store(1, std::memory_order_relaxed);
        // EXP126 = 1 (2026-11-14, owner OK): the ranged write ring's overflow no longer makes QMD
        // reads wait for the whole backlog (ranged writes are stamped, by GPU and device address).
        // Market trial (=2): ~560 reads compared a window, 0 differed; 0/1: 57.7 -> 58.7 fps, 0.1%
        // low 30.1 -> 33.9. The front's wait moves to "fence writers" (next target).
        g_experiments[126].store(1, std::memory_order_relaxed);
        // EXP114 = 2 (2026-11-13): the GPU thread's pending polls wait 2 us with PAUSE instead of
        // yielding. Daytime market, flipped every 20 s: 59.2 -> 59.9 fps, windows 54.8-63.7 ->
        // 59.1-60.4, 1% low 40.0 -> 42.1, back 7.35 -> 7.16 us a draw; the owner felt it steadier.
        g_experiments[114].store(2, std::memory_order_relaxed);
        // EXP13 = 4 + EXP12 = 2 (2026-11-13): the front predicts QMD releases and, for acquires
        // nothing queued can meet, suspends without a full drain. Skeleton fight, flipped every
        // 20 s: 64.8 -> 66.3 fps, 0.1% low 36.4 -> 42.1, the front's split drain 497 -> 297 ms/s;
        // EXP12=3 verify: met there = met after the drain; no prediction error. Owner's kill
        // playtest with both on: never under 48 fps, picture clean.
        g_experiments[13].store(4, std::memory_order_relaxed);
        // EXP12 = 4 (2026-11-14, owner's call after watching the market toggle): the fence path
        // suspends without waiting for every queued compute/copy writer first (exact: the waiter
        // latch and the resume path decide, as after mode 2's wait). With EDEN_NX2_GPU_WAKE=1
        // (built in with it): 59.3 -> 59.8 fps, 0.1% low 31.7 -> 37.0, the owner rarely saw <50.
        g_experiments[12].store(4, std::memory_order_relaxed);
        // EXP52 = 1 + EXP55 = 1 (2026-11-14, autopilot; owner: "v6 gets every exact win"): the
        // pipeline key is patched instead of fully refreshed when only the programs or only the
        // vertex input moved. EXP52=2/EXP55=2 verify at the market: 594,768 keys compared in 10 s,
        // 0 differ. Toggle 57.6 -> 58.2 fps, 1% low 38.8 -> 40.0; part of the combined keepers
        // toggle (62.7 -> 64.1 fps, 1% 44.4 -> 45.5).
        g_experiments[52].store(1, std::memory_order_relaxed);
        g_experiments[55].store(1, std::memory_order_relaxed);
        // EXP49 = 1 (2026-11-10, owner's call for v5.7): the async read-back. Stalchild fight,
        // flipped every 30 s while the owner fought: 51.9 -> 56.9 fps, the back's read flushes
        // 35 -> 6 ms/s, guest lock waits halved. No beams in that session or two later ones
        // played with it on from boot.
        g_experiments[49].store(1, std::memory_order_relaxed);
        // EXP84 = 1 + EXP100 = 1: lock-free guest read checks using image_count_filter and
        // gpu_image_filter (+ image_seq) instead of blocking guest cores on both caches' mutexes.
        g_experiments[83].store(1, std::memory_order_relaxed);
        g_experiments[84].store(1, std::memory_order_relaxed);
        g_experiments[100].store(1, std::memory_order_relaxed);
        // EXP39 = 3: publish to a back that waits after 8 words, no snapshot refresh of entries
        // that ran (see the EXP39 notes in docs/fritchii/switches.md). Timing only.
        g_experiments[39].store(3, std::memory_order_relaxed);
        // EXP27 = 5 (small cards): the front publishes before it waits and the back prefetches the ring.
        if (Common::HostHasSmallVideoMemory()) {
            g_experiments[27].store(5, std::memory_order_relaxed);
        }
        for (int n = 0; n < ExperimentCount; ++n) {
            const std::string name = "EDEN_NX2_EXP" + std::to_string(n);
            if (const char* value = std::getenv(name.c_str())) {
                g_experiments[n].store(std::atoi(value), std::memory_order_relaxed);
            }
        }
        g_experiments_loaded.store(true, std::memory_order_release);
        BumpSwitchGeneration();
        return true;
    }();
    static_cast<void>(loaded);
}

} // namespace VideoCore::Nx2Debug
