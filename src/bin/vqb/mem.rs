//! Counting global allocator: tracks live heap bytes and a resettable high-water
//! mark, so the runner can measure peak heap used during a single encode call.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// System allocator wrapper that records live bytes and their peak.
pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let live = CURRENT.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(live, Relaxed);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            let live = CURRENT.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(live, Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT.fetch_sub(layout.size(), Relaxed);
    }
    // `realloc` uses the default impl, which routes through `alloc`/`dealloc`.
}

/// Live (allocated, not yet freed) bytes.
pub fn current() -> usize {
    CURRENT.load(Relaxed)
}

/// High-water mark of live bytes since the last `reset_peak`.
pub fn peak() -> usize {
    PEAK.load(Relaxed)
}

/// Seed the high-water mark to the current live total (start of a measurement window).
pub fn reset_peak() {
    PEAK.store(CURRENT.load(Relaxed), Relaxed);
}
