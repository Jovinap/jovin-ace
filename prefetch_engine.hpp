#pragma once
#include <liburing.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <vector>
#include <atomic>
#include <cstdlib>

class JovinAcePrefetchEngine {
private:
    struct io_uring ring;
    int ext_fd;              // File descriptor for slow external HP storage
    size_t layer_bytes;
    std::vector<void*> buffers; // Page-aligned double buffers in RAM
    std::atomic<int> active_buffer_idx;
    bool sqpoll_enabled;

public:
    JovinAcePrefetchEngine(const char* gguf_path, size_t layer_sz) 
        : layer_bytes(layer_sz), active_buffer_idx(0), sqpoll_enabled(false) {
        
        // Open the external GGUF file with O_DIRECT and O_NOATIME to bypass Linux page cache completely
        // Bypassing page cache eliminates write-back overhead and duplicate copy operations in RAM
        ext_fd = open(gguf_path, O_RDONLY | O_DIRECT | O_NOATIME);
        if (ext_fd < 0) {
            throw std::runtime_error("Failed to open GGUF file with O_DIRECT | O_NOATIME on external storage");
        }

        // Setup io_uring structures
        struct io_uring_params params;
        std::memset(&params, 0, sizeof(params));
        
        // Use IORING_SETUP_SQPOLL to offload Submission Queue polling to a kernel thread.
        // This is critical for the i3-U CPU, as it frees the user-space thread from context switching
        // and submission overhead, letting the kernel handle pre-fetching work on kernel threads.
        params.flags = IORING_SETUP_SQPOLL;
        params.sq_thread_idle = 2000; // Idle timeout in ms before SQ thread sleeps
        
        int ret = io_uring_queue_init_params(128, &ring, &params);
        if (ret < 0) {
            // Fallback to standard queue setup if SQPOLL privileges are lacking (e.g. non-root without cap)
            ret = io_uring_queue_init(128, &ring, 0);
            if (ret < 0) {
                close(ext_fd);
                throw std::runtime_error("Failed to initialize io_uring queue");
            }
        } else {
            sqpoll_enabled = true;
        }

        // Allocate 2MB/4KB page-aligned buffers for double buffering to prevent direct I/O address faults
        for (int i = 0; i < 2; ++i) {
            void* aligned_buf = nullptr;
            // 4096 alignment is the minimum required for O_DIRECT reads
            if (posix_memalign(&aligned_buf, 4096, layer_bytes) != 0) {
                throw std::runtime_error("Failed to allocate page-aligned buffer for Direct I/O");
            }
            // Advise HugePages on the prefetch buffers to avoid TLB misses on the i3-U
            madvise(aligned_buf, layer_bytes, MADV_HUGEPAGE);
            buffers.push_back(aligned_buf);
        }
    }

    ~JovinAcePrefetchEngine() {
        io_uring_queue_exit(&ring);
        close(ext_fd);
        for (void* buf : buffers) {
            std::free(buf);
        }
    }

    // Submit an asynchronous read to fetch Layer N+1 from SSD while Layer N is active in calculation
    void submit_layer_prefetch(size_t next_layer_offset) {
        // Target the alternate buffer index (double-buffering)
        int prefetch_idx = 1 - active_buffer_idx.load(std::memory_order_relaxed);
        void* dest_buf = buffers[prefetch_idx];

        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) {
            throw std::runtime_error("io_uring Submission Queue is full");
        }

        // Prepare the asynchronous read operation
        io_uring_prep_read(sqe, ext_fd, dest_buf, layer_bytes, next_layer_offset);
        
        // Tag submission entry with the destination buffer pointer for identification on completion
        io_uring_sqe_set_data(sqe, dest_buf);

        // Submit the request. If SQPOLL is active, the kernel thread picks this up instantly.
        io_uring_submit(&ring);
    }

    // Block current thread until the async read finishes, then return the buffer pointer
    void* wait_for_layer_completion() {
        struct io_uring_cqe* cqe;
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) {
            throw std::runtime_error("io_uring wait failed");
        }

        void* completed_buf = io_uring_cqe_get_data(cqe);
        io_uring_cqe_seen(&ring, cqe); // Clear completion entry

        // Swap the active buffer index to the freshly read buffer
        active_buffer_idx.store(1 - active_buffer_idx.load(std::memory_order_relaxed), std::memory_order_release);

        return completed_buf;
    }

    bool is_sqpoll_enabled() const {
        return sqpoll_enabled;
    }
};
```Prefixing lines with + for additions, - for deletions, and a space for unchanged lines:
```diff
-old_function_name()
+new_function_name()
 unchanged_line()
```
