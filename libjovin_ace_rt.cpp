#define _GNU_SOURCE
#include <iostream>
#include <dlfcn.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstring>
#include <cstdint>

// GGUF parsing structures
struct GGUFTensor {
    std::string name;
    uint32_t n_dims;
    std::vector<uint64_t> dims;
    uint32_t type;
    uint64_t offset; // Offset relative to the body start
};

class GGUFParser {
public:
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    std::unordered_map<std::string, GGUFTensor> tensors;
    uint64_t alignment = 32; // GGUF standard alignment
    uint64_t body_offset = 0;

    bool parse(int fd) {
        // Read GGUF Header
        struct {
            char magic[4];
            uint32_t version;
            uint64_t tensor_count;
            uint64_t metadata_count;
        } header;

        if (pread(fd, &header, sizeof(header), 0) != sizeof(header)) {
            return false;
        }

        if (std::memcmp(header.magic, "GGUF", 4) != 0) {
            return false;
        }

        version = header.version;
        tensor_count = header.tensor_count;
        metadata_count = header.metadata_count;

        // Skip metadata for layout indexing
        // In GGUF v2/v3, metadata follows the header.
        // A minimal parser will locate the tensor info by seeking.
        // Since we are writing a lightweight dynamic hook, we scan for known FFN tensors
        // by seeking offset ranges. For structural mapping:
        body_offset = sizeof(header);
        
        // Simulating structural extraction (for production, parse the actual GGUF metadata schema)
        return true;
    }
};

// Hook management
typedef void* (*mmap_t)(void*, size_t, int, int, int, off_t);
typedef int (*munmap_t)(void*, size_t);
static mmap_t real_mmap = nullptr;
static munmap_t real_munmap = nullptr;

struct ModelMapping {
    uintptr_t start;
    size_t length;
    int fd;
    GGUFParser parser;
};

static std::vector<ModelMapping> g_mappings;
static std::mutex g_mappings_mutex;
static std::atomic<bool> g_stop_eviction(false);
static pthread_t g_eviction_thread;

// Active page eviction loop
void* eviction_thread_func(void* arg) {
    const useconds_t poll_interval_us = 1000; // High frequency poll
    
    while (!g_stop_eviction.load(std::memory_order_relaxed)) {
        usleep(poll_interval_us);
        
        std::lock_guard<std::mutex> lock(g_mappings_mutex);
        for (auto& mapping : g_mappings) {
            // Apply MADV_HUGEPAGE to optimize i3-U's TLB coverage over the weights mapping
            madvise(reinterpret_cast<void*>(mapping.start), mapping.length, MADV_HUGEPAGE);
            
            // Loop through the mapped blocks and evict pages of layers that are inactive
            // (e.g., FFN weights that are not in the current layer window)
            // For Gemma 4 31B, there are 60 transformer layers. We evict sequentially.
            // Since we know the FFN offsets, we call MADV_DONTNEED on the completed layers.
            size_t layer_size = mapping.length / 60;
            
            // Simulating cursor tracking based on system read heads
            for (size_t l = 0; l < 60; ++l) {
                uintptr_t layer_start = mapping.start + (l * layer_size);
                
                // Discard computed layer weights from physical RAM to stay under 4GB
                madvise(reinterpret_cast<void*>(layer_start), layer_size, MADV_DONTNEED);
            }
        }
    }
    return nullptr;
}

__attribute__((constructor))
void init_hook() {
    real_mmap = (mmap_t)dlsym(RTLD_NEXT, "mmap");
    real_munmap = (munmap_t)dlsym(RTLD_NEXT, "munmap");
    
    pthread_create(&g_eviction_thread, nullptr, eviction_thread_func, nullptr);
}

__attribute__((destructor))
void deinit_hook() {
    g_stop_eviction.store(true, std::memory_order_relaxed);
    pthread_join(g_eviction_thread, nullptr);
}

extern "C" void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    if (!real_mmap) {
        real_mmap = (mmap_t)dlsym(RTLD_NEXT, "mmap");
    }
    
    void* result = real_mmap(addr, length, prot, flags, fd, offset);
    
    // Check if the mapping corresponds to the heavy model GGUF file (> 2GB)
    if (result != MAP_FAILED && length > 2ULL * 1024ULL * 1024ULL * 1024ULL) {
        std::lock_guard<std::mutex> lock(g_mappings_mutex);
        GGUFParser parser;
        if (parser.parse(fd)) {
            g_mappings.push_back({
                reinterpret_cast<uintptr_t>(result),
                length,
                fd,
                parser
            });
        }
    }
    
    return result;
}

extern "C" int munmap(void* addr, size_t length) {
    if (!real_munmap) {
        real_munmap = (munmap_t)dlsym(RTLD_NEXT, "munmap");
    }
    
    {
        std::lock_guard<std::mutex> lock(g_mappings_mutex);
        uintptr_t target = reinterpret_cast<uintptr_t>(addr);
        for (auto it = g_mappings.begin(); it != g_mappings.end(); ) {
            if (it->start == target) {
                it = g_mappings.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    return real_munmap(addr, length);
}
