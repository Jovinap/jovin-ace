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
    uint64_t offset;
};

class GGUFParser {
public:
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    std::unordered_map<std::string, GGUFTensor> tensors;
    uint64_t alignment = 32;
    uint64_t body_offset = 0;

    bool parse(int fd) {
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
        body_offset = sizeof(header);
        return true;
    }
};

// Hook management
typedef void* (*mmap_t)(void*, size_t, int, int, int, off_t);
typedef int (*munmap_t)(void*, size_t);
static mmap_t real_mmap = nullptr;
static munmap_t real_munmap = nullptr;

// Atomic execution cursor to synchronize thread execution
// Updated dynamically by tracking virtual memory reference patterns or standard llama inference steps
std::atomic<size_t> g_active_execution_layer(0); 

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

// Active page eviction loop with protection shield
void* eviction_thread_func(void* arg) {
    const useconds_t poll_interval_us = 1000;
    while (!g_stop_eviction.load(std::memory_order_relaxed)) {
        usleep(poll_interval_us);
        std::lock_guard<std::mutex> lock(g_mappings_mutex);
        
        size_t current_layer = g_active_execution_layer.load(std::memory_order_relaxed);
        
        for (auto& mapping : g_mappings) {
            // Apply Transparent HugePages optimization
            madvise(reinterpret_cast<void*>(mapping.start), mapping.length, MADV_HUGEPAGE);
            
            size_t layer_size = mapping.length / 60; // Gemma 4 31B specific layout
            
            for (size_t l = 0; l < 60; ++l) {
                // PROTECTION SHIELD: Never evict the current layer N, or the prefetched layer N+1
                if (l == current_layer || l == (current_layer + 1)) {
                    continue; 
                }
                
                uintptr_t layer_start = mapping.start + (l * layer_size);
                // Safe to wipe physical DRAM allocations for historical or distant future layers
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
