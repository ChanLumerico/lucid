// lucid/_C/compile/OpEmitters/OpEmitter.cpp
//
// Pure-C++ implementation of the BuilderContext + emitter registry.
// ObjC types are kept as ``void*`` here — the casts happen in each
// emitter's .mm file.

#include "OpEmitter.h"

#include <mutex>
#include <utility>

namespace lucid::compile {

BuilderContext::BuilderContext(void* graph_void, Device device)
    : graph_(graph_void), device_(device) {}

void* BuilderContext::resolve(TensorId id) const {
    auto it = tensors_.find(id);
    return it == tensors_.end() ? nullptr : it->second;
}

void BuilderContext::bind(TensorId id, void* tensor) {
    tensors_[id] = tensor;
}

namespace {
// Process-global emitter registry.  Mutex-protected because the
// registry is mutated from static initialisers across multiple
// translation units; readers (the builder) also use the same lock.
struct EmitterRegistry {
    std::mutex mu;
    std::unordered_map<std::string, std::unique_ptr<OpEmitter>> map;
};

EmitterRegistry& registry() {
    static EmitterRegistry* instance = new EmitterRegistry();
    return *instance;
}
}  // namespace

OpEmitter* find_emitter(std::string_view op_name) {
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    // unordered_map lookup needs std::string here — string_view→string copy
    // is unavoidable for the heterogeneous lookup until C++20 transparent
    // hashers are wired up (cheap relative to MPSGraph compile cost).
    auto it = r.map.find(std::string(op_name));
    return it == r.map.end() ? nullptr : it->second.get();
}

void register_emitter(std::unique_ptr<OpEmitter> emitter) {
    if (!emitter)
        return;
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    std::string key{emitter->op_name()};
    r.map[std::move(key)] = std::move(emitter);
}

}  // namespace lucid::compile
