#pragma once

namespace lucid {

class Determinism {
public:
    static bool is_enabled();
    static void set_enabled(bool value);
};

}  // namespace lucid
