#pragma once

namespace lucid {

class GradMode {
public:
    static bool is_enabled();
    static void set_enabled(bool value);
};

class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_;
};

}  // namespace lucid
