#include "LRScheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "../core/Exceptions.h"
#include "Optimizer.h"

namespace lucid {

LRScheduler::LRScheduler(Optimizer& opt) : opt_(opt), base_lr_(opt.lr()), epoch_(0) {}

void LRScheduler::step() {
    ++epoch_;
    opt_.set_lr(compute_lr_at(epoch_));
}

void LRScheduler::set_epoch(std::int64_t epoch) {
    epoch_ = epoch;
    opt_.set_lr(compute_lr_at(epoch_));
}

// ---------------- StepLR ----------------

StepLR::StepLR(Optimizer& opt, std::int64_t step_size, double gamma)
    : LRScheduler(opt), step_size_(step_size), gamma_(gamma) {
    if (step_size_ <= 0)
        throw LucidError("StepLR: step_size must be > 0");
}
double StepLR::compute_lr_at(std::int64_t epoch) const {
    const std::int64_t k = epoch / step_size_;
    return base_lr_ * std::pow(gamma_, static_cast<double>(k));
}

// ---------------- ExponentialLR ----------------

ExponentialLR::ExponentialLR(Optimizer& opt, double gamma) : LRScheduler(opt), gamma_(gamma) {}
double ExponentialLR::compute_lr_at(std::int64_t epoch) const {
    return base_lr_ * std::pow(gamma_, static_cast<double>(epoch));
}

// ---------------- MultiStepLR ----------------

MultiStepLR::MultiStepLR(Optimizer& opt, std::vector<std::int64_t> milestones, double gamma)
    : LRScheduler(opt), milestones_(std::move(milestones)), gamma_(gamma) {
    std::sort(milestones_.begin(), milestones_.end());
}
double MultiStepLR::compute_lr_at(std::int64_t epoch) const {
    std::int64_t hits = 0;
    for (auto m : milestones_) {
        if (epoch >= m)
            ++hits;
        else
            break;
    }
    return base_lr_ * std::pow(gamma_, static_cast<double>(hits));
}

// ---------------- CosineAnnealingLR ----------------

CosineAnnealingLR::CosineAnnealingLR(Optimizer& opt, std::int64_t T_max, double eta_min)
    : LRScheduler(opt), T_max_(T_max), eta_min_(eta_min) {
    if (T_max_ <= 0)
        throw LucidError("CosineAnnealingLR: T_max must be > 0");
}
double CosineAnnealingLR::compute_lr_at(std::int64_t epoch) const {
    // Match PyTorch: the cosine continues past T_max (no clamp). User code
    // that wants a hold at eta_min after T_max can wrap with min(...).
    constexpr double PI = 3.14159265358979323846;
    return eta_min_ +
           0.5 * (base_lr_ - eta_min_) *
               (1.0 + std::cos(PI * static_cast<double>(epoch) / static_cast<double>(T_max_)));
}

// ---------------- LambdaLR ----------------

LambdaLR::LambdaLR(Optimizer& opt, std::function<double(std::int64_t)> lr_lambda)
    : LRScheduler(opt), lr_lambda_(std::move(lr_lambda)) {}

double LambdaLR::compute_lr_at(std::int64_t epoch) const {
    return base_lr_ * lr_lambda_(epoch);
}

// ---------------- ReduceLROnPlateau ----------------

ReduceLROnPlateau::ReduceLROnPlateau(Optimizer& opt,
                                     Mode mode,
                                     double factor,
                                     std::int64_t patience,
                                     double threshold,
                                     ThresholdMode threshold_mode,
                                     std::int64_t cooldown,
                                     double min_lr,
                                     double eps)
    : opt_(opt),
      mode_(mode),
      factor_(factor),
      patience_(patience),
      threshold_(threshold),
      threshold_mode_(threshold_mode),
      cooldown_(cooldown),
      min_lr_(min_lr),
      eps_(eps),
      best_(mode == Mode::Min ? std::numeric_limits<double>::infinity()
                              : -std::numeric_limits<double>::infinity()),
      num_bad_epochs_(0),
      cooldown_counter_(0),
      last_lr_(opt.lr()) {
    if (factor_ >= 1.0)
        throw LucidError("ReduceLROnPlateau: factor must be < 1.0");
}

bool ReduceLROnPlateau::is_better(double metric) const {
    double thresh;
    if (threshold_mode_ == ThresholdMode::Rel) {
        thresh = (mode_ == Mode::Min) ? best_ * (1.0 - threshold_) : best_ * (1.0 + threshold_);
    } else {
        thresh = (mode_ == Mode::Min) ? best_ - threshold_ : best_ + threshold_;
    }
    return (mode_ == Mode::Min) ? (metric < thresh) : (metric > thresh);
}

void ReduceLROnPlateau::step(double metric) {
    if (is_better(metric)) {
        best_ = metric;
        num_bad_epochs_ = 0;
    } else {
        ++num_bad_epochs_;
    }
    if (cooldown_counter_ > 0) {
        --cooldown_counter_;
        num_bad_epochs_ = 0;
    }
    if (num_bad_epochs_ > patience_) {
        const double cur = opt_.lr();
        const double new_lr = std::max(cur * factor_, min_lr_);
        if (cur - new_lr > eps_) {
            opt_.set_lr(new_lr);
            last_lr_ = new_lr;
        }
        cooldown_counter_ = cooldown_;
        num_bad_epochs_ = 0;
    }
}

// ---------------- CyclicLR ----------------

CyclicLR::CyclicLR(Optimizer& opt,
                   double base_lr,
                   double max_lr,
                   std::int64_t step_size_up,
                   std::int64_t step_size_down,
                   Mode mode,
                   double gamma)
    : LRScheduler(opt),
      base_lr_cyc_(base_lr),
      max_lr_(max_lr),
      step_size_up_(step_size_up),
      total_size_(step_size_up + (step_size_down > 0 ? step_size_down : step_size_up)),
      mode_(mode),
      gamma_(gamma) {
    if (base_lr >= max_lr)
        throw LucidError("CyclicLR: base_lr must be < max_lr");
    if (step_size_up <= 0)
        throw LucidError("CyclicLR: step_size_up must be > 0");
}

double CyclicLR::compute_lr_at(std::int64_t epoch) const {
    const std::int64_t cycle = epoch / total_size_;
    const double x = std::abs(static_cast<double>(epoch) / static_cast<double>(step_size_up_) -
                              2.0 * cycle - 1.0);
    double scale = 1.0;
    switch (mode_) {
        case Mode::Triangular:
            scale = 1.0;
            break;
        case Mode::Triangular2:
            scale = 1.0 / static_cast<double>(1ll << cycle);
            break;
        case Mode::ExpRange:
            scale = std::pow(gamma_, static_cast<double>(epoch));
            break;
    }
    return base_lr_cyc_ + (max_lr_ - base_lr_cyc_) * std::max(0.0, 1.0 - x) * scale;
}

// ---------------- NoamScheduler ----------------

NoamScheduler::NoamScheduler(Optimizer& opt,
                             std::int64_t model_size,
                             std::int64_t warmup_steps,
                             double factor)
    : LRScheduler(opt), model_size_(model_size), warmup_steps_(warmup_steps), factor_(factor) {
    if (model_size <= 0)
        throw LucidError("NoamScheduler: model_size must be > 0");
    if (warmup_steps <= 0)
        throw LucidError("NoamScheduler: warmup_steps must be > 0");
    if (factor <= 0)
        throw LucidError("NoamScheduler: factor must be > 0");
}

double NoamScheduler::compute_lr_at(std::int64_t epoch) const {
    const double step = std::max<std::int64_t>(epoch, 1);
    const double scale = factor_ * std::pow(static_cast<double>(model_size_), -0.5);
    const double warmup_term = step * std::pow(static_cast<double>(warmup_steps_), -1.5);
    const double decay_term = std::pow(step, -0.5);
    return scale * std::min(decay_term, warmup_term);
}

}  // namespace lucid
