// lucid/_C/optim/LRScheduler.cpp
//
// Implementations of all learning-rate schedule classes. Each
// compute_lr_at() is a pure function of epoch and the schedule's
// hyperparameters; none modify optimizer or scheduler state directly
// (that is handled by the base step() / set_epoch() methods).

#include "LRScheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "Optimizer.h"

namespace lucid {

// Capture base_lr from the optimizer's current learning rate so that
// schedule formulas can reference a stable baseline even if the
// optimizer lr has been externally modified before the scheduler runs.
LRScheduler::LRScheduler(Optimizer& opt) : opt_(opt), base_lr_(opt.lr()), epoch_(0) {}

// Increment epoch and push the new lr into the optimizer.
void LRScheduler::step() {
    ++epoch_;
    opt_.set_lr(compute_lr_at(epoch_));
}

// Jump to an arbitrary epoch and immediately apply the resulting lr.
void LRScheduler::set_epoch(std::int64_t epoch) {
    epoch_ = epoch;
    opt_.set_lr(compute_lr_at(epoch_));
}

StepLR::StepLR(Optimizer& opt, std::int64_t step_size, double gamma)
    : LRScheduler(opt), step_size_(step_size), gamma_(gamma) {
    if (step_size_ <= 0)
        ErrorBuilder("StepLR").fail("step_size must be > 0");
}

// Decay lr by gamma to the power of (epoch / step_size), so every
// step_size epochs the lr is multiplied by one factor of gamma.
double StepLR::compute_lr_at(std::int64_t epoch) const {
    const std::int64_t k = epoch / step_size_;
    return base_lr_ * std::pow(gamma_, static_cast<double>(k));
}

ExponentialLR::ExponentialLR(Optimizer& opt, double gamma) : LRScheduler(opt), gamma_(gamma) {}

// lr decays by gamma every epoch: lr = base_lr * gamma^epoch.
double ExponentialLR::compute_lr_at(std::int64_t epoch) const {
    return base_lr_ * std::pow(gamma_, static_cast<double>(epoch));
}

// milestones_ is sorted ascending at construction so that the linear
// scan in compute_lr_at terminates early once we pass the current epoch.
MultiStepLR::MultiStepLR(Optimizer& opt, std::vector<std::int64_t> milestones, double gamma)
    : LRScheduler(opt), milestones_(std::move(milestones)), gamma_(gamma) {
    std::sort(milestones_.begin(), milestones_.end());
}

// Count how many milestones have been passed at this epoch; each
// passed milestone multiplies by gamma once.
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

CosineAnnealingLR::CosineAnnealingLR(Optimizer& opt, std::int64_t T_max, double eta_min)
    : LRScheduler(opt), T_max_(T_max), eta_min_(eta_min) {
    if (T_max_ <= 0)
        ErrorBuilder("CosineAnnealingLR").fail("T_max must be > 0");
}

// Cosine half-period: starts at base_lr_ when epoch=0, reaches eta_min_
// at epoch=T_max, then rises again (the schedule repeats if you go beyond
// T_max without restarting).
double CosineAnnealingLR::compute_lr_at(std::int64_t epoch) const {
    constexpr double PI = 3.14159265358979323846;
    return eta_min_ +
           0.5 * (base_lr_ - eta_min_) *
               (1.0 + std::cos(PI * static_cast<double>(epoch) / static_cast<double>(T_max_)));
}

LambdaLR::LambdaLR(Optimizer& opt, std::function<double(std::int64_t)> lr_lambda)
    : LRScheduler(opt), lr_lambda_(std::move(lr_lambda)) {}

// The user-supplied lambda returns a multiplicative factor; the actual
// lr is base_lr_ times that factor.
double LambdaLR::compute_lr_at(std::int64_t epoch) const {
    return base_lr_ * lr_lambda_(epoch);
}

// Initialize best_ to infinity (Min mode) or -infinity (Max mode) so
// the very first metric always registers as an improvement.
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
        ErrorBuilder("ReduceLROnPlateau").fail("factor must be < 1.0");
}

// Check whether metric represents a sufficiently large improvement over
// best_. ThresholdMode::Rel computes the threshold as a relative fraction
// of best_; ThresholdMode::Abs uses an absolute delta.
bool ReduceLROnPlateau::is_better(double metric) const {
    double thresh;
    if (threshold_mode_ == ThresholdMode::Rel) {
        thresh = (mode_ == Mode::Min) ? best_ * (1.0 - threshold_) : best_ * (1.0 + threshold_);
    } else {
        thresh = (mode_ == Mode::Min) ? best_ - threshold_ : best_ + threshold_;
    }
    return (mode_ == Mode::Min) ? (metric < thresh) : (metric > thresh);
}

// Update best_, bad-epoch count, and cooldown counter; reduce lr if
// the patience threshold is exceeded and no cooldown is active.
// num_bad_epochs_ is reset to 0 after each reduction and during the
// cooldown period to avoid compounding reductions.
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
        // Only apply the reduction if the change is larger than eps_,
        // preventing no-op updates when lr is already at min_lr_.
        if (cur - new_lr > eps_) {
            opt_.set_lr(new_lr);
            last_lr_ = new_lr;
        }
        cooldown_counter_ = cooldown_;
        num_bad_epochs_ = 0;
    }
}

// step_size_down defaults to step_size_up when 0 so symmetric triangles
// are easy to specify. total_size_ is the full cycle length.
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
        ErrorBuilder("CyclicLR").fail("base_lr must be < max_lr");
    if (step_size_up <= 0)
        ErrorBuilder("CyclicLR").fail("step_size_up must be > 0");
}

// Compute the triangular wave position x in [0, 1] and then scale
// the lr range by the mode-dependent amplitude factor.
double CyclicLR::compute_lr_at(std::int64_t epoch) const {
    const std::int64_t cycle = epoch / total_size_;
    // x measures how far we are from the peak of the triangle.
    // x == 0 means at the peak (max_lr); x == 1 means at the base.
    const double x = std::abs(static_cast<double>(epoch) / static_cast<double>(step_size_up_) -
                              2.0 * cycle - 1.0);
    double scale = 1.0;
    switch (mode_) {
    case Mode::Triangular:
        scale = 1.0;
        break;
    case Mode::Triangular2:
        // Halve amplitude each cycle.
        scale = 1.0 / static_cast<double>(1ll << cycle);
        break;
    case Mode::ExpRange:
        // Decay amplitude exponentially with epoch count.
        scale = std::pow(gamma_, static_cast<double>(epoch));
        break;
    }
    return base_lr_cyc_ + (max_lr_ - base_lr_cyc_) * std::max(0.0, 1.0 - x) * scale;
}

NoamScheduler::NoamScheduler(Optimizer& opt,
                             std::int64_t model_size,
                             std::int64_t warmup_steps,
                             double factor)
    : LRScheduler(opt), model_size_(model_size), warmup_steps_(warmup_steps), factor_(factor) {
    if (model_size <= 0)
        ErrorBuilder("NoamScheduler").fail("model_size must be > 0");
    if (warmup_steps <= 0)
        ErrorBuilder("NoamScheduler").fail("warmup_steps must be > 0");
    if (factor <= 0)
        ErrorBuilder("NoamScheduler").fail("factor must be > 0");
}

// The Noam (transformer) schedule linearly increases lr during warmup then
// decays proportionally to step^(-0.5). The min() selects the warmup slope
// for early steps and the decay term for later steps, creating a peak at
// step = warmup_steps.
double NoamScheduler::compute_lr_at(std::int64_t epoch) const {
    // Clamp step to at least 1 to avoid log(0) / 0^(-0.5) pathologies.
    const double step = std::max<std::int64_t>(epoch, 1);
    const double scale = factor_ * std::pow(static_cast<double>(model_size_), -0.5);
    const double warmup_term = step * std::pow(static_cast<double>(warmup_steps_), -1.5);
    const double decay_term = std::pow(step, -0.5);
    return scale * std::min(decay_term, warmup_term);
}

}  // namespace lucid
