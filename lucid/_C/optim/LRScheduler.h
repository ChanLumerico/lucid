#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "../api.h"

namespace lucid {

class Optimizer;

class LUCID_API LRScheduler {
public:
    explicit LRScheduler(Optimizer& opt);
    virtual ~LRScheduler() = default;

    LRScheduler(const LRScheduler&) = delete;
    LRScheduler& operator=(const LRScheduler&) = delete;

    void step();

    void set_epoch(std::int64_t epoch);
    std::int64_t epoch() const { return epoch_; }

protected:
    virtual double compute_lr_at(std::int64_t epoch) const = 0;

    Optimizer& opt_;
    double base_lr_;
    std::int64_t epoch_;
};

class LUCID_API StepLR : public LRScheduler {
public:
    StepLR(Optimizer& opt, std::int64_t step_size, double gamma = 0.1);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::int64_t step_size_;
    double gamma_;
};

class LUCID_API ExponentialLR : public LRScheduler {
public:
    ExponentialLR(Optimizer& opt, double gamma);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    double gamma_;
};

class LUCID_API MultiStepLR : public LRScheduler {
public:
    MultiStepLR(Optimizer& opt, std::vector<std::int64_t> milestones, double gamma = 0.1);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::vector<std::int64_t> milestones_;
    double gamma_;
};

class LUCID_API CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& opt, std::int64_t T_max, double eta_min = 0.0);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::int64_t T_max_;
    double eta_min_;
};

class LUCID_API LambdaLR : public LRScheduler {
public:
    LambdaLR(Optimizer& opt, std::function<double(std::int64_t)> lr_lambda);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::function<double(std::int64_t)> lr_lambda_;
};

class LUCID_API ReduceLROnPlateau {
public:
    enum class Mode { Min, Max };
    enum class ThresholdMode { Rel, Abs };

    ReduceLROnPlateau(Optimizer& opt,
                      Mode mode = Mode::Min,
                      double factor = 0.1,
                      std::int64_t patience = 10,
                      double threshold = 1e-4,
                      ThresholdMode threshold_mode = ThresholdMode::Rel,
                      std::int64_t cooldown = 0,
                      double min_lr = 0.0,
                      double eps = 1e-8);

    void step(double metric);

    double last_lr() const { return last_lr_; }
    std::int64_t num_bad_epochs() const { return num_bad_epochs_; }

private:
    bool is_better(double metric) const;

    Optimizer& opt_;
    Mode mode_;
    double factor_;
    std::int64_t patience_;
    double threshold_;
    ThresholdMode threshold_mode_;
    std::int64_t cooldown_;
    double min_lr_;
    double eps_;

    double best_;
    std::int64_t num_bad_epochs_;
    std::int64_t cooldown_counter_;
    double last_lr_;
};

class LUCID_API CyclicLR : public LRScheduler {
public:
    enum class Mode { Triangular, Triangular2, ExpRange };

    CyclicLR(Optimizer& opt,
             double base_lr,
             double max_lr,
             std::int64_t step_size_up,
             std::int64_t step_size_down = 0,
             Mode mode = Mode::Triangular,
             double gamma = 1.0);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    double base_lr_cyc_, max_lr_;
    std::int64_t step_size_up_, total_size_;
    Mode mode_;
    double gamma_;
};

class LUCID_API NoamScheduler : public LRScheduler {
public:
    NoamScheduler(Optimizer& opt,
                  std::int64_t model_size,
                  std::int64_t warmup_steps,
                  double factor = 1.0);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::int64_t model_size_;
    std::int64_t warmup_steps_;
    double factor_;
};

}  // namespace lucid
