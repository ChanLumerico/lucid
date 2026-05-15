// lucid/_C/bindings/bind_optim.cpp
//
// Registers all optimizer classes and LR scheduler classes on the top-level
// engine module.  The Python hierarchy mirrors reference framework's standard optimizer layout:
//
//   Optimizer (abstract base) — step(), zero_grad(), lr property
//     SGD, ASGD
//     Adam, AdamW, NAdam, RAdam, Adamax
//     RMSprop, Rprop
//     Adagrad, Adadelta
//
//   LRScheduler (abstract base) — step(), set_epoch(), epoch property
//     StepLR, ExponentialLR, MultiStepLR, CosineAnnealingLR,
//     LambdaLR, CyclicLR, NoamScheduler
//
//   ReduceLROnPlateau (not a LRScheduler subclass; uses metric-based step)
//
// All LRScheduler subclasses take an Optimizer& reference as their first
// constructor argument.  py::keep_alive<1, 2>() ensures the Optimizer object
// (argument 2, the bound "self" being constructed is 1) outlives the scheduler
// — without this the C++ reference would dangle if Python GCs the optimizer
// before the scheduler.
//
// pybind11/functional.h is included for LambdaLR which stores a
// std::function<double(int64_t)> populated from a Python callable.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../optim/Ada.h"
#include "../optim/Adam.h"
#include "../optim/LRScheduler.h"
#include "../optim/Optimizer.h"
#include "../optim/Prop.h"
#include "../optim/SGD.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers all optimizer and LR scheduler classes.
void register_optim(py::module_& m) {
    // Optimizer is the abstract base class; Python cannot instantiate it
    // directly but holds references through the concrete subclass hierarchy.
    // `lr` is read/write so Python can override the learning rate after
    // construction (e.g., manual LR warm-up without a scheduler).
    py::class_<Optimizer>(m, "Optimizer")
        .def("step", &Optimizer::step)
        .def("zero_grad", &Optimizer::zero_grad)
        .def_property("lr", &Optimizer::lr, &Optimizer::set_lr)
        .def_property_readonly("num_params", &Optimizer::num_params)
        // Versioned tag identifying the optimizer family; used by the Python
        // layer to validate state_dict compatibility on load.
        .def_property_readonly("state_dict_id", &Optimizer::state_dict_id)
        // Per-parameter mutable state (Adam moments, SGD momentum buffer ...).
        // Returns an ordered list of (name, tensors) pairs where each
        // ``tensors`` runs parallel to the parameter list — entries may be
        // ``None`` for slots that haven't been touched by step() yet.
        .def("state_buffers", &Optimizer::state_buffers)
        .def("load_state_buffers", &Optimizer::load_state_buffers, py::arg("bufs"))
        .def_property("step_count", &Optimizer::step_count, &Optimizer::set_step_count);

    // SGD with optional Nesterov momentum and L2 weight decay.
    py::class_<SGD, Optimizer>(m, "SGD")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      bool>(),
             py::arg("params"), py::arg("lr"), py::arg("momentum") = 0.0,
             py::arg("dampening") = 0.0, py::arg("weight_decay") = 0.0, py::arg("nesterov") = false,
             "SGD with momentum, Nesterov, and L2 weight decay.")
        .def_property_readonly("momentum", &SGD::momentum)
        .def_property_readonly("weight_decay", &SGD::weight_decay);

    // Adam (Kingma & Ba 2014).  amsgrad=True enables the AMSGrad variant which
    // uses the maximum of past squared gradients for a tighter convergence bound.
    py::class_<Adam, Optimizer>(m, "Adam")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      double, bool>(),
             py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("amsgrad") = false, "Adam (Kingma & Ba 2014) with optional L2 weight decay.")
        .def_property_readonly("beta1", &Adam::beta1)
        .def_property_readonly("beta2", &Adam::beta2)
        .def_property_readonly("eps", &Adam::eps);

    // AdamW: decoupled weight decay applied directly to parameters rather than
    // folded into the gradient (Loshchilov & Hutter 2017).
    py::class_<AdamW, Optimizer>(m, "AdamW")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      double>(),
             py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 1e-2,
             "AdamW (decoupled weight decay, Loshchilov & Hutter 2017).");

    // ASGD averages parameters after t0 steps; useful for convex problems.
    py::class_<ASGD, Optimizer>(m, "ASGD").def(
        py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double, double,
                 double>(),
        py::arg("params"), py::arg("lr") = 1e-3, py::arg("momentum") = 0.0,
        py::arg("weight_decay") = 0.0, py::arg("alpha") = 0.75, py::arg("t0") = 1e6,
        py::arg("lambd") = 1e-4, "Averaged SGD.");

    py::class_<NAdam, Optimizer>(m, "NAdam")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      double, double>(),
             py::arg("params"), py::arg("lr") = 2e-3, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("momentum_decay") = 0.004, "Nesterov-accelerated Adam.");

    py::class_<RAdam, Optimizer>(m, "RAdam")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      double>(),
             py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             "Rectified Adam (Liu et al. 2020).");

    py::class_<RMSprop, Optimizer>(m, "RMSprop")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      double, bool>(),
             py::arg("params"), py::arg("lr") = 1e-2, py::arg("alpha") = 0.99,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0, py::arg("momentum") = 0.0,
             py::arg("centered") = false, "RMSprop with optional centered variance and momentum.");

    py::class_<Rprop, Optimizer>(m, "Rprop")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      double>(),
             py::arg("params"), py::arg("lr") = 1e-2, py::arg("eta_minus") = 0.5,
             py::arg("eta_plus") = 1.2, py::arg("step_min") = 1e-6, py::arg("step_max") = 50.0,
             "Resilient backprop (Rprop).");

    py::class_<Adagrad, Optimizer>(m, "Adagrad")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 1e-2, py::arg("eps") = 1e-10,
             py::arg("weight_decay") = 0.0, py::arg("initial_accumulator_value") = 0.0,
             "Adagrad: per-parameter accumulator of squared grads.");

    py::class_<Adadelta, Optimizer>(m, "Adadelta")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 1.0, py::arg("rho") = 0.9, py::arg("eps") = 1e-6,
             py::arg("weight_decay") = 0.0, "Adadelta: parameter-free adaptive LR (Zeiler 2012).");

    py::class_<Adamax, Optimizer>(m, "Adamax")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      double>(),
             py::arg("params"), py::arg("lr") = 2e-3, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             "Adamax: Adam with infinity norm.");

    // LRScheduler is the abstract base for epoch-based schedules.  Subclasses
    // store a raw reference to the Optimizer and adjust its lr on each call
    // to step().  py::keep_alive<1, 2>() in each subclass constructor prevents
    // Python from GC-ing the optimizer before the scheduler is destroyed.
    py::class_<LRScheduler>(m, "LRScheduler")
        .def("step", &LRScheduler::step)
        .def("set_epoch", &LRScheduler::set_epoch, py::arg("epoch"))
        .def_property_readonly("epoch", &LRScheduler::epoch);

    // StepLR: multiply lr by gamma every step_size epochs.
    py::class_<StepLR, LRScheduler>(m, "StepLR")
        .def(py::init<Optimizer&, std::int64_t, double>(), py::arg("optimizer"),
             py::arg("step_size"), py::arg("gamma") = 0.1, py::keep_alive<1, 2>(),
             "Drop LR by `gamma` every `step_size` epochs.");

    py::class_<ExponentialLR, LRScheduler>(m, "ExponentialLR")
        .def(py::init<Optimizer&, double>(), py::arg("optimizer"), py::arg("gamma"),
             py::keep_alive<1, 2>(), "Multiply LR by `gamma` each epoch.");

    py::class_<MultiStepLR, LRScheduler>(m, "MultiStepLR")
        .def(py::init<Optimizer&, std::vector<std::int64_t>, double>(), py::arg("optimizer"),
             py::arg("milestones"), py::arg("gamma") = 0.1, py::keep_alive<1, 2>(),
             "Drop LR by `gamma` at each milestone epoch.");

    py::class_<CosineAnnealingLR, LRScheduler>(m, "CosineAnnealingLR")
        .def(py::init<Optimizer&, std::int64_t, double>(), py::arg("optimizer"), py::arg("T_max"),
             py::arg("eta_min") = 0.0, py::keep_alive<1, 2>(),
             "Cosine annealing schedule with period T_max.");

    // LambdaLR stores a Python callable via std::function.  pybind11 holds a
    // GIL-safe reference to the callable inside the std::function object, so
    // the Python function will not be GC-ed while the scheduler is alive.
    py::class_<LambdaLR, LRScheduler>(m, "LambdaLR")
        .def(py::init<Optimizer&, std::function<double(std::int64_t)>>(), py::arg("optimizer"),
             py::arg("lr_lambda"), py::keep_alive<1, 2>(),
             "Multiply base LR by lr_lambda(epoch). Lambda is a Python callable.");

    // CyclicLR needs a nested Mode enum.  The enum is defined on the class
    // object (cyclic) rather than on the module so it is accessed as
    // engine.CyclicLR.Mode.Triangular from Python.
    py::class_<CyclicLR, LRScheduler> cyclic(m, "CyclicLR");
    py::enum_<CyclicLR::Mode>(cyclic, "Mode")
        .value("Triangular", CyclicLR::Mode::Triangular)
        .value("Triangular2", CyclicLR::Mode::Triangular2)
        .value("ExpRange", CyclicLR::Mode::ExpRange);
    cyclic.def(
        py::init<Optimizer&, double, double, std::int64_t, std::int64_t, CyclicLR::Mode, double>(),
        py::arg("optimizer"), py::arg("base_lr"), py::arg("max_lr"), py::arg("step_size_up"),
        py::arg("step_size_down") = 0, py::arg("mode") = CyclicLR::Mode::Triangular,
        py::arg("gamma") = 1.0, py::keep_alive<1, 2>(),
        "Cyclic LR (Smith 2017): triangular wave between base_lr and max_lr.");

    // Noam schedule: lr = factor * model_size^(-0.5) *
    //   min(step^(-0.5), step * warmup_steps^(-1.5)).
    // Commonly used in Transformer training (Vaswani et al. 2017).
    py::class_<NoamScheduler, LRScheduler>(m, "NoamScheduler")
        .def(py::init<Optimizer&, std::int64_t, std::int64_t, double>(), py::arg("optimizer"),
             py::arg("model_size"), py::arg("warmup_steps"), py::arg("factor") = 1.0,
             py::keep_alive<1, 2>(), "Transformer-style warmup-then-decay schedule.");

    // ReduceLROnPlateau is NOT a LRScheduler subclass because its step(metric)
    // signature differs — it receives a scalar metric value rather than
    // advancing an epoch counter.  Mode and ThresholdMode nested enums are
    // defined on the class object (rlrp) for the same reason as CyclicLR.Mode.
    py::class_<ReduceLROnPlateau> rlrp(m, "ReduceLROnPlateau");
    py::enum_<ReduceLROnPlateau::Mode>(rlrp, "Mode")
        .value("Min", ReduceLROnPlateau::Mode::Min)
        .value("Max", ReduceLROnPlateau::Mode::Max);
    py::enum_<ReduceLROnPlateau::ThresholdMode>(rlrp, "ThresholdMode")
        .value("Rel", ReduceLROnPlateau::ThresholdMode::Rel)
        .value("Abs", ReduceLROnPlateau::ThresholdMode::Abs);
    rlrp.def(py::init<Optimizer&, ReduceLROnPlateau::Mode, double, std::int64_t, double,
                      ReduceLROnPlateau::ThresholdMode, std::int64_t, double, double>(),
             py::arg("optimizer"), py::arg("mode") = ReduceLROnPlateau::Mode::Min,
             py::arg("factor") = 0.1, py::arg("patience") = 10, py::arg("threshold") = 1e-4,
             py::arg("threshold_mode") = ReduceLROnPlateau::ThresholdMode::Rel,
             py::arg("cooldown") = 0, py::arg("min_lr") = 0.0, py::arg("eps") = 1e-8,
             py::keep_alive<1, 2>())
        .def("step", &ReduceLROnPlateau::step, py::arg("metric"))
        .def_property_readonly("last_lr", &ReduceLROnPlateau::last_lr)
        .def_property_readonly("num_bad_epochs", &ReduceLROnPlateau::num_bad_epochs);
}

}  // namespace lucid::bindings
