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

// Each optimizer class also takes its hyper-parameters as a dict keyed by
// the Python wrapper's ``param_groups`` names: ``set_hyperparams`` applies
// a group's edited values between steps, and the static
// ``check_hyperparams`` runs the same rules on a group the wrapper is given.
// One reader per class serves both, so they cannot take different names.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <initializer_list>
#include <string>
#include <tuple>

#include "../core/ErrorBuilder.h"
#include "../core/TensorImpl.h"
#include "../optim/Ada.h"
#include "../optim/Adam.h"
#include "../optim/LRScheduler.h"
#include "../optim/Optimizer.h"
#include "../optim/Prop.h"
#include "../optim/SGD.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

// One optimizer's hyper-parameters as the Python wrapper passes them.
//
// The dict must hold exactly the names the engine takes.  A name it did not
// take would be dropped without a word, and a missing one means the
// wrapper's hyper-parameter table has drifted from this binding; either is
// an ``InvalidArgument`` naming the key.
class HyperparamDict {
public:
    HyperparamDict(const char* op, const py::dict& values, std::initializer_list<const char*> names)
        : op_(op), values_(values) {
        for (const char* name : names) {
            if (!values_.contains(name))
                ErrorBuilder(op_).invalid_argument(std::string("missing hyper-parameter '") + name +
                                                   "'");
        }
        if (py::len(values_) == names.size())
            return;
        for (const auto& item : values_) {
            const std::string key = py::str(item.first);
            const bool known = std::any_of(names.begin(), names.end(),
                                           [&key](const char* name) { return key == name; });
            if (!known)
                ErrorBuilder(op_).invalid_argument("unknown hyper-parameter '" + key + "'");
        }
    }

    // A numeric hyper-parameter; anything ``float()`` accepts.
    double num(const char* name) const {
        const py::object value = values_[name];
        try {
            return value.cast<double>();
        } catch (const py::cast_error&) {
            throw py::type_error(std::string(op_) + ": " + name + " must be a number, not " +
                                 Py_TYPE(value.ptr())->tp_name);
        }
    }

    // A boolean hyper-parameter; read by truth value, as ``if flag:`` reads it.
    bool flag(const char* name) const {
        const py::object value = values_[name];
        try {
            return value.cast<bool>();
        } catch (const py::cast_error&) {
            throw py::type_error(std::string(op_) + ": " + name + " must be a bool, not " +
                                 Py_TYPE(value.ptr())->tp_name);
        }
    }

private:
    const char* op_;
    const py::dict& values_;
};

// Binds ``set_hyperparams`` and the static ``check_hyperparams`` of ``Opt``.
//
// ``read`` turns the wrapper's dict into the argument tuple of
// ``Opt::set_hyperparams``; ``check`` takes that same tuple and runs the
// class's rules.
template <class Opt, class Read, class Check>
void def_hyperparams(py::class_<Opt, Optimizer>& cls, Read read, Check check) {
    cls.def(
        "set_hyperparams",
        [read](Opt& self, const py::dict& values) {
            std::apply([&self](auto... args) { self.set_hyperparams(args...); }, read(values));
        },
        py::arg("values"),
        "Replace the hyper-parameters between steps, from a dict keyed by "
        "param_groups name.  Checked as the constructor checks them; nothing "
        "is applied when a value is rejected.");
    cls.def_static(
        "check_hyperparams",
        [read, check](const py::dict& values) { std::apply(check, read(values)); },
        py::arg("values"),
        "Raise InvalidArgument when the hyper-parameters break the constructor's rules.");
}

}  // namespace

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
    py::class_<SGD, Optimizer> sgd(m, "SGD");
    sgd.def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                     bool>(),
            py::arg("params"), py::arg("lr"), py::arg("momentum") = 0.0, py::arg("dampening") = 0.0,
            py::arg("weight_decay") = 0.0, py::arg("nesterov") = false,
            "SGD with momentum, Nesterov, and L2 weight decay.")
        .def_property_readonly("momentum", &SGD::momentum)
        .def_property_readonly("weight_decay", &SGD::weight_decay);
    def_hyperparams(
        sgd,
        [](const py::dict& v) {
            const HyperparamDict h("SGD", v,
                                   {"lr", "momentum", "dampening", "weight_decay", "nesterov"});
            return std::make_tuple(h.num("lr"), h.num("momentum"), h.num("dampening"),
                                   h.num("weight_decay"), h.flag("nesterov"));
        },
        &SGD::check_hyperparams);

    // The Adam family's shared table: AMSGrad is a flag without a rule.
    const auto adam_args = [](const char* op) {
        return [op](const py::dict& v) {
            const HyperparamDict h(op, v,
                                   {"lr", "beta1", "beta2", "eps", "weight_decay", "amsgrad"});
            return std::make_tuple(h.num("lr"), h.num("beta1"), h.num("beta2"), h.num("eps"),
                                   h.num("weight_decay"), h.flag("amsgrad"));
        };
    };
    // NAdam, RAdam and Adamax take the same names without ``amsgrad``.
    const auto moment_args = [](const char* op) {
        return [op](const py::dict& v) {
            const HyperparamDict h(op, v, {"lr", "beta1", "beta2", "eps", "weight_decay"});
            return std::make_tuple(h.num("lr"), h.num("beta1"), h.num("beta2"), h.num("eps"),
                                   h.num("weight_decay"));
        };
    };

    // Adam (Kingma & Ba 2014).  amsgrad=True enables the AMSGrad variant which
    // uses the maximum of past squared gradients for a tighter convergence bound.
    py::class_<Adam, Optimizer> adam(m, "Adam");
    adam.def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      double, bool>(),
             py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("amsgrad") = false, "Adam (Kingma & Ba 2014) with optional L2 weight decay.")
        .def_property_readonly("beta1", &Adam::beta1)
        .def_property_readonly("beta2", &Adam::beta2)
        .def_property_readonly("eps", &Adam::eps);
    def_hyperparams(adam, adam_args("Adam"),
                    [](double lr, double beta1, double beta2, double eps, double wd, bool) {
                        Adam::check_hyperparams(lr, beta1, beta2, eps, wd);
                    });

    // AdamW: decoupled weight decay applied directly to parameters rather than
    // folded into the gradient (Loshchilov & Hutter 2017).
    py::class_<AdamW, Optimizer> adamw(m, "AdamW");
    adamw.def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                       double, bool>(),
              py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
              py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 1e-2,
              py::arg("amsgrad") = false,
              "AdamW (decoupled weight decay, Loshchilov & Hutter 2017).");
    def_hyperparams(adamw, adam_args("AdamW"),
                    [](double lr, double beta1, double beta2, double eps, double wd, bool) {
                        AdamW::check_hyperparams(lr, beta1, beta2, eps, wd);
                    });

    // ASGD: decaying step size, decayed parameter, running average from t0.
    py::class_<ASGD, Optimizer> asgd(m, "ASGD");
    asgd.def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      double>(),
             py::arg("params"), py::arg("lr") = 1e-2, py::arg("lambd") = 1e-4,
             py::arg("alpha") = 0.75, py::arg("t0") = 1e6, py::arg("weight_decay") = 0.0,
             "Averaged SGD.");
    def_hyperparams(
        asgd,
        [](const py::dict& v) {
            const HyperparamDict h("ASGD", v, {"lr", "lambd", "alpha", "t0", "weight_decay"});
            return std::make_tuple(h.num("lr"), h.num("lambd"), h.num("alpha"), h.num("t0"),
                                   h.num("weight_decay"));
        },
        [](double lr, double, double, double, double wd) { ASGD::check_hyperparams(lr, wd); });

    py::class_<NAdam, Optimizer> nadam(m, "NAdam");
    nadam.def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                       double, double>(),
              py::arg("params"), py::arg("lr") = 2e-3, py::arg("beta1") = 0.9,
              py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
              py::arg("momentum_decay") = NAdam::kDefaultMomentumDecay,
              "Nesterov-accelerated Adam.");
    // The wrapper's groups have no ``momentum_decay``: it builds every NAdam
    // with the default, which is what its groups are checked with.
    def_hyperparams(nadam, moment_args("NAdam"),
                    [](double lr, double beta1, double beta2, double eps, double wd) {
                        NAdam::check_hyperparams(lr, beta1, beta2, eps, wd,
                                                 NAdam::kDefaultMomentumDecay);
                    });

    py::class_<RAdam, Optimizer> radam(m, "RAdam");
    radam.def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                       double>(),
              py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
              py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
              "Rectified Adam (Liu et al. 2020).");
    def_hyperparams(radam, moment_args("RAdam"), &RAdam::check_hyperparams);

    py::class_<RMSprop, Optimizer> rmsprop(m, "RMSprop");
    rmsprop.def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                         double, bool>(),
                py::arg("params"), py::arg("lr") = 1e-2, py::arg("alpha") = 0.99,
                py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0, py::arg("momentum") = 0.0,
                py::arg("centered") = false,
                "RMSprop with optional centered variance and momentum.");
    def_hyperparams(
        rmsprop,
        [](const py::dict& v) {
            const HyperparamDict h("RMSprop", v,
                                   {"lr", "alpha", "eps", "weight_decay", "momentum", "centered"});
            return std::make_tuple(h.num("lr"), h.num("alpha"), h.num("eps"), h.num("weight_decay"),
                                   h.num("momentum"), h.flag("centered"));
        },
        [](double lr, double alpha, double eps, double wd, double momentum, bool) {
            RMSprop::check_hyperparams(lr, alpha, eps, wd, momentum);
        });

    py::class_<Rprop, Optimizer> rprop(m, "Rprop");
    rprop.def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                       double>(),
              py::arg("params"), py::arg("lr") = 1e-2, py::arg("eta_minus") = 0.5,
              py::arg("eta_plus") = 1.2, py::arg("step_min") = 1e-6, py::arg("step_max") = 50.0,
              "Resilient backprop (Rprop).");
    def_hyperparams(
        rprop,
        [](const py::dict& v) {
            const HyperparamDict h("Rprop", v,
                                   {"lr", "eta_minus", "eta_plus", "step_min", "step_max"});
            return std::make_tuple(h.num("lr"), h.num("eta_minus"), h.num("eta_plus"),
                                   h.num("step_min"), h.num("step_max"));
        },
        [](double lr, double eta_minus, double eta_plus, double, double) {
            Rprop::check_hyperparams(lr, eta_minus, eta_plus);
        });

    py::class_<Adagrad, Optimizer> adagrad(m, "Adagrad");
    adagrad.def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                         double>(),
                py::arg("params"), py::arg("lr") = 1e-2, py::arg("lr_decay") = 0.0,
                py::arg("weight_decay") = 0.0, py::arg("initial_accumulator_value") = 0.0,
                py::arg("eps") = 1e-10, "Adagrad: per-parameter accumulator of squared grads.");
    def_hyperparams(
        adagrad,
        [](const py::dict& v) {
            const HyperparamDict h(
                "Adagrad", v,
                {"lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"});
            return std::make_tuple(h.num("lr"), h.num("lr_decay"), h.num("weight_decay"),
                                   h.num("initial_accumulator_value"), h.num("eps"));
        },
        &Adagrad::check_hyperparams);

    py::class_<Adadelta, Optimizer> adadelta(m, "Adadelta");
    adadelta.def(
        py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double>(),
        py::arg("params"), py::arg("lr") = 1.0, py::arg("rho") = 0.9, py::arg("eps") = 1e-6,
        py::arg("weight_decay") = 0.0, "Adadelta: parameter-free adaptive LR (Zeiler 2012).");
    def_hyperparams(
        adadelta,
        [](const py::dict& v) {
            const HyperparamDict h("Adadelta", v, {"lr", "rho", "eps", "weight_decay"});
            return std::make_tuple(h.num("lr"), h.num("rho"), h.num("eps"), h.num("weight_decay"));
        },
        &Adadelta::check_hyperparams);

    py::class_<Adamax, Optimizer> adamax(m, "Adamax");
    adamax.def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                        double>(),
               py::arg("params"), py::arg("lr") = 2e-3, py::arg("beta1") = 0.9,
               py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
               "Adamax: Adam with infinity norm.");
    def_hyperparams(adamax, moment_args("Adamax"), &Adamax::check_hyperparams);

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
