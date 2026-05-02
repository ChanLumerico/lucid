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

void register_optim(py::module_& m) {
    py::class_<Optimizer>(m, "Optimizer")
        .def("step", &Optimizer::step)
        .def("zero_grad", &Optimizer::zero_grad)
        .def_property("lr", &Optimizer::lr, &Optimizer::set_lr)
        .def_property_readonly("num_params", &Optimizer::num_params)
        .def_property_readonly("state_dict_id", &Optimizer::state_dict_id);

    py::class_<SGD, Optimizer>(m, "SGD")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      bool>(),
             py::arg("params"), py::arg("lr"), py::arg("momentum") = 0.0,
             py::arg("dampening") = 0.0, py::arg("weight_decay") = 0.0, py::arg("nesterov") = false,
             "SGD with momentum, Nesterov, and L2 weight decay.")
        .def_property_readonly("momentum", &SGD::momentum)
        .def_property_readonly("weight_decay", &SGD::weight_decay);

    py::class_<Adam, Optimizer>(m, "Adam")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      double, bool>(),
             py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("amsgrad") = false, "Adam (Kingma & Ba 2014) with optional L2 weight decay.")
        .def_property_readonly("beta1", &Adam::beta1)
        .def_property_readonly("beta2", &Adam::beta2)
        .def_property_readonly("eps", &Adam::eps);

    py::class_<AdamW, Optimizer>(m, "AdamW")
        .def(py::init<std::vector<std::shared_ptr<TensorImpl>>, double, double, double, double,
                      double>(),
             py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 1e-2,
             "AdamW (decoupled weight decay, Loshchilov & Hutter 2017).");

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

    py::class_<LRScheduler>(m, "LRScheduler")
        .def("step", &LRScheduler::step)
        .def("set_epoch", &LRScheduler::set_epoch, py::arg("epoch"))
        .def_property_readonly("epoch", &LRScheduler::epoch);

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

    py::class_<LambdaLR, LRScheduler>(m, "LambdaLR")
        .def(py::init<Optimizer&, std::function<double(std::int64_t)>>(), py::arg("optimizer"),
             py::arg("lr_lambda"), py::keep_alive<1, 2>(),
             "Multiply base LR by lr_lambda(epoch). Lambda is a Python callable.");

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

    py::class_<NoamScheduler, LRScheduler>(m, "NoamScheduler")
        .def(py::init<Optimizer&, std::int64_t, std::int64_t, double>(), py::arg("optimizer"),
             py::arg("model_size"), py::arg("warmup_steps"), py::arg("factor") = 1.0,
             py::keep_alive<1, 2>(), "Transformer-style warmup-then-decay schedule.");

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
