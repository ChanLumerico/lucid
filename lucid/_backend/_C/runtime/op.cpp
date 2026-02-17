#include "op.hpp"

#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>
#include <vector>

namespace lucid::backend::runtime {

    namespace detail {

        inline py::object call_object(
            const py::handle& callable,
            const py::tuple& args,
            const py::dict& kwargs = py::dict()
        ) {
            PyObject* out = PyObject_Call(callable.ptr(), args.ptr(), kwargs.ptr());
            if (out == nullptr) {
                throw py::error_already_set();
            }
            return py::reinterpret_steal<py::object>(out);
        }

        inline py::module_ lucid_module() {
            return py::module_::import("lucid");
        }

        inline py::module_ weakref_module() {
            return py::module_::import("weakref");
        }

        inline py::module_ core_module() {
            return py::module_::import("lucid._backend.core");
        }

        inline py::object tensor_like_protocol() {
            return py::module_::import("lucid.types").attr("_TensorLike");
        }

        inline py::object builtins_bool() {
            return py::module_::import("builtins").attr("bool");
        }

        inline py::tuple tuple_slice(const py::tuple& src, std::size_t start, std::size_t stop) {
            if (start > stop || stop > static_cast<std::size_t>(src.size())) {
                throw py::value_error("Invalid tuple slice range.");
            }

            py::tuple out(stop - start);
            for (std::size_t i = start; i < stop; ++i) {
                out[i - start] = src[i];
            }
            return out;
        }

        inline py::tuple tuple_from_objects(const std::vector<py::object>& objs) {
            py::tuple out(objs.size());
            for (std::size_t i = 0; i < objs.size(); ++i) {
                out[i] = objs[i];
            }
            return out;
        }

        inline py::tuple concat_tuples(const py::tuple& left, const py::tuple& right) {
            py::tuple out(static_cast<std::size_t>(left.size() + right.size()));
            std::size_t cursor = 0;

            for (const auto& v : left) out[cursor++] = v;
            for (const auto& v : right) out[cursor++] = v;

            return out;
        }

        inline bool is_tensor_like(const py::handle& obj) {
            return py::isinstance(obj, tensor_like_protocol());
        }

        inline py::object make_backward_operation(
            const py::object& op_self,
            const py::object& grad_func,
            const py::tuple& tensor_refs,
            const py::tuple& versions,
            const std::string& device
        ) {
            py::object weakref_ref = weakref_module().attr("ref");
            py::dict kwargs;

            kwargs["forward_op_ref"] = weakref_ref(op_self);
            kwargs["grad_func"] = grad_func;
            kwargs["tensor_refs"] = tensor_refs;
            kwargs["versions"] = versions;
            kwargs["device"] = py::str(device);

            return call_object(core_module().attr("BackwardOperation"), py::tuple(), kwargs);
        }

        inline std::string device_mismatch_message(
            const py::object& tensor,
            const std::string& device,
            const py::object& op_self
        ) {
            std::ostringstream oss;
            oss << py::str(tensor.attr("device")).cast<std::string>();
            oss << " tensor of shape ";
            oss << py::str(tensor.attr("shape")).cast<std::string>();
            oss << " passed for " << device;
            oss << " operation('"
                << py::str(py::type::of(op_self).attr("__name__")).cast<std::string>()
                << "').";
            return oss.str();
        }

    }

    py::object FuncOpExecutor::execute(
        const py::object& op_self,
        const py::object& forward_func,
        const py::tuple& args,
        const py::dict& kwargs,
        const FuncOpSpec& spec
    ) {
        py::module_ lucid = detail::lucid_module();
        py::object weakref_ref = detail::weakref_module().attr("ref");

        py::tuple tensor_args = args;
        if (spec.n_in.has_value()) {
            const std::size_t n_in = *spec.n_in;
            if (static_cast<std::size_t>(args.size()) < n_in) {
                throw py::value_error(
                    "Expected at least " + std::to_string(n_in)
                    + " tensor arguments, got " + std::to_string(args.size())
                );
            }
            tensor_args = detail::tuple_slice(args, 0, n_in);
        }

        py::object dtype_hint = py::none();
        for (const auto& arg : tensor_args) {
            if (detail::is_tensor_like(arg)) {
                dtype_hint = py::reinterpret_borrow<py::object>(arg).attr("dtype");
                break;
            }
        }

        std::vector<py::object> tensors;
        tensors.reserve(static_cast<std::size_t>(tensor_args.size()));

        bool requires_grad = false;
        bool is_free = true;
        py::object inplace_target = py::none();

        for (const auto& arg : tensor_args) {
            py::dict check_kwargs;
            check_kwargs["device"] = py::str(spec.device);
            check_kwargs["dtype"] = dtype_hint;

            py::object tensor = detail::call_object(
                lucid.attr("_check_is_tensor"),
                py::make_tuple(arg),
                check_kwargs
            );
            tensors.push_back(tensor);

            requires_grad = requires_grad || tensor.attr("requires_grad").cast<bool>();
            if (tensor.attr("is_free").cast<bool>()) {
                tensor.attr("to")(py::str(spec.device));

            } else {
                is_free = false;
                const auto tensor_device = tensor.attr("device").cast<std::string>();
                if (tensor_device != spec.device) {
                    throw std::runtime_error(
                        detail::device_mismatch_message(tensor, spec.device, op_self)
                    );
                }
            }
        }

        if (op_self.attr("_inplace").cast<bool>()) {
            if (!spec.n_ret.has_value() || *spec.n_ret != 1) {
                throw py::value_error("inplace op must have a single return value.");
            }

            const std::size_t inplace_target_idx = (
                op_self.attr("_inplace_target").cast<std::size_t>()
            );
            if (inplace_target_idx >= tensors.size()) {
                throw py::value_error("inplace_target is out of range.");
            }

            py::object target = tensors[inplace_target_idx];
            const bool grad_enabled = lucid.attr("grad_enabled")().cast<bool>();

            if (
                grad_enabled
                && target.attr("requires_grad").cast<bool>()
                && target.attr("is_leaf").cast<bool>()
            ) {
                throw std::runtime_error(
                    "A leaf tensor with 'requires_grad=True' cannot be "
                    "subjected to inplace operations."
                );
            }
            inplace_target = target;

            py::object proxy = target.attr("new_tensor")();
            proxy.attr("_op") = target.attr("_op");
            proxy.attr("_prev") = py::list(target.attr("_prev"));
            proxy.attr("_backward_op") = target.attr("_backward_op");
            proxy.attr("_backward_hooks") = py::list(target.attr("_backward_hooks"));
            proxy.attr("grad") = py::none();
            proxy.attr("_version") = target.attr("_version");

            if (py::hasattr(target, "_is_free")) {
                proxy.attr("_is_free") = target.attr("_is_free");
            }
            if (py::hasattr(target, "_is_bool_tensor")) {
                proxy.attr("_is_bool_tensor") = target.attr("_is_bool_tensor");
            }

            tensors[inplace_target_idx] = proxy;
        }

        py::tuple new_args = detail::tuple_from_objects(tensors);
        if (spec.n_in.has_value()) {
            const py::tuple non_tensor_args = detail::tuple_slice(
                args, *spec.n_in, static_cast<std::size_t>(args.size())
            );
            new_args = detail::concat_tuples(new_args, non_tensor_args);
        }

        py::object func_return_pairs = detail::call_object(
            forward_func,
            detail::concat_tuples(py::make_tuple(op_self), new_args),
            kwargs
        );

        py::tuple tensor_refs(tensors.size());
        py::tuple versions(tensors.size());
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            tensor_refs[i] = weakref_ref(tensors[i]);
            versions[i] = tensors[i].attr("_version");
        }

        const bool grad_enabled = lucid.attr("grad_enabled")().cast<bool>();
        const bool flops_enabled = lucid.attr("flops_enabled")().cast<bool>();
        const bool track_graph = flops_enabled || (grad_enabled && requires_grad);

        if (flops_enabled) {
            py::object flops = detail::call_object(op_self.attr("__flops__"), new_args, kwargs);
            op_self.attr("flops") = flops;
        }

        std::size_t num_returns = 0;
        if (!spec.n_ret.has_value()) {
            if (!py::isinstance<py::tuple>(func_return_pairs)) {
                throw py::value_error(
                    std::string(py::str(forward_func.attr("__name__")))
                    + " should return multiple '_ReturnGradFuncPair'."
                );
            }
            num_returns = static_cast<std::size_t>(
                py::reinterpret_borrow<py::tuple>(func_return_pairs).size()
            );

        } else {
            num_returns = *spec.n_ret;
        }

        py::tuple pairs;
        if (num_returns == 1) {
            pairs = py::make_tuple(func_return_pairs);

        } else {
            if (op_self.attr("_inplace").cast<bool>()) {
                throw py::value_error("inplace op must have a single return value.");
            }
            pairs = py::reinterpret_borrow<py::tuple>(func_return_pairs);
        }

        if (op_self.attr("_inplace").cast<bool>()) {
            py::tuple first = py::reinterpret_borrow<py::tuple>(pairs[0]);
            py::object ret_value = py::reinterpret_borrow<py::object>(first[0]);
            py::object grad_func = py::reinterpret_borrow<py::object>(first[1]);

            if (inplace_target.is_none()) {
                throw std::runtime_error("Missing inplace target tensor.");
            }

            inplace_target.attr("data") = ret_value.attr("data");

            const py::object dtype_obj = (
                py::reinterpret_borrow<py::object>(ret_value.attr("dtype"))
            );
            const bool is_bool = (dtype_obj.ptr() == detail::builtins_bool().ptr());
            if (is_bool) {
                inplace_target.attr("_is_bool_tensor") = py::bool_(true);
                inplace_target.attr("dtype") = detail::builtins_bool();

            } else {
                inplace_target.attr("_is_bool_tensor") = py::bool_(false);
                inplace_target.attr("dtype") = ret_value.attr("dtype");
            }

            const auto version = inplace_target.attr("_version").cast<int>();
            inplace_target.attr("_version") = py::int_(version + 1);
            pairs = py::make_tuple(py::make_tuple(inplace_target, grad_func));
        }

        std::vector<py::object> results;
        results.reserve(pairs.size());

        for (const auto& pair_obj : pairs) {
            py::tuple pair = py::reinterpret_borrow<py::tuple>(pair_obj);
            py::object result = py::reinterpret_borrow<py::object>(pair[0]);
            py::object grad_func = py::reinterpret_borrow<py::object>(pair[1]);

            const bool result_requires_grad = requires_grad && spec.has_gradient && grad_enabled;
            result.attr("requires_grad") = py::bool_(result_requires_grad);
            result.attr("to")(py::str(spec.device));

            if (is_free) result.attr("free")();
            results.push_back(result);

            if (!track_graph) continue;

            result.attr("_op") = op_self;
            if (result_requires_grad || lucid.attr("flops_enabled")().cast<bool>()) {
                result.attr("_prev") = py::list(detail::tuple_from_objects(tensors));
                if (!result_requires_grad) continue;

                result.attr("_backward_op") = detail::make_backward_operation(
                    op_self, grad_func, tensor_refs, versions, spec.device
                );
            }
        }

        if (track_graph) {
            try {
                if (num_returns > 1) {
                    op_self.attr("result") = detail::tuple_from_objects(results);
                } else {
                    op_self.attr("result") = results[0];
                }
            } catch (const py::error_already_set& e) {
                if (!e.matches(PyExc_Exception)) {
                    throw;
                }
            }

        } else {
            try {
                op_self.attr("clear")();
            } catch (const py::error_already_set& e) {
                if (!e.matches(PyExc_Exception)) {
                    throw;
                }
                try {
                    op_self.attr("result") = py::none();
                } catch (const py::error_already_set& e2) {
                    if (!e2.matches(PyExc_Exception)) {
                        throw;
                    }
                }
            }
        }

        if (num_returns > 1) return detail::tuple_from_objects(results);
        return results[0];
    }
}
