#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    if (batch == 0) {
        throw std::invalid_argument("batch must be positive");
    }
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    std::vector<float> logits_grad(batch * k);
    std::vector<float> theta_grad(n * k);

    for (size_t start = 0; start < m; start += batch) {
        const size_t batch_size = std::min(batch, m - start);

        // Compute stable softmax probabilities and turn them into dL/dZ.
        for (size_t example = 0; example < batch_size; ++example) {
            const size_t data_row = start + example;
            float *logits_row = logits_grad.data() + example * k;
            std::fill(logits_row, logits_row + k, 0.0f);

            // Keep the class dimension innermost so theta is read
            // contiguously in row-major order.
            for (size_t feature = 0; feature < n; ++feature) {
                const float input = X[data_row * n + feature];
                const float *theta_row = theta + feature * k;
                for (size_t class_id = 0; class_id < k; ++class_id) {
                    logits_row[class_id] += input * theta_row[class_id];
                }
            }

            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t class_id = 0; class_id < k; ++class_id) {
                max_logit = std::max(max_logit, logits_row[class_id]);
            }

            float normalizer = 0.0f;
            for (size_t class_id = 0; class_id < k; ++class_id) {
                float probability = std::exp(logits_row[class_id] - max_logit);
                logits_row[class_id] = probability;
                normalizer += probability;
            }
            for (size_t class_id = 0; class_id < k; ++class_id) {
                logits_row[class_id] /= normalizer;
            }
            logits_row[y[data_row]] -= 1.0f;
        }

        // theta <- theta - lr * X_batch^T @ dL/dZ / batch_size.
        std::fill(theta_grad.begin(), theta_grad.end(), 0.0f);
        for (size_t example = 0; example < batch_size; ++example) {
            const float *input_row = X + (start + example) * n;
            const float *logits_row = logits_grad.data() + example * k;
            for (size_t feature = 0; feature < n; ++feature) {
                const float input = input_row[feature];
                float *gradient_row = theta_grad.data() + feature * k;
                for (size_t class_id = 0; class_id < k; ++class_id) {
                    gradient_row[class_id] += input * logits_row[class_id];
                }
            }
        }

        const float scale = lr / static_cast<float>(batch_size);
        for (size_t parameter = 0; parameter < n * k; ++parameter) {
            theta[parameter] -= scale * theta_grad[parameter];
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        if (batch <= 0) {
            throw std::invalid_argument("batch must be positive");
        }
        if (X.ndim() != 2 || y.ndim() != 1 || theta.ndim() != 2) {
            throw std::invalid_argument("X and theta must be 2D; y must be 1D");
        }
        if (X.shape(0) != y.shape(0) || X.shape(1) != theta.shape(0)) {
            throw std::invalid_argument("incompatible X, y, and theta shapes");
        }
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
