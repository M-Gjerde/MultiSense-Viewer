//
// Created by magnus on 11/21/24.
//

#include "RenderGaussianFunction.h"


namespace VkRender::DR{
    torch::Tensor RenderGaussianFunction::forward(torch::autograd::AutogradContext *ctx,
                                                  torch::Tensor center,
                                                  torch::Tensor variance,
                                                  torch::Tensor xv,
                                                  torch::Tensor yv) {


        // Compute the difference between each pixel and the center
        torch::Tensor dx = xv - center[0];
        torch::Tensor dy = yv - center[1];

        // Adjust distance with per-axis variances
        torch::Tensor d_squared = (dx.pow(2) / variance[0]) + (dy.pow(2) / variance[1]);

        // Compute Gaussian function
        torch::Tensor img = torch::exp(-0.5 * d_squared);

        // Save tensors for backward pass
        ctx->save_for_backward({center, variance, dx, dy, d_squared, img});

        return img;
    }

    torch::autograd::tensor_list RenderGaussianFunction::backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {
        // Retrieve saved tensors
        auto saved = ctx->get_saved_variables();
        torch::Tensor center = saved[0];
        torch::Tensor variance = saved[1];
        torch::Tensor dx = saved[2];
        torch::Tensor dy = saved[3];
        torch::Tensor d_squared = saved[4];
        torch::Tensor img = saved[5];

        // Get grad_output
        torch::Tensor grad_output = grad_outputs[0];

        // Compute derivatives with respect to dx and dy
        torch::Tensor dd_dx = dx / variance[0] * img;
        torch::Tensor dd_dy = dy / variance[1] * img;

        // Optionally, compute gradient magnitude (if needed for debugging or visualization)
        torch::Tensor gradient_magnitude = torch::sqrt(dd_dx.pow(2) + dd_dy.pow(2));
        // If you have a method to capture the gradient image, you can call it here
        // ctx->self.capture_gradient_image(gradient_magnitude);

        // Compute derivatives with respect to variances
        torch::Tensor dd_d_variance_x = 0.5 * dx.pow(2) / variance[0].pow(2) * img;
        torch::Tensor dd_d_variance_y = 0.5 * dy.pow(2) / variance[1].pow(2) * img;

        // Gradients of the Gaussian function with respect to dx, dy, and variances
        torch::Tensor grad_dx = grad_output * dd_dx;
        torch::Tensor grad_dy = grad_output * dd_dy;
        torch::Tensor grad_variance_x = grad_output * dd_d_variance_x;
        torch::Tensor grad_variance_y = grad_output * dd_d_variance_y;

        // Gradients with respect to center and variance
        torch::Tensor grad_center = torch::stack({torch::sum(grad_dx), torch::sum(grad_dy)});
        torch::Tensor grad_variance = torch::stack({torch::sum(grad_variance_x), torch::sum(grad_variance_y)});

        // Return gradients with respect to inputs: center, variance, xv, yv
        return {
                grad_center,     // Gradient with respect to center
                grad_variance,   // Gradient with respect to variance
                torch::Tensor(), // No gradient with respect to xv
                torch::Tensor()  // No gradient with respect to yv
        };
    }
}