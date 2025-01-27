//
// Created by magnus on 1/27/25.
//

#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildFunction.h"
#include "stb_image_write.h"

namespace VkRender::PathTracer {
    static void save_gradient_to_png(torch::Tensor gradient, const std::string& filename) {
        // Ensure the tensor is on CPU and in float32
        gradient = gradient.detach().cpu().to(torch::kFloat32);

        // Normalize to [0, 255]
        auto min = gradient.min().item<float>();
        auto max = gradient.max().item<float>();
        auto normalized = (gradient - min) / (max - min) * 255.0;

        // Convert to uint8
        auto uint8_tensor = normalized.to(torch::kUInt8);

        // Get raw pointer
        uint8_t* data = uint8_tensor.data_ptr<uint8_t>();

        // Get dimensions
        int width = gradient.size(1);
        int height = gradient.size(0);
        // Save as PNG using stb_image_write
        stbi_write_png(filename.c_str(), width, height, 1, data, width);
    }

    // Save as


    torch::Tensor PhotonRebuildFunction::forward(torch::autograd::AutogradContext* ctx,
                                                 PhotonTracer::Settings& settings, PhotonTracer* pathTracer,
                                                 torch::Tensor positions, torch::Tensor scales,
                                                 torch::Tensor normals, torch::Tensor emissions,
                                                 torch::Tensor colors,
                                                 torch::Tensor specular,
                                                 torch::Tensor diffuse) {
        // =================
        // 1) Save for backward any Tensors or scalar values you need
        //    to compute derivatives later. For example:
        ctx->save_for_backward({positions, scales, normals, emissions, colors, specular, diffuse});
        // If you have non-tensor data you want in backward(), you can store
        // them as attributes:
        ctx->saved_data["someScalar"] = 42.0; // example
        // or store the pointer as a raw pointer or shared pointer if you prefer
        // (but be careful with lifetimes).

        // =================
        // 2) Run your path tracer code that renders an image.

        // Example pseudo-code:

        pathTracer->update(settings);

        // Suppose the path tracer writes out to pathTracer->m_imageMemory,
        // with shape [height * width] or [height * width * channels].
        // We'll build a Torch tensor from that raw memory.

        // For illustration:
        int64_t height = pathTracer->m_height;
        int64_t width = pathTracer->m_width;
        float* rawImage = pathTracer->getImage();
        // e.g. a float[height * width]  (gray) or float[height * width * 3]

        // Wrap it in a Torch tensor.
        // Note from_blob does not take ownership, so we typically clone().
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto output = torch::from_blob(rawImage, {height, width}, options).clone();

        // Return the rendered image
        return output;
    }

    torch::autograd::tensor_list PhotonRebuildFunction::backward(torch::autograd::AutogradContext* ctx,
                                                                 torch::autograd::tensor_list grad_outputs) {
        // Usually, the forward returned 1 tensor => grad_outputs.size() == 1
        // grad_outputs[0] is d(L)/d(output).

        auto dLoss_dRenderedImage = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto positions = saved[0];
        auto scales = saved[1];
        auto normals = saved[2];

        // We'll do a trivial zero gradient for demonstration
        auto grad_positions = torch::zeros_like(positions);
        auto grad_scales = torch::zeros_like(scales);
        auto grad_normals = torch::zeros_like(normals);
        //grad_positions.index_put_({at::indexing::Slice(), 1}, 1.0);
        save_gradient_to_png(dLoss_dRenderedImage, "gradient.png");

        auto gradPosA = grad_positions.accessor<float, 2>(); // [i, 3]
        auto posA = positions.accessor<float, 2>(); // [i, 3]

        uint32_t numLights = 1;
        uint32_t height = 600;
        uint32_t width = 960;
        // We'll accumulate dLoss/dPos[i,0..2] in a float3
        float dLdPx = 0.f, dLdPy = 0.f, dLdPz = 0.f;
        for (int i = 0; i < numLights; i++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    /*
                    float dist = savedFactorsA[i][y][x][0]; // distance
                    float cosTheta = savedFactorsA[i][y][x][1]; // cos term

                    // dLoss/dImage at pixel (y,x):
                    float dL_dI = dLoss_dImageA[y][x];

                    // partial derivative of I_pred w.r.t. the light's position
                    // Insert your derived expression:
                    //   I_pred = q*(dist^2)/(cosTheta)
                    //   let dI_dPos = (analytic expression)
                    //
                    // Here we do just a fake placeholder:
                    //   dI/dPos = d( q * dist^2 / cosTheta )/dPos
                    //   We also need direction to figure sign, etc.

                    // E.g. from your math:
                    //    dist = ||(F - E_o)||
                    //    cosTheta = dot( (F - E_o)/dist, F_n )
                    // so you'll do something like:
                    //
                    //   float dI_dDist     = q*( 2*dist/cosTheta )
                    //   float dI_dCosTheta = q*( -dist^2 / cosTheta^2 )
                    //   // then chain rule for dist wrt pos, cosTheta wrt pos, etc.
                    //
                    // For demonstration, suppose we have the partial derivatives wrt. each axis
                    // precomputed as dI_dx, dI_dy, dI_dz:
                    //
                    float dI_dx = 0.f;
                    float dI_dy = 0.f;
                    float dI_dz = 0.f;

                    // ... fill in your formula, e.g.:
                    // dI_dx = ... ( big expression using positions[i], dist, cosTheta, etc.)
                    // dI_dy = ...
                    // dI_dz = ...

                    // Combine with chain rule from dLoss/dImage:
                    dLdPx += dL_dI * dI_dx;
                    dLdPy += dL_dI * dI_dy;
                    dLdPz += dL_dI * dI_dz;
                    */
                }
            }
            // Store result in grad_positions
            gradPosA[i][0] = dLdPx;
            gradPosA[i][1] = dLdPy;
            gradPosA[i][2] = dLdPz;
        }

        // Return them in the same order as forward inputs
        return {
            torch::Tensor(), // wrt settings (not a Tensor)
            torch::Tensor(), // wrt pathTracer (not a Tensor)
            grad_positions, // wrt positions
            grad_scales, // wrt scales
            grad_normals, // wrt normals
            torch::Tensor(), // emission
            torch::Tensor(), // colors
            torch::Tensor(), // specular
            torch::Tensor(), // diffuse
        };
    }
}
