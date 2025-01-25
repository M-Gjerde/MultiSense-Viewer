//
// Created by magnus-desktop on 1/25/25.
//
#ifndef PHOTONREBUILDFUNCTION_H
#define PHOTONREBUILDFUNCTION_H

#include <torch/torch.h>
#include <vector>
#include <memory>

#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer.h"
#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayerUI.h"
#include "Viewer/Scenes/Scene.h"

namespace VkRender::PathTracer
{
    /**
     * A custom autograd Function that calls your path tracer in forward(),
     * and defines a custom backward() pass.
     */
    class PhotonRebuildFunction : public torch::autograd::Function<PhotonRebuildFunction>
    {
    public:
        /**
         * forward()
         * @param ctx  Autograd context used to save variables for backward
         * @param uiLayer - Some parameter that configures the path tracer
         * @param scenePtr - A pointer to your Scene
         * @param someTensorParam - Example of a torch::Tensor parameter
         *                          that might influence the path tracer
         *
         * Return: A single torch::Tensor with the rendered image.
         */
        static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            // You can pass any Torch Tensors or scalars you want:
            torch::Tensor someTensorParam,
            // Non-tensor arguments can also be captured by custom means:
            PhotonTracer::Settings& settings,
            std::shared_ptr<Scene> scenePtr,
            PhotonTracer* pathTracer
        )
        {
            // =================
            // 1) Save for backward any Tensors or scalar values you need
            //    to compute derivatives later. For example:
            ctx->save_for_backward({someTensorParam});
            // If you have non-tensor data you want in backward(), you can store
            // them as attributes:
            ctx->saved_data["someScalar"] = 42.0; // example
            // or store the pointer as a raw pointer or shared pointer if you prefer
            // (but be careful with lifetimes).

            // =================
            // 2) Run your path tracer code that renders an image.

            // Example pseudo-code:
            pathTracer->update(settings, scenePtr);

            // Suppose the path tracer writes out to pathTracer->m_imageMemory,
            // with shape [height * width] or [height * width * channels].
            // We'll build a Torch tensor from that raw memory.

            // For illustration:
            int64_t height = pathTracer->m_height;
            int64_t width =  pathTracer->m_width;
            float* rawImage = pathTracer->getImage();
            // e.g. a float[height * width]  (gray) or float[height * width * 3]

            // Wrap it in a Torch tensor.
            // Note from_blob does not take ownership, so we typically clone().
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            auto output = torch::from_blob(rawImage, {height, width}, options).clone();

            // Return the rendered image
            return output;
        }

        /**
         * backward()
         * @param ctx  Autograd context that has the saved_for_backward() data
         * @param grad_output The gradient of some scalar loss wrt. the output from forward()
         *
         * Return: A list of gradients w.r.t. each forward input that requires grad.
         */
        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs)
        {
            // Usually, the forward returned 1 tensor => grad_outputs.size() == 1
            // grad_outputs[0] is d(L)/d(output).

            auto dLoss_dRenderedImage = grad_outputs[0];

            // 1) Recover the stuff we saved in forward():
            auto savedVars = ctx->get_saved_variables();
            // e.g. if forward saved 1 tensor in save_for_backward():
            torch::Tensor someTensorParam = savedVars[0];
            // If we stored a scalar in saved_data:
            double someScalar = ctx->saved_data["someScalar"].toDouble();

            // 2) YOUR custom gradient logic
            // For instance, you might want to compute dLoss/d(someTensorParam).
            // If that parameter influences the path tracer, you can define how.

            // For demonstration: we create a dummy gradient for someTensorParam
            // with the same shape as someTensorParam, filled with zeros:
            auto grad_someTensorParam = torch::zeros_like(someTensorParam);

            // Possibly do real math here: e.g.
            // grad_someTensorParam = ...
            // based on dLoss_dRenderedImage and how your tracer depends on someTensorParam.

            // 3) Return as many gradient Tensors as there were Tensors in forward's signature.
            // In our forward, we had 1 Tensor param: (someTensorParam).
            // The other arguments (uiLayer, scenePtr, pathTracer) are not Tensors, so we return
            // a placeholder (torch::Tensor()) for each of those if they appear in the signature.
            return {
                grad_someTensorParam, // gradient wrt someTensorParam
                torch::Tensor(), // gradient wrt uiLayer (not a Tensor)
                torch::Tensor(), // gradient wrt scenePtr (not a Tensor)
                torch::Tensor() // gradient wrt pathTracer (not a Tensor)
            };
        }
    };
}
#endif //PHOTONREBUILDFUNCTION_H
