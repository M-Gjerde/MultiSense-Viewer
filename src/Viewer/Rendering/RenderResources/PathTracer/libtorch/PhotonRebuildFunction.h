//
// Created by magnus-desktop on 1/25/25.
//
#ifndef PHOTONREBUILDFUNCTION_H
#define PHOTONREBUILDFUNCTION_H

#include <torch/torch.h>
#include <vector>

#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer.h"


namespace VkRender::PathTracer {
    /**
     * A custom autograd Function that calls your path tracer in forward(),
     * and defines a custom backward() pass.
     */
    class PhotonRebuildFunction : public torch::autograd::Function<PhotonRebuildFunction> {
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
            // Non-tensor arguments can also be captured by custom means:
            IterationInfo& settings,
            PhotonTracer* pathTracer,
            torch::Tensor positions,
            torch::Tensor scales,
            torch::Tensor normals,
            torch::Tensor emissions,
            torch::Tensor colors,
            torch::Tensor specular,
            torch::Tensor diffuse
        );
        /**
         * backward()
         * @param ctx  Autograd context that has the saved_for_backward() data
         * @param grad_output The gradient of some scalar loss wrt. the output from forward()
         *
         * Return: A list of gradients w.r.t. each forward input that requires grad.
         */
        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
    };

}
#endif //PHOTONREBUILDFUNCTION_H
