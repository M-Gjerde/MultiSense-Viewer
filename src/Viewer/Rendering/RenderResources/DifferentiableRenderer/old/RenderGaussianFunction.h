//
// Created by magnus on 11/21/24.
//

#ifndef MULTISENSE_VIEWER_RENDERGAUSSIANFUNCTION_H
#define MULTISENSE_VIEWER_RENDERGAUSSIANFUNCTION_H

#include <torch/torch.h>
#include <iostream>


namespace VkRender::DR{
    // Custom autograd function
    class RenderGaussianFunction : public torch::autograd::Function<RenderGaussianFunction> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                     torch::Tensor center,
                                     torch::Tensor variance,
                                     torch::Tensor xv,
                                     torch::Tensor yv);

        static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
                                           torch::autograd::tensor_list grad_outputs);
    };

}



#endif //MULTISENSE_VIEWER_RENDERGAUSSIANFUNCTION_H
