//
// Created by magnus on 11/21/24.
//

#ifndef MULTISENSE_VIEWER_RENDERGAUSSIAN_H
#define MULTISENSE_VIEWER_RENDERGAUSSIAN_H

#include <torch/torch.h>
#include "Viewer/Rendering/RenderResources/DifferentiableRenderer/RenderGaussianFunction.h"


namespace VkRender::DR{
    class RenderGaussian : public torch::nn::Module {
    public:
        RenderGaussian(int64_t image_size);

        torch::Tensor forward(torch::Tensor center, torch::Tensor variance);

        torch::Tensor getLastRenderedImage();
        uint32_t imageSize() const{
            return m_imageSize;
        }

    private:
        int64_t m_imageSize;
        torch::Tensor xv;
        torch::Tensor yv;
        torch::Tensor m_lastRenderedImage;  // Store the last rendered image
    };

}


#endif //MULTISENSE_VIEWER_RENDERGAUSSIAN_H
