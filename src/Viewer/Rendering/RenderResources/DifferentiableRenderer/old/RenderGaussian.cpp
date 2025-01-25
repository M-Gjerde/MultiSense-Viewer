//
// Created by magnus on 11/21/24.
//

#include "RenderGaussian.h"


namespace VkRender::DR {
    RenderGaussian::RenderGaussian(int64_t image_size_) : m_imageSize(image_size_) {
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);

        // Create 1D tensors for x and y coordinates
        torch::Tensor x = torch::arange(0, image_size_, options); // [0, 1, ..., image_size - 1]
        torch::Tensor y = torch::arange(0, image_size_, options);

        // Create 2D tensors using meshgrid
        auto meshgrid = torch::meshgrid({x, y}, /*indexing=*/"ij");
        xv = meshgrid[0];
        yv = meshgrid[1];

        xv = xv.to(torch::kCUDA);
        yv = yv.to(torch::kCUDA);

    }

    torch::Tensor RenderGaussian::forward(torch::Tensor center, torch::Tensor variance) {
        double min_variance = 1e-3;

        variance = torch::nn::functional::softplus(variance) + min_variance;

        m_lastRenderedImage = RenderGaussianFunction::apply(
                center,
                variance,
                xv,
                yv
        );

        return m_lastRenderedImage.clone();
    }

    torch::Tensor RenderGaussian::getLastRenderedImage() {
        return m_lastRenderedImage;
    }
}