//
// Created by magnus on 1/27/25.
//

#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildFunction.h"
#include "stb_image_write.h"
#include <random>
#include <glm/gtx/quaternion.hpp>

namespace VkRender::PathTracer {
    static void save_gradient_to_png(torch::Tensor gradient, const std::filesystem::path& filename) {
        std::filesystem::path dir = filename.parent_path();

        // Create directory if it doesn't exist
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }


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


        std::filesystem::path filenamePath = filename;
        std::ofstream file(filenamePath.replace_extension(".pfm"), std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filenamePath.replace_extension(".pfm").string());
        }
        // Write the PFM header
        // "PF" indicates a color image. Use "Pf" for grayscale.
        file << "Pf\n" << width << " " << height << "\n-1.0\n";

        // PFM expects the data in binary format, row by row from top to bottom
        // Assuming your m_imageMemory is in RGBA format with floats

        // Allocate a temporary buffer for RGB data
        std::vector<float> rgbData(width * height);

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t pixelIndex = (y * width + x);
                uint32_t rgbIndex = (y * width + x);
                rgbData[rgbIndex + 0] = gradient.data_ptr<float>()[pixelIndex] * 255; // R
            }
        }

        // Write the RGB float data
        file.write(reinterpret_cast<const char*>(rgbData.data()), rgbData.size() * sizeof(float));

        if (!file) {
            throw std::runtime_error("Failed to write PFM data to file: " + filenamePath.replace_extension(".pfm").string());
        }

        file.close();
    }


    void printProgressBar(float progress) {
        const int barWidth = 50; // Width of the progress bar
        std::cout << "\r["; // Carriage return to overwrite the line
        int pos = static_cast<int>(barWidth * progress);
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(2) << (progress * 100.0) << "%";
        std::cout.flush();
    }


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
        ctx->saved_data["pathTracer"] = reinterpret_cast<int64_t>(pathTracer);

        // If you have non-tensor data you want in backward(), you can store
        // them as attributes:
        ctx->saved_data["settings"] = reinterpret_cast<int64_t>(&settings); // example
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
        auto emissions = saved[3];
        auto colors = saved[4];
        auto diffuse = saved[5];
        auto specular = saved[6];
        // Retrieve the path tracer pointer
        auto pathTracerRaw = ctx->saved_data["pathTracer"].toInt();
        PhotonTracer* pathTracer = reinterpret_cast<PhotonTracer*>(pathTracerRaw);
        // Retrieve the path tracer pointer
        auto settingsPtr = ctx->saved_data["settings"].toInt();
        PhotonTracer::Settings* settings = reinterpret_cast<PhotonTracer::Settings*>(settingsPtr);

        // We'll do a trivial zero gradient for demonstration
        save_gradient_to_png(dLoss_dRenderedImage,"gradients/gradient_" + std::to_string(pathTracer->m_renderInformation->frameID) + ".png");


        pathTracer->m_backwardInfo.gradientImage = dLoss_dRenderedImage.data_ptr<float>();

        auto gradients = pathTracer->backward(*settings);
        glm::vec3 grad = *gradients.sumGradients;

        float grad_x = grad.x;
        float grad_y = grad.y;
        float grad_z = grad.z;

        auto grad_positions = torch::zeros_like(positions);
        auto gradPosA = grad_positions.accessor<float, 2>();
        for (int i = 0; i < grad_positions.size(0); ++i) {
            gradPosA[i][0] = grad_x;
            gradPosA[i][1] = grad_y;
            gradPosA[i][2] = grad_z;
        }

        // Return them in the same order as forward inputs
        return {
            torch::Tensor(), // wrt settings (not a Tensor)
            torch::Tensor(), // wrt pathTracer (not a Tensor)
            grad_positions, // wrt positions
            torch::Tensor(), // wrt scales
            torch::Tensor(), // wrt normals
            torch::Tensor(), // emission
            torch::Tensor(), // colors
            torch::Tensor(), // specular
            torch::Tensor(), // diffuse
        };
    }
}
