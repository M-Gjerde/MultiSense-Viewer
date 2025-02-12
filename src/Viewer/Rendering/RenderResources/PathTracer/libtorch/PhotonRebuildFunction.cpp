//
// Created by magnus on 1/27/25.
//

#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildFunction.h"
#include "stb_image_write.h"
#include <random>
#include <glm/gtx/quaternion.hpp>
#include <OpenImageDenoise/oidn.hpp>

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

    static void denoiseImage(float* singleChannelImage, uint32_t width, uint32_t height,
                                        std::vector<float>& output) {
        // Initialize OIDN device and commit
        oidn::DeviceRef device = oidn::newDevice();
        device.commit();
        const uint32_t imageSize = width * height;

        // Allocate input and output buffers for OIDN
        oidn::BufferRef inputBuffer = device.newBuffer(imageSize * sizeof(float));
        oidn::BufferRef outputBuffer = device.newBuffer(imageSize * sizeof(float));

        // Copy input data to the device buffer
        std::memcpy(inputBuffer.getData(), singleChannelImage, imageSize * sizeof(float));

        // Create and configure the denoising filter
        oidn::FilterRef filter = device.newFilter("RT");
        filter.set("hdr", true);
        filter.setImage("color", inputBuffer, oidn::Format::Float, width, height);
        filter.setImage("output", outputBuffer, oidn::Format::Float, width, height);
        filter.commit();

        // Execute the filter
        filter.execute();

        // Check for errors from OIDN
        const char* errorMessage;
        if (device.getError(errorMessage) != oidn::Error::None) {
            std::cerr << "OIDN Error: " << errorMessage << std::endl;
            return;
        }

        // Retrieve the denoised image data
        output.resize(imageSize);
        std::memcpy(output.data(), outputBuffer.getData(), imageSize * sizeof(float));
    }

    torch::Tensor PhotonRebuildFunction::forward(torch::autograd::AutogradContext* ctx,
                                                 IterationInfo& iterationInfo, PhotonTracer* pathTracer,
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
        ctx->saved_data["IterationInfo"] = reinterpret_cast<int64_t>(&iterationInfo); // example
        // or store the pointer as a raw pointer or shared pointer if you prefer
        // (but be careful with lifetimes).

        // =================
        // 2) Run your path tracer code that renders an image.

        // Example pseudo-code:

        pathTracer->update(iterationInfo.renderSettings);


        // For illustration:
        const PhotonTracer::PipelineSettings& photonTracerSettings = pathTracer->getPipelineSettings();
        int64_t height =photonTracerSettings.height;
        int64_t width = photonTracerSettings.width;
        float* rawImage = pathTracer->getImage();

        std::vector<float> denoisedImage;
        if (iterationInfo.denoise) {
            denoiseImage(rawImage, width, height, denoisedImage);
            rawImage = denoisedImage.data();
        }


        // Suppose the path tracer writes out to pathTracer->m_imageMemory,
        // with shape [height * width] or [height * width * channels].
        // We'll build a Torch tensor from that raw memory.


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
        auto settingsPtr = ctx->saved_data["IterationInfo"].toInt();
        IterationInfo* iterationInfo = reinterpret_cast<IterationInfo*>(settingsPtr);
        save_gradient_to_png(dLoss_dRenderedImage,"gradients/gradient_" + std::to_string(iterationInfo->iteration) + ".png");
        pathTracer->m_backwardInfo.gradientImage = dLoss_dRenderedImage.data_ptr<float>();
        auto gradients = pathTracer->backward(iterationInfo->renderSettings);

        glm::vec3* grad = gradients.sumGradients;

        float grad_x = grad[0].x;
        float grad_y = grad[0].y;
        float grad_z = grad[0].z;

        float grad2_x = grad[1].x;
        float grad2_y = grad[1].y;
        float grad2_z = grad[1].z;

        Log::Logger::getInstance()->info("Gradients: First: {},{},{}, Second: {},{},{}",grad_x, grad_y, grad_z, grad2_x, grad2_y, grad2_z);

        auto posA = positions.accessor<float, 2>();

        Log::Logger::getInstance()->info("Positions: First: {},{},{}, Second: {},{},{}",posA[0][0], posA[0][2], posA[0][3], posA[1][0], posA[1][2], posA[1][3]);

        auto grad_positions = torch::zeros_like(positions);

        auto gradPosA = grad_positions.accessor<float, 2>();
        for (int i = 0; i < grad_positions.size(0); ++i) {
            float gx = grad[i].x;
            float gy = grad[i].y;
            float gz = grad[i].z;

            // Replace NaNs with zero (or another fallback value)
            gradPosA[i][0] = std::isnan(gx) ? 0.0f : gx;
            gradPosA[i][1] = std::isnan(gy) ? 0.0f : gy;
            gradPosA[i][2] = std::isnan(gz) ? 0.0f : gz;
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
