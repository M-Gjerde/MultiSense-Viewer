//
// Created by magnus on 1/15/25.
//

#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracer.h"

#include <stb_image_write.h>
#include <yaml-cpp/emitter.h>

#include "Viewer/Application/Application.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"
#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayerUI.h"
#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer.h"

#ifdef SYCL_ENABLED
#include <OpenImageDenoise/oidn.hpp>
#endif

namespace VkRender {
    EditorPathTracer::EditorPathTracer(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorPathTracerLayer");
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUIData<EditorPathTracerLayerUI>();

        m_descriptorRegistry.createManager(DescriptorManagerType::Viewport3DTexture, m_context->vkDevice());

        m_shaderSelectionBuffer.resize(m_context->swapChainBuffers().size());
        for (auto &frameIndex: m_shaderSelectionBuffer) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                frameIndex,
                sizeof(EditorPathTracerLayerUI::ShaderSelection), nullptr, "EditorPathTracer:ShaderSelectionBuffer",
                m_context->getDebugUtilsObjectNameFunction());
        }

        m_editorCamera = std::make_shared<ArcballCamera>();
        m_editorCamera->setDefaultPosition({-90.0f, -60.0f}, 1.5f);
    }

    void EditorPathTracer::onEditorResize() {
        float width = m_createInfo.width;
        float height = m_createInfo.height;
        float editorAspect = static_cast<float>(m_createInfo.width) /
                             static_cast<float>(m_createInfo.height);

        m_editorCamera = std::make_shared<ArcballCamera>(
            static_cast<float>(m_createInfo.width) / static_cast<float>(m_createInfo.height));
        m_editorCamera->setDefaultPosition({-90.0f, -60.0f}, 1.5f);

        auto imageUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_ui);

        auto sceneCamera = m_context->activeScene()->getActiveCamera();
        // 1. Figure out what camera we are using and the correct resolution.
        bool useSceneCamera = (imageUI->useSceneCamera && sceneCamera && sceneCamera->cameraType ==
                               CameraComponent::PINHOLE);
        uint32_t newWidth = m_createInfo.width;
        uint32_t newHeight = m_createInfo.height;
        if (useSceneCamera) {
            newWidth = sceneCamera->pinholeParameters.width;
            newHeight = sceneCamera->pinholeParameters.height;
        }
    }

    void EditorPathTracer::onFileDrop(const std::filesystem::path &path) {
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension == ".png" || extension == ".jpg") {
            m_colorTexture = EditorUtils::createTextureFromFile(path, m_context);
        }
    }


    void EditorPathTracer::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_editorCamera = std::make_shared<ArcballCamera>(
            static_cast<float>(m_createInfo.width) / static_cast<float>(m_createInfo.height));
        m_editorCamera->setDefaultPosition({-90.0f, -60.0f}, 1.5f);

        m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height,
                                                         VK_FORMAT_R8G8B8A8_UNORM, m_context);
        auto activeCamera = m_context->activeScene()->getActiveCamera();
        m_lastActiveCamera = activeCamera;

        std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_ui)->resetPathTracer = true;
        updatePathTracerSettings();
    }

    void EditorPathTracer::updatePathTracerSettings() {
        auto imageUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_ui);
        auto activeCamera = m_context->activeScene()->getActiveCamera();
        if (imageUI->switchKernelDevice || imageUI->resetPathTracer) {
            Log::Logger::getInstance()->info("Setting New Kernel Device");
            SYCLDeviceType deviceType = SYCLDeviceType::CPU;
            if (imageUI->kernelDevice == "GPU") {
                deviceType = SYCLDeviceType::GPU;
            }
            auto syclDevice = m_context->getSyclDeviceSelector().getDevice(deviceType);
            uint32_t width = m_createInfo.width;
            uint32_t height = m_createInfo.height;
            if (activeCamera && imageUI->useSceneCamera) {
                width = activeCamera->pinholeParameters.width;
                height = activeCamera->pinholeParameters.height;
            }
            PathTracer::PhotonTracer::PipelineSettings pipelineSettings(syclDevice, width, height);
            pipelineSettings.photonCount = imageUI->photonCount;
            pipelineSettings.numBounces = imageUI->numBounces;
            m_pathTracer.reset();
            vkDeviceWaitIdle(m_context->vkDevice().m_LogicalDevice);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            m_pathTracer = std::make_unique<PathTracer::PhotonTracer>(m_context, pipelineSettings,
                                                                      m_context->activeScene());
            syclDevice->getQueue().wait();
            auto list = syclDevice->getQueue().get_wait_list();
            for (const auto &event: list) {
                try {
                    auto status = event.get_info<sycl::info::event::command_execution_status>();
                    Log::Logger::getInstance()->info("Event status: {}", static_cast<int>(status));
                } catch (const std::exception &e) {
                    Log::Logger::getInstance()->error("Error retrieving event info: {}", e.what());
                }
            }
            vkDeviceWaitIdle(m_context->vkDevice().m_LogicalDevice);
            float editorAspect = static_cast<float>(m_createInfo.width) /
                                 static_cast<float>(m_createInfo.height);
            float sceneCameraAspect = static_cast<float>(width) /
                                      static_cast<float>(height);
            float scaleX = 1.0f, scaleY = 1.0f;
            if (editorAspect > sceneCameraAspect) {
                scaleX = sceneCameraAspect / editorAspect;
            } else {
                scaleY = editorAspect / sceneCameraAspect;
            }
            m_meshInstances.reset();
            m_meshInstances = EditorUtils::setupMesh(m_context, scaleX, scaleY);

            Log::Logger::getInstance()->info("Created New Color Texture");
        }

        imageUI->switchKernelDevice = false;
    }

    void EditorPathTracer::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_ui);

        auto activeCamera = m_context->activeScene()->getActiveCamera();
        bool newCamera = m_previousSceneCamera != activeCamera;

        updatePathTracerSettings();

        if (imageUI->clearImageMemory || newCamera) {
            m_pathTracer->resetImage();
        }

        // 4. If user wants to render/preview, update the path tracer with the latest camera.
        if (imageUI->render || imageUI->toggleRendering) {
            PathTracer::PhotonTracer::RenderSettings renderSettings;

            bool useSceneCamera = activeCamera;
            if (useSceneCamera && imageUI->useSceneCamera) {
                auto entity = m_context->activeScene()->getActiveCameraEntity();
                auto camera = entity.getComponent<CameraComponent>();
                auto transform = entity.getComponent<TransformComponent>();
                renderSettings.camera = *camera.getPinholeCamera();
                renderSettings.cameraTransform = transform;
                if (entity.getComponent<TransformComponent>().moved())
                    m_pathTracer->resetImage();
            } else {
                // Otherwise, construct a default pinhole camera for your editor camera
                PinholeParameters pinholeParameters;
                SharedCameraSettings cameraSettings;
                pinholeParameters.width = m_createInfo.width;
                pinholeParameters.height = m_createInfo.height;
                pinholeParameters.cx = pinholeParameters.width / 2.0f;
                pinholeParameters.cy = pinholeParameters.height / 2.0f;
                pinholeParameters.fx = 600.0f;
                pinholeParameters.fy = 600.0f;
                // Construct the pinhole
                PinholeCamera defaultCam(cameraSettings, pinholeParameters);
                renderSettings.camera = defaultCam;
                renderSettings.cameraTransform = TransformComponent(m_editorCamera->matrices.transform);
                if (m_movedCamera) {
                    m_pathTracer->resetImage();
                }
            }

            renderSettings.gammaCorrection = imageUI->shaderSelection.gammaCorrection;

            if (imageUI->clearImageMemory) {
                uint32_t width = m_createInfo.width;
                uint32_t height = m_createInfo.height;
                if (activeCamera && imageUI->useSceneCamera) {
                    width = activeCamera->pinholeParameters.width;
                    height = activeCamera->pinholeParameters.height;
                }
                m_colorTexture = EditorUtils::createEmptyTexture(
                    width,
                    height,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    m_context);
            }

            bool imageSizeMatch = static_cast<uint32_t>(renderSettings.camera.m_parameters.width) == m_colorTexture->
                                  width() &&
                                  static_cast<uint32_t>(renderSettings.camera.m_parameters.height) == m_colorTexture
                                  ->height();
            if (imageSizeMatch) {
                m_pathTracer->update(renderSettings);

                float *image = m_pathTracer->getImage();
                const uint32_t texWidth = m_colorTexture->width();
                const uint32_t texHeight = m_colorTexture->height();

                if (image) {
                    const size_t totalPixels = static_cast<size_t>(texWidth) * texHeight;

                    // Prepare a container for the final image (after optional denoising)
                    float *finalImage = image;
                    std::vector<float> denoisedImage; // only used if denoising is enabled

                    // Denoise if requested; otherwise, use the original image directly.
                    if (imageUI->denoise) {
                        denoiseImage(image, texWidth, texHeight, denoisedImage);
                        // Use the denoised data if the denoising call was successful.
                        finalImage = denoisedImage.data();
                    }
                    // Allocate the converted image buffer (RGBA: 4 channels per pixel) // TODO Make the shader accepts floating point images and clamp/convert it in shader instead
                    std::vector<uint8_t> convertedImage(totalPixels * 4);
                    // Convert the final image from float [0, 1] to 8-bit RGBA.
                    // We assume a grayscale image is stored in the red channel.
                    for (size_t i = 0; i < totalPixels; ++i) {
                        // Clamp the float value and scale to 0-255.
                        // (Multiplication by 255.0f and conversion to uint8_t)
                        float clamped = std::clamp(finalImage[i], 0.0f, 1.0f);
                        auto value = static_cast<uint8_t>(clamped * 255.0f);
                        size_t offset = i * 4;
                        convertedImage[offset + 0] = value; // R
                        convertedImage[offset + 1] = value; // G
                        convertedImage[offset + 2] = value; // B
                        convertedImage[offset + 3] = 255; // A (fully opaque)
                    }
                    // Upload the texture
                    Log::Logger::getInstance()->trace(
                        "Uploading Path Tracer Image to Color Texture. Size: {} bytes into {}",
                        convertedImage.size(), m_colorTexture->getSize());


                    m_colorTexture->loadImage(convertedImage.data(), convertedImage.size());
                    Log::Logger::getInstance()->trace("Uploaded new Texture Data");
                }
            } else {
                Log::Logger::getInstance()->warning("Image size Mismatch! Texture: {}x{}, Camera: {}x{}",
                                                    m_colorTexture->width(), m_colorTexture->height(),
                                                    renderSettings.camera.m_parameters.width,
                                                    renderSettings.camera.m_parameters.height);
            }
        }
        if (imageUI->saveImage || imageUI->bypassSave) {
            saveImage();
        }


        auto frameIndex = m_context->currentFrameIndex();
        void *data;
        vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                    m_shaderSelectionBuffer[frameIndex]->m_memory, 0,
                    sizeof(EditorPathTracerLayerUI::ShaderSelection),
                    0, &data);
        memcpy(data, &imageUI->shaderSelection, sizeof(EditorPathTracerLayerUI::ShaderSelection));
        vkUnmapMemory(m_context->vkDevice().m_LogicalDevice, m_shaderSelectionBuffer[frameIndex]->m_memory);

        // Reset some state variables
        m_movedCamera = false;
        imageUI->clearImageMemory = false;
        m_previousSceneCamera = activeCamera;
    }


    void EditorPathTracer::onMouseMove(const MouseButtons &mouse) {
        if (ui()->hovered && mouse.left && !ui()->resizeActive) {
            m_editorCamera->rotate(mouse.dx, mouse.dy);
            m_movedCamera = true;
        } else if (ui()->hovered && mouse.right && !ui()->resizeActive) {
            m_editorCamera->translate(mouse.dx, mouse.dy);
            m_movedCamera = true;
        }
    }

    void EditorPathTracer::onMouseScroll(float change) {
        if (ui()->hovered) {
            m_editorCamera->zoom((change > 0.0f) ? 0.95f : 1.05f);
            m_movedCamera = true;
        }
    }

    void EditorPathTracer::onKeyCallback(const Input &input) {
        if (input.lastKeyPress == GLFW_KEY_SPACE) {
            m_editorCamera->setDefaultPosition({-90.0f, -60.0f}, 1.5f);
        };
    }

    void EditorPathTracer::onRender(CommandBuffer &commandBuffer) {
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand> > renderGroups;
        collectRenderCommands(renderGroups, commandBuffer.frameIndex);
        Log::Logger::getInstance()->trace("Collected Drawing commands");

        // Render each group
        for (auto &[pipeline, commands]: renderGroups) {
            pipeline->bind(commandBuffer);
            for (auto &command: commands) {
                // Bind resources and draw
                bindResourcesAndDraw(commandBuffer, command);
            }
        }

        Log::Logger::getInstance()->trace("Drawing Path tracer");
    }

    void EditorPathTracer::collectRenderCommands(
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand> > &renderGroups,
        uint32_t frameIndex) {
        if (!m_meshInstances) {
            m_meshInstances = EditorUtils::setupMesh(m_context);
            Log::Logger::getInstance()->info("Created MeshInstance for 3DViewport");
        }
        if (!m_meshInstances)
            return;
        PipelineKey key = {};
        key.setLayouts.resize(1);
        auto imageUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_ui);
        Log::Logger::getInstance()->trace("Collecting Render commands for Path Tracer");

        // Prepare descriptor writes based on your texture or other resources
        std::array<VkWriteDescriptorSet, 2> writeDescriptors{};
        writeDescriptors[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptors[0].dstBinding = 0; // Binding index
        writeDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptors[0].descriptorCount = 1;
        writeDescriptors[0].pImageInfo = &m_colorTexture->getDescriptorInfo();
        writeDescriptors[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptors[1].dstBinding = 1; // Binding index
        writeDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptors[1].descriptorCount = 1;
        writeDescriptors[1].pBufferInfo = &m_shaderSelectionBuffer[frameIndex]->m_descriptorBufferInfo;
        std::vector descriptorWrites = {writeDescriptors[0], writeDescriptors[1]};
        VkDescriptorSet descriptorSet = m_descriptorRegistry.getManager(
            DescriptorManagerType::Viewport3DTexture).getOrCreateDescriptorSet(descriptorWrites);
        key.setLayouts[0] = m_descriptorRegistry.getManager(
            DescriptorManagerType::Viewport3DTexture).getDescriptorSetLayout();
        // Use default descriptor set layout
        key.vertexShaderName = "default2D.vert";
        key.fragmentShaderName = "EditorPathTracerTexture.frag";
        key.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        key.polygonMode = VK_POLYGON_MODE_FILL;
        std::vector<VkVertexInputBindingDescription> vertexInputBinding = {
            {0, sizeof(VkRender::ImageVertex), VK_VERTEX_INPUT_RATE_VERTEX}
        };
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
            {1, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 2},
        };
        key.vertexInputBindingDescriptions = vertexInputBinding;
        key.vertexInputAttributes = vertexInputAttributes;

        // Create or retrieve the pipeline
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        renderPassInfo.debugName = "EditorPathTracer::";
        auto pipeline = m_pipelineManager.getOrCreatePipeline(key, renderPassInfo, m_context);
        // Create the render command
        RenderCommand command;
        command.pipeline = pipeline;
        command.meshInstance = m_meshInstances.get();
        command.descriptorSets[DescriptorManagerType::Viewport3DTexture] = descriptorSet;
        // Assign the descriptor set
        // Add to render group
        renderGroups[pipeline].push_back(command);
    }

    void EditorPathTracer::bindResourcesAndDraw(const CommandBuffer &commandBuffer, RenderCommand &command) {
        VkCommandBuffer cmdBuffer = commandBuffer.getActiveBuffer();
        uint32_t frameIndex = commandBuffer.frameIndex;

        if (command.meshInstance->vertexBuffer) {
            VkBuffer vertexBuffers[] = {command.meshInstance->vertexBuffer->m_buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vertexBuffers, offsets);
            // Bind index buffer if the mesh has indices
            if (command.meshInstance->indexBuffer) {
                vkCmdBindIndexBuffer(cmdBuffer, command.meshInstance->indexBuffer->m_buffer, 0,
                                     VK_INDEX_TYPE_UINT32);
            }
        }

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          command.pipeline->pipeline()->getPipeline());


        for (auto &[index, descriptorSet]: command.descriptorSets) {
            vkCmdBindDescriptorSets(
                cmdBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                command.pipeline->pipeline()->getPipelineLayout(),
                0, // TODO can't reuse the approach in SceneRenderer since we have different manager types
                1,
                &descriptorSet,
                0,
                nullptr
            );
        }

        if (command.meshInstance->indexCount > 0) {
            vkCmdDrawIndexed(cmdBuffer, command.meshInstance->indexCount, 1, 0, 0, 0);
        }
    }


    void EditorPathTracer::saveImage() {
        auto imageUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_ui);

        // Save render information:
        PathTracer::RenderInformation *info = m_pathTracer->getRenderInfo();
        // Create a YAML emitter
        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "Gamma" << YAML::Value << info->gamma;
        out << YAML::Key << "PhotonHitCount" << YAML::Value << info->photonsAccumulated;
        out << YAML::Key << "PhotonBounceCount" << YAML::Value << info->numBounces;
        out << YAML::Key << "PhotonsEmitted" << YAML::Value << info->totalPhotons;
        out << YAML::Key << "FrameCount" << YAML::Value << info->frameID; // Start from 0
        out << YAML::EndMap;

        auto sceneCamera = m_context->activeScene()->getActiveCamera();
        std::string prefix = "Viewport";
        if (sceneCamera) {
            prefix = m_context->activeScene()->getActiveCameraEntity().getName();
        }

        // Write YAML content to a file
        std::string infoFileName = "output/" + prefix + ":render_info.yaml";
        // Change to your desired infoFileName/path
        std::ofstream fout(infoFileName);
        if (fout.is_open()) {
            fout << out.c_str();
            fout.close();
        }

        uint32_t width = m_colorTexture->width();
        uint32_t height = m_colorTexture->height();
        float *image = m_pathTracer->getImage();
        std::vector<float> denoisedImage;

        if (imageUI->denoise) {
            denoiseImage(image, width, height, denoisedImage);
            image = denoisedImage.data();
        }


        std::filesystem::path filename = "output/" + prefix + ".pfm";
        if (!std::filesystem::exists(filename)) {
            std::filesystem::create_directory(filename.parent_path());
        }
        std::ofstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename.string());
        }
        // Write the PFM header
        // "PF" indicates a color image. Use "Pf" for grayscale.
        file << "PF\n" << width << " " << height << "\n-1.0\n";

        // PFM expects the data in binary format, row by row from top to bottom
        // Assuming your m_imageMemory is in RGBA format with floats

        // Allocate a temporary buffer for RGB data
        std::vector<float> rgbData(width * height * 3);

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t pixelIndex = (y * width + x);
                uint32_t rgbIndex = (y * width + x) * 3;

                rgbData[rgbIndex + 0] = image[pixelIndex]; // R
                rgbData[rgbIndex + 1] = image[pixelIndex]; // G
                rgbData[rgbIndex + 2] = image[pixelIndex]; // B
            }
        }

        // Write the RGB float data
        file.write(reinterpret_cast<const char *>(rgbData.data()), rgbData.size() * sizeof(float));

        if (!file) {
            throw std::runtime_error("Failed to write PFM data to file: " + filename.string());
        }

        file.close();

        std::vector<uint8_t> rgbDataPng(width * height * 3);

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t pixelIndex = (y * width + x);
                uint32_t rgbIndex = pixelIndex * 3;

                // Assuming image is in RGBA format with float values in range [0.0, 1.0]
                rgbDataPng[rgbIndex + 0] = static_cast<uint8_t>(std::clamp(image[pixelIndex], 0.0f, 1.0f) * 255.0f);
                // R
                rgbDataPng[rgbIndex + 1] = static_cast<uint8_t>(std::clamp(image[pixelIndex], 0.0f, 1.0f) * 255.0f);
                // G
                rgbDataPng[rgbIndex + 2] = static_cast<uint8_t>(std::clamp(image[pixelIndex], 0.0f, 1.0f) * 255.0f);
                // B
            }
        }


        // Write the image to a PNG file
        if (!stbi_write_png(filename.replace_extension(".png").string().c_str(), width, height, 3,
                            rgbDataPng.data(),
                            width * 3)) {
            throw std::runtime_error("Failed to write PNG file: " + filename.string());
        }
    }


    void EditorPathTracer::denoiseImage(float *singleChannelImage, uint32_t width, uint32_t height,
                                        std::vector<float> &output) {
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
        filter.set("hdr", false);
        filter.setImage("color", inputBuffer, oidn::Format::Float, width, height);
        filter.setImage("output", outputBuffer, oidn::Format::Float, width, height);
        filter.commit();

        // Execute the filter
        filter.execute();

        // Check for errors from OIDN
        const char *errorMessage;
        if (device.getError(errorMessage) != oidn::Error::None) {
            std::cerr << "OIDN Error: " << errorMessage << std::endl;
            return;
        }

        // Retrieve the denoised image data
        output.resize(imageSize);
        std::memcpy(output.data(), outputBuffer.getData(), imageSize * sizeof(float));
    }
}
