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
    EditorPathTracer::EditorPathTracer(EditorCreateInfo& createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorPathTracerLayer");
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUIData<EditorPathTracerLayerUI>();

        m_descriptorRegistry.createManager(DescriptorManagerType::Viewport3DTexture, m_context->vkDevice());

        m_shaderSelectionBuffer.resize(m_context->swapChainBuffers().size());
        for (auto& frameIndex : m_shaderSelectionBuffer) {
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
        m_activeScene = m_context->activeScene();

        float width = m_createInfo.width;
        float height = m_createInfo.height;
        float editorAspect = static_cast<float>(m_createInfo.width) /
                             static_cast<float>(m_createInfo.height);

        m_editorCamera = std::make_shared<ArcballCamera>(
                static_cast<float>(m_createInfo.width) / static_cast<float>(m_createInfo.height));
        m_editorCamera->setDefaultPosition({-90.0f, -60.0f}, 1.5f);




        if (m_lastActiveCamera) {
            auto camera = m_lastActiveCamera->getPinholeCamera();
            float textureAspect = static_cast<float>(camera->parameters().width) / static_cast<float>(camera->parameters().height);
            float editorAspect = static_cast<float>(m_createInfo.width) / static_cast<float>(m_createInfo.height);
            // Calculate scaling factors
            float scaleX = 1.0f, scaleY = 1.0f;
            if (editorAspect > textureAspect) {
                scaleX = textureAspect / editorAspect;
            } else {
                scaleY = editorAspect / textureAspect;
            }
            m_meshInstances.reset();
            m_meshInstances = nullptr;
            m_meshInstances = EditorUtils::setupMesh(m_context, scaleX, scaleY);
            float width = camera->parameters().width;
            float height = camera->parameters().height;
            m_pathTracer = std::make_unique<PathTracer::PhotonTracer>(m_context, m_activeScene, width, height);
            m_colorTexture = EditorUtils::createEmptyTexture(width, height, VK_FORMAT_R8G8B8A8_UNORM, m_context);

        } else {

            m_meshInstances.reset();
            m_meshInstances = nullptr;
            m_meshInstances = EditorUtils::setupMesh(m_context);
            TransformComponent transformComponent;
            transformComponent.setPosition(m_editorCamera->matrices.position);
            transformComponent.setRotationQuaternion(glm::quat_cast(glm::inverse(m_editorCamera->matrices.view)));
            transformComponent.setMoving(true);
            m_pathTracer = std::make_unique<PathTracer::PhotonTracer>(m_context, m_activeScene, m_createInfo.width, m_createInfo.height);
            m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height, VK_FORMAT_R8G8B8A8_UNORM, m_context);
            m_pathTracer->setActiveCamera(transformComponent, m_createInfo.width, m_createInfo.height);

        }
    }

    void EditorPathTracer::onFileDrop(const std::filesystem::path& path) {
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension == ".png" || extension == ".jpg") {
            m_colorTexture = EditorUtils::createTextureFromFile(path, m_context);
        }
    }


    void EditorPathTracer::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_activeScene = scene;

        m_editorCamera = std::make_shared<ArcballCamera>(
                static_cast<float>(m_createInfo.width) / static_cast<float>(m_createInfo.height));
        m_editorCamera->setDefaultPosition({-90.0f, -60.0f}, 1.5f);

        if (m_lastActiveCamera) {
            auto camera = m_lastActiveCamera->getPinholeCamera();
            float textureAspect = static_cast<float>(camera->parameters().width) / static_cast<float>(camera->parameters().height);
            float editorAspect = static_cast<float>(m_createInfo.width) / static_cast<float>(m_createInfo.height);
            // Calculate scaling factors
            float scaleX = 1.0f, scaleY = 1.0f;
            if (editorAspect > textureAspect) {
                scaleX = textureAspect / editorAspect;
            } else {
                scaleY = editorAspect / textureAspect;
            }
            m_meshInstances.reset();
            m_meshInstances = nullptr;
            m_meshInstances = EditorUtils::setupMesh(m_context, scaleX, scaleY);
            float width = camera->parameters().width;
            float height = camera->parameters().height;
            m_pathTracer = std::make_unique<PathTracer::PhotonTracer>(m_context, m_activeScene, width, height);
            m_colorTexture = EditorUtils::createEmptyTexture(width, height, VK_FORMAT_R8G8B8A8_UNORM, m_context);
        } else {
            m_meshInstances.reset();
            m_meshInstances = nullptr;
            m_meshInstances = EditorUtils::setupMesh(m_context);
            TransformComponent transformComponent;
            transformComponent.setPosition(m_editorCamera->matrices.position);
            transformComponent.setRotationQuaternion(glm::quat_cast(glm::inverse(m_editorCamera->matrices.view)));
            transformComponent.setMoving(true);
            m_pathTracer = std::make_unique<PathTracer::PhotonTracer>(m_context, m_activeScene, m_createInfo.width, m_createInfo.height);
            m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height, VK_FORMAT_R8G8B8A8_UNORM, m_context);
            m_pathTracer->setActiveCamera(transformComponent, m_createInfo.width, m_createInfo.height);
            m_lastActiveCamera = nullptr;
        }
    }


    void EditorPathTracer::onPipelineReload() {
    }

    // Example function for single-channel to 3-channel expansion
    std::vector<float> expandTo3Channels(const float* singleChannelImage, uint32_t width, uint32_t height) {
        std::vector<float> rgbImage(width * height * 3);
        for (uint32_t i = 0; i < width * height; ++i) {
            rgbImage[i * 3 + 0] = static_cast<uint8_t>(std::clamp(singleChannelImage[i], 0.0f, 1.0f)); // R
            rgbImage[i * 3 + 1] = static_cast<uint8_t>(std::clamp(singleChannelImage[i], 0.0f, 1.0f)); // G
            rgbImage[i * 3 + 2] = static_cast<uint8_t>(std::clamp(singleChannelImage[i], 0.0f, 1.0f)); // B
        }
        return rgbImage;
    }

    void denoiseImage(float* singleChannelImage, uint32_t width, uint32_t height, std::vector<float>& output) {
#ifdef SYCL_ENABLED

        // Expand single-channel image to 3 channels
        // Initialize OIDN
        oidn::DeviceRef device = oidn::newDevice();
        device.commit();
        uint32_t imageSize = width * height;
        // Allocate input and output buffers
        oidn::BufferRef inputBuffer = device.newBuffer(imageSize * sizeof(float));
        oidn::BufferRef outputBuffer = device.newBuffer(imageSize * sizeof(float));

        // Copy input data to OIDN buffer
        std::memcpy(inputBuffer.getData(), singleChannelImage, imageSize * sizeof(float));

        // Create and configure the denoising filter
        oidn::FilterRef filter = device.newFilter("RT");
        filter.set("hdr", true);
        filter.setImage("color", inputBuffer, oidn::Format::Float, width, height);
        filter.setImage("output", outputBuffer, oidn::Format::Float, width, height);
        filter.commit();

        // Execute the filter
        filter.execute();

        // Check for errors
        const char* errorMessage;
        if (device.getError(errorMessage) != oidn::Error::None) {
            std::cerr << "OIDN Error: " << errorMessage << std::endl;
            return;
        }
/*
        // Retrieve and normalize the denoised image
        const float* outputBufferData = static_cast<const float*>(outputBuffer.getData());
        // Find min and max values
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::lowest();
        for (uint32_t i = 0; i < imageSize; ++i) {
            minVal = std::min(minVal, outputBufferData[i]);
            maxVal = std::max(maxVal, outputBufferData[i]);
        }

        // Normalize and copy to output
        output.resize(imageSize);
        float range = maxVal - minVal;
        for (uint32_t i = 0; i < imageSize; ++i) {
            output[i] = (outputBufferData[i] - minVal) / range;
        }
        */

        // Retrieve the denoised image
        output.resize(imageSize);
        std::memcpy(output.data(), outputBuffer.getData(), output.size() * sizeof(float));
#endif

    }

    void EditorPathTracer::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_ui);
        m_activeScene = m_context->activeScene();

        updateActiveCamera();

        auto frameIndex = m_context->currentFrameIndex();

        void* data;
        vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                    m_shaderSelectionBuffer[frameIndex]->m_memory, 0, sizeof(EditorPathTracerLayerUI::ShaderSelection), 0, &data);
        memcpy(data, &imageUI->shaderSelection, sizeof(EditorPathTracerLayerUI::ShaderSelection));
        vkUnmapMemory(m_context->vkDevice().m_LogicalDevice, m_shaderSelectionBuffer[frameIndex]->m_memory);

        if (imageUI->uploadScene) {
            m_pathTracer->upload(m_context->activeScene());
        }

        if (imageUI->switchKernelDevice) {
            PathTracer::PhotonTracer::Settings settings;
            settings.kernelDevice = imageUI->kernelDevice;
            m_pathTracer->setExecutionDevice(settings);
            m_pathTracer->upload(m_context->activeScene());
        }

        if (imageUI->render || imageUI->toggleRendering) {
            PathTracer::PhotonTracer::Settings settings;
            settings.clearImageMemory = imageUI->clearImageMemory;
            settings.kernelType = imageUI->kernel;
            settings.kernelDevice = imageUI->kernelDevice;
            settings.photonCount = imageUI->photonCount;
            settings.numBounces = imageUI->numBounces;
            settings.gammaCorrection = imageUI->shaderSelection.gammaCorrection;

            m_pathTracer->update(settings);
            float* image = m_pathTracer->getImage();
            if (image) {
                uint32_t width = m_colorTexture->width();
                uint32_t height = m_colorTexture->height();
                // Handle denoising if enabled
                std::vector<uint8_t> convertedImage(width * height * 4); // 4 channels: RGBA

                if (imageUI->denoise) {
                    std::vector<float> denoisedImage;
                    denoiseImage(image, width, height, denoisedImage);
                    for (uint32_t i = 0; i < width * height; ++i) {
                        float r = denoisedImage[i]; // Assuming grayscale in the red channel

                        convertedImage[i * 4 + 0] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f); // R
                        convertedImage[i * 4 + 1] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f); // G
                        convertedImage[i * 4 + 2] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f); // B
                        convertedImage[i * 4 + 3] = 255; // A (fully opaque)
                    }

                } else {
                    for (uint32_t i = 0; i < width * height; ++i) {
                        float r = image[i]; // Assuming grayscale in the red channel
                        convertedImage[i * 4 + 0] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f); // R
                        convertedImage[i * 4 + 1] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f); // G
                        convertedImage[i * 4 + 2] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f); // B
                        convertedImage[i * 4 + 3] = 255; // A (fully opaque)
                    }
                }
                // Upload the texture
                m_colorTexture->loadImage(convertedImage.data(), convertedImage.size());
            }
        }
        if (imageUI->saveImage) {
            // Save render information:
            PathTracer::RenderInformation info = m_pathTracer->getRenderInfo();
            // Create a YAML emitter
            YAML::Emitter out;
            out << YAML::BeginMap;
            out << YAML::Key << "Gamma"                    << YAML::Value << info.gamma;
            out << YAML::Key << "PhotonHitCount"       << YAML::Value << info.photonsAccumulated;
            out << YAML::Key << "PhotonBounceCount"       << YAML::Value << info.numBounces;
            out << YAML::Key << "PhotonsEmitted"       << YAML::Value << info.totalPhotons;
            out << YAML::Key << "FrameCount"                  << YAML::Value << info.frameID; // Start from 0
            out << YAML::EndMap;

            // Write YAML content to a file
            std::string infoFileName = "render_info.yaml";  // Change to your desired infoFileName/path
            std::ofstream fout(infoFileName);
            if (fout.is_open()) {
                fout << out.c_str();
                fout.close();
            }

            uint32_t width = m_colorTexture->width();
            uint32_t height = m_colorTexture->height();
            float* image = m_pathTracer->getImage();
            std::vector<float> denoisedImage;

            if (imageUI->denoise) {
                denoiseImage(image, width, height, denoisedImage);
                image = denoisedImage.data();
            }


            std::filesystem::path filename = "screenshot.pfm";
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
            file.write(reinterpret_cast<const char*>(rgbData.data()), rgbData.size() * sizeof(float));

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
                    rgbDataPng[rgbIndex + 0] = static_cast<uint8_t>(std::clamp(image[pixelIndex], 0.0f, 1.0f) * 255.0f); // R
                    rgbDataPng[rgbIndex + 1] = static_cast<uint8_t>(std::clamp(image[pixelIndex], 0.0f, 1.0f) * 255.0f); // G
                    rgbDataPng[rgbIndex + 2] = static_cast<uint8_t>(std::clamp(image[pixelIndex], 0.0f, 1.0f) * 255.0f); // B
                }
            }


            // Write the image to a PNG file
            if (!stbi_write_png(filename.replace_extension(".png").string().c_str(), width, height, 3, rgbDataPng.data(), width * 3)) {
                throw std::runtime_error("Failed to write PNG file: " + filename.string());
            }

        }
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

    void EditorPathTracer::onRender(CommandBuffer& commandBuffer) {
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>> renderGroups;
        collectRenderCommands(renderGroups, commandBuffer.frameIndex);

        // Render each group
        for (auto& [pipeline, commands] : renderGroups) {
            pipeline->bind(commandBuffer);
            for (auto& command : commands) {
                // Bind resources and draw
                bindResourcesAndDraw(commandBuffer, command);
            }
        }
    }

    void EditorPathTracer::collectRenderCommands(
            std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>>& renderGroups,
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
        command.descriptorSets[DescriptorManagerType::Viewport3DTexture] = descriptorSet; // Assign the descriptor set
        // Add to render group
        renderGroups[pipeline].push_back(command);
    }

    void EditorPathTracer::bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand& command) {
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


        for (auto& [index, descriptorSet] : command.descriptorSets) {
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


    void EditorPathTracer::updateActiveCamera(){
        auto imageUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_ui);

        // By default, assume we are *not* using a scene camera this frame
        bool isSceneCameraActive = false;
        CameraComponent *sceneCameraToUse = nullptr;
        TransformComponent* tfComponentPtr;
        // If the user wants to render from a viewpoint, see if there's a valid scene camera
        if (imageUI->renderFromViewpoint) {
            auto view = m_activeScene->getRegistry().view<CameraComponent>();
            for (auto e: view) {
                Entity entity(e, m_activeScene.get());
                auto &cameraComponent = entity.getComponent<CameraComponent>();

                if (cameraComponent.renderFromViewpoint()) {
                    // Use the first camera found that can render from viewpoint
                    sceneCameraToUse = &cameraComponent;
                    tfComponentPtr = &entity.getComponent<TransformComponent>();
                    break;
                }
            }
        }

        // If we found a valid scene camera, use that
        if (sceneCameraToUse) {
            isSceneCameraActive = true;

            // Detect if we’re switching from editor camera to scene camera,
            // switching from one scene camera to another, or if the scene camera
            // explicitly requests an update (updateTrigger).
            bool isNewCameraSelected = (!m_wasSceneCameraActive ||
                                        (m_lastActiveCamera != sceneCameraToUse));

            if (isNewCameraSelected && sceneCameraToUse->updateTrigger()) {
                // Recompute mesh scaling for the new camera
                float sceneCameraAspect = 1.0f;
                float width;
                float height;
                float editorAspect = static_cast<float>(m_createInfo.width) /
                                     static_cast<float>(m_createInfo.height);

                switch (sceneCameraToUse->cameraType) {
                    case CameraComponent::PERSPECTIVE: {
                        auto perspectiveCamera = sceneCameraToUse->getPerspectiveCamera();
                        sceneCameraAspect = perspectiveCamera->m_parameters.aspect;
                        width = m_createInfo.width;
                        height = m_createInfo.height;
                        break;
                    }
                    case CameraComponent::PINHOLE: {
                        auto pinholeCamera = sceneCameraToUse->getPinholeCamera();
                        sceneCameraAspect = pinholeCamera->parameters().width / pinholeCamera->parameters().height;
                        width = pinholeCamera->parameters().width;
                        height = pinholeCamera->parameters().height;
                        break;
                    }
                    default:
                        Log::Logger::getInstance()->error("Camera type not implemented for scene cameras");
                        width = m_createInfo.width;
                        height = m_createInfo.height;
                        sceneCameraAspect = editorAspect;
                        break;
                }

                float scaleX = 1.0f, scaleY = 1.0f;
                if (editorAspect > sceneCameraAspect) {
                    scaleX = sceneCameraAspect / editorAspect;
                } else {
                    scaleY = editorAspect / sceneCameraAspect;
                }

                // Reset and rebuild mesh
                m_meshInstances.reset();
                m_meshInstances = EditorUtils::setupMesh(m_context, scaleX, scaleY);

                m_pathTracer = std::make_unique<PathTracer::PhotonTracer>(m_context, m_activeScene, width, height);
                m_colorTexture = EditorUtils::createEmptyTexture(width, height, VK_FORMAT_R8G8B8A8_UNORM, m_context);
            }
            m_lastActiveCamera = sceneCameraToUse;

            // Activate the scene camera
            auto pinholeCamera = sceneCameraToUse->getPinholeCamera();
            float width = pinholeCamera->parameters().width;
            float height = pinholeCamera->parameters().height;

            m_pathTracer->setActiveCamera(pinholeCamera, tfComponentPtr);

        } else {
            // No valid scene camera — revert to editor camera
            isSceneCameraActive = false;

            // Only do the mesh/aspect revert if we were using a scene camera last frame
            if (m_wasSceneCameraActive) {
                // Restore editor mesh and aspect ratio
                m_meshInstances.reset();
                m_meshInstances = EditorUtils::setupMesh(m_context);
                float width = m_createInfo.width;
                float height = m_createInfo.height;
                /*
                auto &ci = m_sceneRenderer->getCreateInfo();
                ci.width = m_createInfo.width;
                ci.height = m_createInfo.height;
                m_sceneRenderer->resize(ci);

                onRenderSettingsChanged();
                */

                m_pathTracer = std::make_unique<PathTracer::PhotonTracer>(m_context, m_activeScene, width, height);
                m_colorTexture = EditorUtils::createEmptyTexture(width, height, VK_FORMAT_R8G8B8A8_UNORM, m_context);
            }

            float width = m_createInfo.width;
            float height = m_createInfo.height;

            TransformComponent transformComponent;
            transformComponent.setPosition(m_editorCamera->matrices.position);
            transformComponent.setRotationQuaternion(glm::quat_cast(glm::inverse(m_editorCamera->matrices.view)));
            transformComponent.setMoving(m_movedCamera);
            m_pathTracer->setActiveCamera(transformComponent, width, height);
            m_movedCamera = false;
            m_lastActiveCamera = nullptr;
        }

        // Store this frame's scene camera status for next frame’s comparison
        m_wasSceneCameraActive = isSceneCameraActive;
    }



}
