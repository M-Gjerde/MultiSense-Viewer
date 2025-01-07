//
// Created by mgjer on 18/08/2024.
//

#include "EditorImage.h"
#include "Viewer/Application/Application.h"
#include "EditorImageLayer.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"

#include <OpenImageDenoise/oidn.hpp>

namespace VkRender {
    EditorImage::EditorImage(EditorCreateInfo& createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorImageLayer");
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUIData<EditorImageUI>();

        diffRenderEntry = std::make_unique<VkRender::DR::DiffRenderEntry>();
        diffRenderEntry->setup();

        m_descriptorRegistry.createManager(DescriptorManagerType::Viewport3DTexture, m_context->vkDevice());

        m_colorTexture = EditorUtils::createEmptyTexture(1280, 720, VK_FORMAT_R8G8B8A8_UNORM, m_context,
                                                         VMA_MEMORY_USAGE_GPU_ONLY, true);
        m_shaderSelectionBuffer.resize(m_context->swapChainBuffers().size());
        for (auto& frameIndex : m_shaderSelectionBuffer) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                frameIndex,
                sizeof(EditorImageUI::ShaderSelection), nullptr, "EditorImage:ShaderSelectionBuffer",
                m_context->getDebugUtilsObjectNameFunction());
        }
    }

    void EditorImage::onEditorResize() {
        auto scene = m_context->activeScene();

        float width = m_createInfo.width;
        float height = m_createInfo.height;
        float editorAspect = static_cast<float>(m_createInfo.width) /
            static_cast<float>(m_createInfo.height);
        PinholeCamera* pinholeCamera = nullptr;
        auto view = m_context->activeScene()->getRegistry().view<CameraComponent>();
        for (auto e : view) {
            Entity entity(e, m_context->activeScene().get());
            auto& cameraComponent = entity.getComponent<CameraComponent>();

            if (cameraComponent.cameraType == CameraComponent::PINHOLE) {
                // Use the first camera found that can render from viewpoint
                pinholeCamera = cameraComponent.getPinholeCamera().get();
                break;
            }
        }
        if (pinholeCamera) {
            float sceneCameraAspect = pinholeCamera->m_width / pinholeCamera->m_height;
            width = pinholeCamera->m_width;
            height = pinholeCamera->m_height;

            float scaleX = 1.0f, scaleY = 1.0f;
            if (editorAspect > sceneCameraAspect) {
                scaleX = sceneCameraAspect / editorAspect;
            }
            else {
                scaleY = editorAspect / sceneCameraAspect;
            }
            // Reset and rebuild mesh
            m_meshInstances.reset();
            m_meshInstances = EditorUtils::setupMesh(m_context, scaleX, scaleY);
        }
        m_rayTracer = std::make_unique<VkRender::RT::RayTracer>(m_context, scene, width, height);
        m_colorTexture = EditorUtils::createEmptyTexture(width, height, VK_FORMAT_R8G8B8A8_UNORM, m_context);
    }

    void EditorImage::onFileDrop(const std::filesystem::path& path) {
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension == ".png" || extension == ".jpg") {
            m_colorTexture = EditorUtils::createTextureFromFile(path, m_context);
        }
    }


    void EditorImage::onSceneLoad(std::shared_ptr<Scene> scene) {
        float width = m_createInfo.width;
        float height = m_createInfo.height;
        float editorAspect = static_cast<float>(m_createInfo.width) /
            static_cast<float>(m_createInfo.height);
        PinholeCamera* pinholeCamera = nullptr;
        auto view = m_context->activeScene()->getRegistry().view<CameraComponent>();
        for (auto e : view) {
            Entity entity(e, m_context->activeScene().get());
            auto& cameraComponent = entity.getComponent<CameraComponent>();

            if (cameraComponent.cameraType == CameraComponent::PINHOLE) {
                // Use the first camera found that can render from viewpoint
                pinholeCamera = cameraComponent.getPinholeCamera().get();
                break;
            }
        }
        if (pinholeCamera) {
            float sceneCameraAspect = pinholeCamera->m_width / pinholeCamera->m_height;
            width = pinholeCamera->m_width;
            height = pinholeCamera->m_height;

            float scaleX = 1.0f, scaleY = 1.0f;
            if (editorAspect > sceneCameraAspect) {
                scaleX = sceneCameraAspect / editorAspect;
            }
            else {
                scaleY = editorAspect / sceneCameraAspect;
            }
            // Reset and rebuild mesh
            m_meshInstances.reset();
            m_meshInstances = EditorUtils::setupMesh(m_context, scaleX, scaleY);
        }
        m_rayTracer = std::make_unique<VkRender::RT::RayTracer>(m_context, scene, width, height);
        m_colorTexture = EditorUtils::createEmptyTexture(width, height, VK_FORMAT_R8G8B8A8_UNORM, m_context);
    }


    void EditorImage::onPipelineReload() {
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

        // Retrieve the denoised image
        output.resize(imageSize);
        std::memcpy(output.data(), outputBuffer.getData(), output.size() * sizeof(float));

    }

    void EditorImage::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<EditorImageUI>(m_ui);

        auto frameIndex = m_context->currentFrameIndex();

        void* data;
        vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                    m_shaderSelectionBuffer[frameIndex]->m_memory, 0, sizeof(EditorImageUI::ShaderSelection), 0, &data);
        memcpy(data, &imageUI->shaderSelection, sizeof(EditorImageUI::ShaderSelection));
        vkUnmapMemory(m_context->vkDevice().m_LogicalDevice, m_shaderSelectionBuffer[frameIndex]->m_memory);

        if (imageUI->uploadScene) {
            m_rayTracer->upload(m_context->activeScene());
        }

        if (imageUI->render) {
            m_rayTracer->update(*imageUI, m_context->activeScene());
            float* image = m_rayTracer->getImage();
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
            uint32_t width = m_colorTexture->width();
            uint32_t height = m_colorTexture->height();
            float* image = m_rayTracer->getImage();
            std::filesystem::path filename = "cornell.pfm";
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
        }
    }

    void EditorImage::onMouseMove(const MouseButtons& mouse) {
    }

    void EditorImage::onMouseScroll(float change) {
    }

    void EditorImage::onRender(CommandBuffer& commandBuffer) {
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

    void EditorImage::collectRenderCommands(
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
        auto imageUI = std::dynamic_pointer_cast<EditorImageUI>(m_ui);

        if (imageUI->uploadScene) {
            m_descriptorRegistry.getManager(DescriptorManagerType::Viewport3DTexture).freeDescriptorSets();
        }
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
        key.fragmentShaderName = "EditorImageViewportTexture.frag";
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
        renderPassInfo.debugName = "EditorImage::";
        auto pipeline = m_pipelineManager.getOrCreatePipeline(key, renderPassInfo, m_context);
        // Create the render command
        RenderCommand command;
        command.pipeline = pipeline;
        command.meshInstance = m_meshInstances.get();
        command.descriptorSets[DescriptorManagerType::Viewport3DTexture] = descriptorSet; // Assign the descriptor set
        // Add to render group
        renderGroups[pipeline].push_back(command);
    }

    void EditorImage::bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand& command) {
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
}
