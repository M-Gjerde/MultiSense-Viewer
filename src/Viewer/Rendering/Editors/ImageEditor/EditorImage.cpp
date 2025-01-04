//
// Created by mgjer on 18/08/2024.
//

#include "EditorImage.h"
#include "Viewer/Application/Application.h"
#include "EditorImageLayer.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"

namespace VkRender {
    EditorImage::EditorImage(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
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
        for (auto &frameIndex: m_shaderSelectionBuffer) {
            m_context->vkDevice().createBuffer(
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    frameIndex,
                    sizeof(int32_t), nullptr, "EditorImage:ShaderSelectionBuffer",
                    m_context->getDebugUtilsObjectNameFunction());
        }

    }

    void EditorImage::onEditorResize() {
        auto scene = m_context->activeScene();

        float width = m_createInfo.width;
        float height = m_createInfo.height;
        float editorAspect = static_cast<float>(m_createInfo.width) /
                             static_cast<float>(m_createInfo.height);
        PinholeCamera *pinholeCamera = nullptr;
        auto view = m_context->activeScene()->getRegistry().view<CameraComponent>();
        for (auto e: view) {
            Entity entity(e, m_context->activeScene().get());
            auto &cameraComponent = entity.getComponent<CameraComponent>();

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
            } else {
                scaleY = editorAspect / sceneCameraAspect;
            }
            // Reset and rebuild mesh
            m_meshInstances.reset();
            m_meshInstances = EditorUtils::setupMesh(m_context, scaleX, scaleY);
        }
        m_rayTracer = std::make_unique<VkRender::RT::RayTracer>(m_context, scene, width, height);
        m_colorTexture = EditorUtils::createEmptyTexture(width, height, VK_FORMAT_R8G8B8A8_UNORM, m_context);
    }

    void EditorImage::onFileDrop(const std::filesystem::path &path) {
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
        PinholeCamera *pinholeCamera = nullptr;
        auto view = m_context->activeScene()->getRegistry().view<CameraComponent>();
        for (auto e: view) {
            Entity entity(e, m_context->activeScene().get());
            auto &cameraComponent = entity.getComponent<CameraComponent>();

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
            } else {
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

    void EditorImage::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<EditorImageUI>(m_ui);

        if (imageUI->uploadScene) {
            m_rayTracer->upload(m_context->activeScene());

        }

        if (imageUI->render) {
            m_rayTracer->update(*imageUI);
            float* image = m_rayTracer->getImage();
            if (image) {
                uint32_t width = m_colorTexture->width();
                uint32_t height = m_colorTexture->height();
                std::vector<uint8_t> convertedImage(width * height * 4); // 4 channels: RGBA
                // Parallelize the conversion loop for better performance
                for (int i = 0; i < static_cast<int>(width * height); ++i) {
                    float intensity = image[i];
                    // Clamp and scale to [0, 255]
                    uint8_t value = static_cast<uint8_t>(std::clamp(intensity, 0.0f, 1.0f) * 255.0f);
                    convertedImage[i * 4 + 0] = value; // R
                    convertedImage[i * 4 + 1] = value; // G
                    convertedImage[i * 4 + 2] = value; // B
                    convertedImage[i * 4 + 3] = 255;   // A (fully opaque)
                }

                // Upload the texture (ensure the format matches)
                m_colorTexture->loadImage(convertedImage.data(), convertedImage.size());
            }
        }

    }

    void EditorImage::onMouseMove(const MouseButtons &mouse) {
    }

    void EditorImage::onMouseScroll(float change) {
    }

    void EditorImage::onRender(CommandBuffer &commandBuffer) {
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>> renderGroups;
        collectRenderCommands(renderGroups, commandBuffer.frameIndex);

        // Render each group
        for (auto &[pipeline, commands]: renderGroups) {
            pipeline->bind(commandBuffer);
            for (auto &command: commands) {
                // Bind resources and draw
                bindResourcesAndDraw(commandBuffer, command);
            }
        }
    }

    void EditorImage::collectRenderCommands(
            std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>> &renderGroups,
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

    void EditorImage::bindResourcesAndDraw(const CommandBuffer &commandBuffer, RenderCommand &command) {
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

}
