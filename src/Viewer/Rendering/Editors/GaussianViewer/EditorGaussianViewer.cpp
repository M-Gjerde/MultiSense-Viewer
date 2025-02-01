//
// Created by magnus on 8/15/24.
//

#include "EditorGaussianViewer.h"

#include <multisense_viewer/src/Viewer/Rendering/Editors/CommonEditorFunctions.h>

#include "EditorGaussianViewerLayer.h"

#include "Viewer/Application/Application.h"

namespace VkRender {

    EditorGaussianViewer::EditorGaussianViewer(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid),
                                                                                          gaussianRenderer2D(
                                                                                                  m_deviceSelector.getQueue()),
                                                                                          gaussianRenderer3D(
                                                                                                  m_deviceSelector.getQueue()) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("EditorGaussianViewerLayer");
        addUIData<EditorGaussianViewerUI>();

        m_descriptorRegistry.createManager(DescriptorManagerType::Viewport3DTexture, m_context->vkDevice());
        m_editorCamera = std::make_shared<Camera>(m_createInfo.width, m_createInfo.height);

        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height,
                                                         VK_FORMAT_B8G8R8A8_UNORM, m_context);

        m_shaderSelectionBuffer.resize(m_context->swapChainBuffers().size());
        for (auto &frameIndex: m_shaderSelectionBuffer) {
            m_context->vkDevice().createBuffer(
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    frameIndex,
                    sizeof(int32_t), nullptr, "Editor3DViewport:ShaderSelectionBuffer",
                    m_context->getDebugUtilsObjectNameFunction());
        }
    }

    void EditorGaussianViewer::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_editorCamera = std::make_shared<Camera>(m_createInfo.width, m_createInfo.height);
        m_activeScene = m_context->activeScene();

        m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height,
                                                         VK_FORMAT_B8G8R8A8_UNORM, m_context);

    }

    void EditorGaussianViewer::onEditorResize() {
        m_editorCamera->setCameraResolution(m_createInfo.width, m_createInfo.height);
        m_editorCamera->setPerspective(static_cast<float>(m_createInfo.width) / m_createInfo.height);

        m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height,
                                                         VK_FORMAT_B8G8R8A8_UNORM, m_context);


    }


    void EditorGaussianViewer::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<EditorGaussianViewerUI>(m_ui);
        if (imageUI->useImageFrom3DViewport) {

        }

        auto &e = m_context->getSelectedEntity();
        if (e && e.hasComponent<CameraComponent>()) {
            auto &camera = e.getComponent<CameraComponent>();
            if (camera.isActiveCamera()) {
                // If the selected entity has a camera with isActiveCamera, use it
                //gaussianRenderer3D.setActiveCamera(camera.camera);
                m_lastActiveCamera = &camera; // Update the last active camera
            }
        } else if (m_lastActiveCamera && m_lastActiveCamera->isActiveCamera()) {
            // Use the last active camera if it still has isActiveCamera enabled
            //gaussianRenderer3D.setActiveCamera(m_lastActiveCamera->camera);
        }


        auto frameIndex = m_context->currentFrameIndex();
        void *data;
        vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                    m_shaderSelectionBuffer[frameIndex]->m_memory, 0, sizeof(int32_t), 0, &data);
        memcpy(data, &imageUI->colorOption, sizeof(int32_t));
        vkUnmapMemory(m_context->vkDevice().m_LogicalDevice, m_shaderSelectionBuffer[frameIndex]->m_memory);

    }

    void EditorGaussianViewer::onRender(CommandBuffer &commandBuffer) {
        auto scene = m_context->activeScene();

        auto imageUI = std::dynamic_pointer_cast<EditorGaussianViewerUI>(m_ui);

        bool updateRender = false;
        m_activeScene->getRegistry().view<GaussianComponent>().each([&](auto entity, GaussianComponent &gaussianComp) {
            auto e = Entity(entity, m_activeScene.get());
            if (e.getComponent<GaussianComponent>().addToRenderer)
                updateRender = false;
        });

        if (imageUI->render3dgsImage || updateRender) {
            if (imageUI->radioButton == 0) {
                //gaussianRenderer3D.render(scene, m_colorTexture);

            } else {
                //gaussianRenderer2D.render(scene, m_colorTexture);

            }
        }


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

    void EditorGaussianViewer::collectRenderCommands(
            std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>> &renderGroups,
            uint32_t frameIndex) {
        if (!m_meshInstances) {
            m_meshInstances = setupMesh();
            Log::Logger::getInstance()->info("Created MeshInstance for 3DViewport");
        }
        if (!m_meshInstances)
            return;

        PipelineKey key = {};
        key.setLayouts.resize(1);
        if (m_ui->resizeActive) {
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

        std::vector<VkWriteDescriptorSet> descriptorWrites = {writeDescriptors[0], writeDescriptors[1]};
        // Get or create the descriptor set using the DescriptorSetManager
        VkDescriptorSet descriptorSet = m_descriptorRegistry.getManager(DescriptorManagerType::Viewport3DTexture).getOrCreateDescriptorSet(descriptorWrites);
        key.setLayouts[0] = m_descriptorRegistry.getManager(DescriptorManagerType::Viewport3DTexture).getDescriptorSetLayout();
        // Use default descriptor set layout
        key.vertexShaderName = "default2D.vert";
        key.fragmentShaderName = "default2D.frag";
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
        auto pipeline = m_pipelineManager.getOrCreatePipeline(key, renderPassInfo, m_context);
        // Create the render command
        RenderCommand command;
        command.pipeline = pipeline;
        command.meshInstance = m_meshInstances.get();
        command.descriptorSets[DescriptorManagerType::Viewport3DTexture] = descriptorSet; // Assign the descriptor set

        // Add to render group
        renderGroups[pipeline].push_back(command);
    }

    void EditorGaussianViewer::bindResourcesAndDraw(const CommandBuffer &commandBuffer, RenderCommand &command) {
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

    void EditorGaussianViewer::onMouseMove(const MouseButtons &mouse) {
        if (ui()->hovered && mouse.left && !ui()->resizeActive) {
            m_editorCamera->rotate(mouse.dx, mouse.dy);
        }
    }

    void EditorGaussianViewer::onMouseScroll(float change) {
        if (ui()->hovered)
            m_editorCamera->setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
    }

    void EditorGaussianViewer::onKeyCallback(const Input &input) {

        m_editorCamera->keys.up = input.keys.up;
        m_editorCamera->keys.down = input.keys.down;
        m_editorCamera->keys.left = input.keys.left;
        m_editorCamera->keys.right = input.keys.right;

    }

    std::shared_ptr<MeshInstance> EditorGaussianViewer::setupMesh() {
        std::vector<VkRender::ImageVertex> vertices = {
                // Bottom-left corner
                {glm::vec2{-1.0f, -1.0f}, glm::vec2{0.0f, 0.0f}},
                // Bottom-right corner
                {glm::vec2{1.0f, -1.0f},  glm::vec2{1.0f, 0.0f}},
                // Top-right corner
                {glm::vec2{1.0f, 1.0f},   glm::vec2{1.0f, 1.0f}},
                // Top-left corner
                {glm::vec2{-1.0f, 1.0f},  glm::vec2{0.0f, 1.0f}}
        };
        // Define the indices for two triangles that make up the quad
        std::vector<uint32_t> indices = {
                0, 1, 2, // First triangle (bottom-left to top-right)
                2, 3, 0 // Second triangle (top-right to bottom-left)
        };

        auto meshInstance = std::make_shared<MeshInstance>();

        meshInstance->vertexCount = vertices.size();
        meshInstance->indexCount = indices.size();
        VkDeviceSize vertexBufferSize = vertices.size() * sizeof(VkRender::ImageVertex);
        VkDeviceSize indexBufferSize = indices.size() * sizeof(uint32_t);

        struct StagingBuffer {
            VkBuffer buffer;
            VkDeviceMemory memory;
        } vertexStaging{}, indexStaging{};

        // Create staging buffers
        // Vertex m_DataPtr
        CHECK_RESULT(m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertexBufferSize,
                &vertexStaging.buffer,
                &vertexStaging.memory,
                vertices.data()))
        // Index m_DataPtr
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_context->vkDevice().createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    indexBufferSize,
                    &indexStaging.buffer,
                    &indexStaging.memory,
                    indices.data()))
        }

        CHECK_RESULT(m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                meshInstance->vertexBuffer, vertexBufferSize, nullptr, "SceneRenderer:InitializeMesh:Vertex",
                m_context->getDebugUtilsObjectNameFunction()));
        // Index buffer
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_context->vkDevice().createBuffer(
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    meshInstance->indexBuffer,
                    indexBufferSize, nullptr, "SceneRenderer:InitializeMesh:Index",
                    m_context->getDebugUtilsObjectNameFunction()));
        }

        // Copy from staging buffers
        VkCommandBuffer copyCmd = m_context->vkDevice().createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkBufferCopy copyRegion = {};
        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, meshInstance->vertexBuffer->m_buffer, 1, &copyRegion);
        if (indexBufferSize > 0) {
            copyRegion.size = indexBufferSize;
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, meshInstance->indexBuffer->m_buffer, 1, &copyRegion);
        }
        m_context->vkDevice().flushCommandBuffer(copyCmd, m_context->vkDevice().m_TransferQueue, true);

        vkDestroyBuffer(m_context->vkDevice().m_LogicalDevice, vertexStaging.buffer, nullptr);
        vkFreeMemory(m_context->vkDevice().m_LogicalDevice, vertexStaging.memory, nullptr);

        if (indexBufferSize > 0) {
            vkDestroyBuffer(m_context->vkDevice().m_LogicalDevice, indexStaging.buffer, nullptr);
            vkFreeMemory(m_context->vkDevice().m_LogicalDevice, indexStaging.memory, nullptr);
        }

        return meshInstance;
    }
}