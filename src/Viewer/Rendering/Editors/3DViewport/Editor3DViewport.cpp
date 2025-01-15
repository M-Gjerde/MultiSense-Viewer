//
// Created by mgjer on 04/08/2024.
//

#include "Editor3DViewport.h"
#include "Editor3DLayer.h"

#include "Viewer/Rendering/Components/MaterialComponent.h"
#include "Viewer/Rendering/Editors/ArcballCamera.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"
#include "Viewer/Application/Application.h"
#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/Components/MeshComponent.h"

namespace VkRender {
    Editor3DViewport::Editor3DViewport(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("Editor3DLayer");
        addUIData<Editor3DViewportUI>();

        m_descriptorRegistry.createManager(DescriptorManagerType::Viewport3DTexture, m_context->vkDevice());
        m_editorCamera = std::make_shared<ArcballCamera>();
        m_editorCamera->setDefaultPosition({-90.0f, -60.0f}, 1.5f);

        m_sceneRenderer = m_context->getOrAddSceneRendererByUUID(uuid, m_createInfo);
        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
        textureCreateInfo.image = m_sceneRenderer->getOffscreenFramebuffer().resolvedImage;
        m_colorTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
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

    void Editor3DViewport::onEditorResize() {
        m_editorCamera = std::make_shared<ArcballCamera>(
                static_cast<float>(m_createInfo.width) / static_cast<float>(m_createInfo.height));
        m_editorCamera->setDefaultPosition({-90.0f, -60.0f}, 1.5f);
        m_activeScene = m_context->activeScene();

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
            auto &ci = m_sceneRenderer->getCreateInfo();
            ci.width = camera->parameters().width;
            ci.height = camera->parameters().height;
            m_sceneRenderer->resize(ci);
            onRenderSettingsChanged();
        } else {
            m_meshInstances.reset();
            m_meshInstances = nullptr;
            m_meshInstances = EditorUtils::setupMesh(m_context);
            m_sceneRenderer->setActiveCamera(m_editorCamera);
            auto &ci = m_sceneRenderer->getCreateInfo();
            ci.width = m_createInfo.width;
            ci.height = m_createInfo.height;
            m_sceneRenderer->resize(ci);
            onRenderSettingsChanged();
        }
    }

    void Editor3DViewport::onRenderSettingsChanged() {
        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);
        m_sceneRenderer = m_context->getOrAddSceneRendererByUUID(getUUID(), m_createInfo);
        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());

        if (imageUI->selectedImageType == OutputTextureImageType::Color) {
            textureCreateInfo.image = m_sceneRenderer->getOffscreenFramebuffer().resolvedImage;
        } else if (imageUI->selectedImageType == OutputTextureImageType::Depth) {
            textureCreateInfo.image = m_sceneRenderer->getOffscreenFramebuffer().resolvedDepthImage;
        }

        m_colorTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
    }

    void Editor3DViewport::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_sceneRenderer->onSceneLoad(scene);

        m_editorCamera = std::make_shared<ArcballCamera>(
                static_cast<float>(m_createInfo.width) / static_cast<float>(m_createInfo.height));
        m_editorCamera->setDefaultPosition({-90.0f, -60.0f}, 1.5f);
        m_activeScene = m_context->activeScene();

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
            auto &ci = m_sceneRenderer->getCreateInfo();
            ci.width = camera->parameters().width;
            ci.height = camera->parameters().height;
            m_sceneRenderer->resize(ci);
            onRenderSettingsChanged();
        } else {

            m_meshInstances.reset();
            m_meshInstances = nullptr;
            m_meshInstances = EditorUtils::setupMesh(m_context);
            m_sceneRenderer->setActiveCamera(m_editorCamera);
            auto &ci = m_sceneRenderer->getCreateInfo();
            ci.width = m_createInfo.width;
            ci.height = m_createInfo.height;
            m_sceneRenderer->resize(ci);
            onRenderSettingsChanged();
        }
    }

    static std::pair<float, float> computeScaleFactors(float textureAspect, float editorAspect) {
        float scaleX = 1.0f;
        float scaleY = 1.0f;

        if (editorAspect > textureAspect) {
            // Width is limiting factor, so adjust scale in X
            scaleX = textureAspect / editorAspect;
        } else {
            // Height is limiting factor, so adjust scale in Y
            scaleY = editorAspect / textureAspect;
        }

        return {scaleX, scaleY};
    }

    void Editor3DViewport::onUpdate() {

        m_activeScene = m_context->activeScene();
        if (!m_activeScene)
            return;

        updateActiveCamera();

        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);
        m_sceneRenderer->m_saveNextFrame = imageUI->saveNextFrame;


        auto frameIndex = m_context->currentFrameIndex();
        // Map and copy data to the global uniform buffer

        void *data;
        vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                    m_shaderSelectionBuffer[frameIndex]->m_memory, 0, sizeof(int32_t), 0, &data);
        memcpy(data, &imageUI->depthColorOption, sizeof(int32_t));
        vkUnmapMemory(m_context->vkDevice().m_LogicalDevice, m_shaderSelectionBuffer[frameIndex]->m_memory);

        m_sceneRenderer->update();
    }


    void Editor3DViewport::onRender(CommandBuffer &commandBuffer) {
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

    void Editor3DViewport::collectRenderCommands(
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
        std::vector descriptorWrites = {writeDescriptors[0], writeDescriptors[1]};
        VkDescriptorSet descriptorSet = m_descriptorRegistry.getManager(DescriptorManagerType::Viewport3DTexture).
                getOrCreateDescriptorSet(descriptorWrites);
        key.setLayouts[0] = m_descriptorRegistry.getManager(DescriptorManagerType::Viewport3DTexture).
                getDescriptorSetLayout();
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
        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);
        if (imageUI->reloadViewportShader) {
            m_pipelineManager.removePipeline(key);
        }
        // Create or retrieve the pipeline
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        renderPassInfo.debugName = "Editor3DViewport::";
        auto pipeline = m_pipelineManager.getOrCreatePipeline(key, renderPassInfo, m_context);
        // Create the render command
        RenderCommand command;
        command.pipeline = pipeline;
        command.meshInstance = m_meshInstances.get();
        command.descriptorSets[DescriptorManagerType::Viewport3DTexture] = descriptorSet; // Assign the descriptor set
        // Add to render group
        renderGroups[pipeline].push_back(command);
    }

    void Editor3DViewport::bindResourcesAndDraw(const CommandBuffer &commandBuffer, RenderCommand &command) {
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

    void Editor3DViewport::onMouseMove(const MouseButtons &mouse) {
        if (ui()->hovered && mouse.left && !ui()->resizeActive) {
            m_editorCamera->rotate(mouse.dx, mouse.dy);
        }
        else if (ui()->hovered && mouse.right && !ui()->resizeActive) {
            m_editorCamera->translate(mouse.dx, mouse.dy);
        }
    }

    void Editor3DViewport::onMouseScroll(float change) {
        if (ui()->hovered) {
            m_editorCamera->zoom((change > 0.0f) ? 0.95f : 1.05f);
        }
    }

    void Editor3DViewport::onKeyCallback(const Input &input) {
    }

    void Editor3DViewport::updateActiveCamera(){
        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);
        // By default, assume we are *not* using a scene camera this frame
        bool isSceneCameraActive = false;
        CameraComponent *sceneCameraToUse = nullptr;

        // If the user wants to render from a viewpoint, see if there's a valid scene camera
        if (imageUI->renderFromViewpoint) {
            auto view = m_activeScene->getRegistry().view<CameraComponent>();
            for (auto e: view) {
                Entity entity(e, m_activeScene.get());
                auto &cameraComponent = entity.getComponent<CameraComponent>();

                if (cameraComponent.renderFromViewpoint()) {
                    // Use the first camera found that can render from viewpoint
                    sceneCameraToUse = &cameraComponent;
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

            if (isNewCameraSelected || sceneCameraToUse->updateTrigger()) {
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

                // Resize the scene renderer
                auto &ci = m_sceneRenderer->getCreateInfo();
                ci.width = width;
                ci.height = height;
                m_sceneRenderer->resize(ci);

                onRenderSettingsChanged();
            }

            // Activate the scene camera
            m_sceneRenderer->setActiveCamera(sceneCameraToUse->camera);
            m_lastActiveCamera = sceneCameraToUse;
        } else {
            // No valid scene camera — revert to editor camera
            isSceneCameraActive = false;

            // Only do the mesh/aspect revert if we were using a scene camera last frame
            if (m_wasSceneCameraActive) {
                // Restore editor mesh and aspect ratio
                m_meshInstances.reset();
                m_meshInstances = EditorUtils::setupMesh(m_context);

                auto &ci = m_sceneRenderer->getCreateInfo();
                ci.width = m_createInfo.width;
                ci.height = m_createInfo.height;
                m_sceneRenderer->resize(ci);

                onRenderSettingsChanged();
            }

            m_sceneRenderer->setActiveCamera(m_editorCamera);
            m_lastActiveCamera = nullptr;
        }

        // Store this frame's scene camera status for next frame’s comparison
        m_wasSceneCameraActive = isSceneCameraActive;
    }
};
