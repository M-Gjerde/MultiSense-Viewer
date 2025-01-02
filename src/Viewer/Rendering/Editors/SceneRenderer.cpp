//
// Created by mgjer on 30/09/2024.
//


#include "Viewer/Rendering/Editors/SceneRenderer.h"
#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/Rendering/Components/MeshComponent.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"

#include "Viewer/Application/Application.h"

#include "Viewer/Scenes/Entity.h"

#include "Viewer/Rendering/MeshInstance.h"


namespace VkRender {
    SceneRenderer::SceneRenderer(EditorCreateInfo& createInfo, UUID uuid) : Editor(createInfo, uuid) {
        m_renderToOffscreen = true;
        m_activeCamera = std::make_shared<BaseCamera>(m_createInfo.width / m_createInfo.height);
        descriptorRegistry.createManager(DescriptorManagerType::MVP, m_context->vkDevice());
        descriptorRegistry.createManager(DescriptorManagerType::Material, m_context->vkDevice());
        descriptorRegistry.createManager(DescriptorManagerType::DynamicCameraGizmo, m_context->vkDevice());

        m_meshResourceManager = std::make_unique<MeshResourceManager>(m_context);
    }

    void SceneRenderer::onEditorResize() {
    }

    SceneRenderer::~SceneRenderer() {
        m_entityRenderData.clear();
    }

    void SceneRenderer::onSceneLoad(std::shared_ptr<Scene> scene) {
        // Once we load a scene we need to create pipelines according to the objects specified in the scene.
        // For OBJModels we are alright with a default rendering pipeline (Phong lightining and stuff)
        // The pipelines also define memory handles between CPU and GPU. It makes more logical scenes if these attributes belong to the OBJModelComponent
        // But we need it accessed in the pipeline
        m_activeScene = m_context->activeScene();
        // TODO not clear what this does but it creates render reosurces of the editor was copied as part of a split operation
        auto view = m_activeScene->getRegistry().view<IDComponent>();
        for (auto e : view) {
            auto entity = Entity(e, m_activeScene.get());
            auto name = entity.getName();
            if (entity.hasComponent<MaterialComponent>()) {
                onComponentAdded(entity, entity.getComponent<MaterialComponent>());
            }
            if (entity.hasComponent<PointCloudComponent>()) {
                onComponentAdded(entity, entity.getComponent<PointCloudComponent>());
            }
            if (entity.hasComponent<MeshComponent>()) {
                onComponentAdded(entity, entity.getComponent<MeshComponent>());
            }
        }
    }

    void SceneRenderer::onUpdate() {
        m_activeScene = m_context->activeScene();
        if (!m_activeScene)
            return;
        auto view = m_activeScene->getRegistry().view<IDComponent>();
        for (auto entity : view) {
            updateGlobalUniformBuffer(m_context->currentFrameIndex(), Entity(entity, m_activeScene.get()));
        }

    }


    void SceneRenderer::onRender(CommandBuffer& commandBuffer) {
        collectRenderCommands(m_renderGroups, commandBuffer.getActiveFrameIndex());

        // Render each group
        for (auto& command : m_renderGroups) {
            bindResourcesAndDraw(commandBuffer, command);
        }
    }

    void SceneRenderer::bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand& command) {
        // Bind vertex buffers
        VkCommandBuffer cmdBuffer = commandBuffer.getActiveBuffer();
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, command.pipeline->pipeline()->getPipeline());
        for (auto& [index, descriptorSet] : command.descriptorSets) {
            vkCmdBindDescriptorSets(
                cmdBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                command.pipeline->pipeline()->getPipelineLayout(),
                static_cast<uint32_t>(index),
                1,
                &descriptorSet,
                0,
                nullptr
            );
        }

        // Bind vertex buffer
        bool usesVertexBuffers = command.meshInstance->vertexBuffer && command.meshInstance->usesVertexBuffers;
        if (usesVertexBuffers) {
            VkBuffer vertexBuffers[] = { command.meshInstance->vertexBuffer->m_buffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vertexBuffers, offsets);

            // Bind index buffer if present
            if (command.meshInstance->indexBuffer) {
                vkCmdBindIndexBuffer(cmdBuffer, command.meshInstance->indexBuffer->m_buffer, 0, VK_INDEX_TYPE_UINT32);
            }
        }

        // Issue the draw call
        if (command.meshInstance->indexBuffer && !command.meshInstance->SSBO) {
            // Indexed draw call with vertex buffers
            vkCmdDrawIndexed(cmdBuffer, command.meshInstance->indexCount, 1, 0, 0, 0);
        } else {
            // Non-indexed draw call or draw call without vertex buffers
            vkCmdDraw(cmdBuffer, command.meshInstance->drawCount, 1, 0, 0);
        }
    }

    void SceneRenderer::collectRenderCommands(
        std::vector<RenderCommand>& renderGroups, uint32_t frameIndex) {
        auto view = m_activeScene->getRegistry().view<MeshComponent, TransformComponent>();

        m_renderGroups.clear();
        m_renderGroups.reserve(view.size_hint());
        for (auto e : view) {
            Entity entity(e, m_activeScene.get());

            // Initialize a flag to determine if we should skip this entity
            bool skipEntity = false;

            // Start with the current entity
            Entity current = entity;
            // Traverse up the parent hierarchy
            while (current.getParent()) {
                // Move to the parent entity
                current = current.getParent();
                // Check if the parent has both GroupComponent and VisibilityComponent
                if (current.hasComponent<GroupComponent>() && current.hasComponent<VisibleComponent>()) {
                    // Retrieve the VisibilityComponent
                    auto& visibility = current.getComponent<VisibleComponent>();
                    // If visibility is set to false, mark to skip this entity
                    if (!visibility.visible) {
                        skipEntity = true;
                        break; // No need to check further ancestors
                    }
                }
            }
            // If an ancestor with visible == false was found, skip to the next entity
            if (skipEntity) {
                continue;
            }

            std::string tag = entity.getName();
            UUID uuid = entity.getUUID();
            auto& meshComponent = entity.getComponent<MeshComponent>();
            std::unordered_map<DescriptorManagerType, VkDescriptorSet> descriptorSets; // Add the descriptor set here
            std::unordered_map<DescriptorManagerType, std::vector<VkWriteDescriptorSet>> descriptorWritesTracker; // TODO remove
            PipelineKey key = {};
            key.setLayouts.resize(3);

            auto& renderData = m_entityRenderData[entity.getUUID()];

            {
                auto& writes = descriptorWritesTracker[DescriptorManagerType::MVP];
                writes.resize(2);
                writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[0].dstBinding = 0;
                writes[0].dstArrayElement = 0;
                writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writes[0].descriptorCount = 1;
                writes[0].pBufferInfo = &renderData.cameraBuffer[frameIndex]->m_descriptorBufferInfo;
                writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[1].dstBinding = 1;
                writes[1].dstArrayElement = 0;
                writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writes[1].descriptorCount = 1;
                writes[1].pBufferInfo = &renderData.modelBuffer[frameIndex]->m_descriptorBufferInfo;
                VkDescriptorSet mvpDescriptorSet = descriptorRegistry.getManager(DescriptorManagerType::MVP).
                                                                      getOrCreateDescriptorSet(writes);
                descriptorSets[DescriptorManagerType::MVP] = mvpDescriptorSet;

                key.setLayouts[0] = descriptorRegistry.getManager(DescriptorManagerType::MVP).getDescriptorSetLayout();
            }
            // Use default descriptor set layout

            std::shared_ptr<MeshData> meshData = m_meshManager.getMeshData(meshComponent);
            if (!meshData)
                continue;
            // Update meshData
            std::shared_ptr<MeshInstance> meshInstance = m_meshResourceManager->getMeshInstance(meshComponent.getCacheIdentifier(), meshData, meshComponent.meshDataType());
            if (!meshInstance) {
                continue;
            }
            key.topology = meshInstance->topology;
            key.polygonMode = meshComponent.polygonMode();
            // Check if the entity has a MaterialComponent
            std::shared_ptr<MaterialInstance> materialInstance = nullptr;
            if (entity.hasComponent<MaterialComponent>()) {
                auto& materialComponent = entity.getComponent<MaterialComponent>();
                auto materialIt = m_materialInstances.find(uuid);
                if (materialIt == m_materialInstances.end()) {
                    materialInstance = initializeMaterial(entity, materialComponent);
                    m_materialInstances[uuid] = materialInstance;
                }
                else {
                    materialInstance = materialIt->second;
                }
                key.vertexShaderName = materialComponent.vertexShaderName;
                key.fragmentShaderName = materialComponent.fragmentShaderName;
                key.renderMode = materialInstance->renderMode;
                key.materialPtr = reinterpret_cast<uint64_t*>(materialInstance.get());

                auto& writes = descriptorWritesTracker[DescriptorManagerType::Material];
                writes.resize(2);
                writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[0].dstBinding = 0;
                writes[0].dstArrayElement = 0;
                writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writes[0].descriptorCount = 1;
                writes[0].pBufferInfo = &renderData.materialBuffer[frameIndex]->m_descriptorBufferInfo;
                writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[1].dstBinding = 1;
                writes[1].dstArrayElement = 0;
                writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[1].descriptorCount = 1;
                writes[1].pImageInfo = &materialInstance->baseColorTexture->getDescriptorInfo();
                VkDescriptorSet materialDescriptorSet = descriptorRegistry.getManager(DescriptorManagerType::Material).getOrCreateDescriptorSet(writes);
                descriptorSets[DescriptorManagerType::Material] = materialDescriptorSet;
            }
            key.setLayouts[1] = descriptorRegistry.getManager(DescriptorManagerType::Material).getDescriptorSetLayout();

            std::vector<VkVertexInputBindingDescription> vertexInputBinding = {
                {0, sizeof(VkRender::Vertex), VK_VERTEX_INPUT_RATE_VERTEX}
            };

            std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3},
                {2, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 6},
                {3, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 8},
                {4, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 10},
            };
            key.vertexInputBindingDescriptions = vertexInputBinding;
            key.vertexInputAttributes = vertexInputAttributes;

            if (meshData->isDynamic) {
                key.vertexShaderName = "CameraGizmo.vert";
                key.fragmentShaderName = "default.frag";
                key.vertexInputBindingDescriptions.clear();
                key.vertexInputAttributes.clear();
                key.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

                auto& writes = descriptorWritesTracker[DescriptorManagerType::DynamicCameraGizmo];
                writes.resize(2);
                writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[0].dstBinding = 0;
                writes[0].dstArrayElement = 0;
                writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[0].descriptorCount = 1;
                writes[0].pBufferInfo = &meshInstance->vertexBuffer->m_descriptorBufferInfo;
                writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[1].dstBinding = 1;
                writes[1].dstArrayElement = 0;
                writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[1].descriptorCount = 1;
                writes[1].pBufferInfo = &meshInstance->indexBuffer->m_descriptorBufferInfo;
                VkDescriptorSet dynamicCameraDescriptorSet = descriptorRegistry.getManager(
                    DescriptorManagerType::DynamicCameraGizmo).getOrCreateDescriptorSet(writes);
                descriptorSets[DescriptorManagerType::DynamicCameraGizmo] = dynamicCameraDescriptorSet;


            }
            key.setLayouts[2] = descriptorRegistry.getManager(DescriptorManagerType::DynamicCameraGizmo).getDescriptorSetLayout();
            // Create or retrieve the pipeline
            RenderPassInfo renderPassInfo{};
            renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
            renderPassInfo.renderPass = m_renderPass->getRenderPass();
            renderPassInfo.debugName = "SceneRenderer::";

            auto pipeline = m_pipelineManager.getOrCreatePipeline(key, renderPassInfo, m_context);
            // Create the render command

            RenderCommand command;
            command.descriptorSets = descriptorSets;
            command.entity = entity;
            command.pipeline = pipeline;
            command.meshInstance = meshInstance.get();
            command.materialInstance = materialInstance.get(); // May be null if no material is attached
            // Add to render group
            renderGroups.emplace_back(command);
        }
    }

    void SceneRenderer::onComponentAdded(Entity entity, MeshComponent& meshComponent) {
        // Check if I readd a meshcomponent then we should destroy the renderresources attached to it:
        m_entityRenderData[entity.getUUID()].cameraBuffer.resize(m_context->swapChainBuffers().size());
        m_entityRenderData[entity.getUUID()].modelBuffer.resize(m_context->swapChainBuffers().size());
        // Create attachable UBO buffers and such
        for (int frameIndex = 0; frameIndex < m_context->swapChainBuffers().size(); ++frameIndex) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].cameraBuffer[frameIndex],
                sizeof(GlobalUniformBufferObject), nullptr, "SceneRenderer:MeshComponent:Camera",
                m_context->getDebugUtilsObjectNameFunction());
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].modelBuffer[frameIndex],
                sizeof(glm::mat4), nullptr, "SceneRenderer:MeshComponent:Model",
                m_context->getDebugUtilsObjectNameFunction());
        }
    }

    void SceneRenderer::onComponentRemoved(Entity entity, MeshComponent& meshComponent) {
        if (m_entityRenderData.contains(entity.getUUID())) {
            m_entityRenderData[entity.getUUID()].cameraBuffer.clear();
            m_entityRenderData[entity.getUUID()].modelBuffer.clear();
        }
    }

    void SceneRenderer::onComponentUpdated(Entity entity, MeshComponent& meshComponent) {

    }

    void SceneRenderer::onComponentAdded(Entity entity, MaterialComponent& materialComponent) {
        // Check if I readd a meshcomponent then we should destroy the renderresources attached to it:
        if (m_materialInstances.contains(entity.getUUID())) {
            m_materialInstances.erase(entity.getUUID());
        }
        m_entityRenderData[entity.getUUID()].materialBuffer.resize(m_context->swapChainBuffers().size());
        // Create attachable UBO buffers and such
        for (int i = 0; i < m_context->swapChainBuffers().size(); ++i) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].materialBuffer[i],
                sizeof(MaterialBufferObject), nullptr, "SceneRenderer:MaterialComponent",
                m_context->getDebugUtilsObjectNameFunction());;
        }
    }

    void SceneRenderer::onComponentRemoved(Entity entity, MaterialComponent& materialComponent) {
        if (m_materialInstances.contains(entity.getUUID())) {
            m_materialInstances.erase(entity.getUUID());
        }
    }

    void SceneRenderer::onComponentUpdated(Entity entity, MaterialComponent& materialComponent) {
        // add a video source if selected
        if (m_materialInstances.contains(
            entity.getUUID())) {
            // TODO look into just replacing what changed instead of erasing, triggering a new pipeline creation. However, the cost for recreating everything in a material is very small
            m_materialInstances.erase(entity.getUUID());
        }
    }

    void SceneRenderer::onComponentAdded(Entity entity, PointCloudComponent& pointCloudComponent) {
        // Check if I readd a meshcomponent then we should destroy the renderresources attached to it:
        m_entityRenderData[entity.getUUID()].cameraBuffer.resize(m_context->swapChainBuffers().size());
        m_entityRenderData[entity.getUUID()].modelBuffer.resize(m_context->swapChainBuffers().size());
        m_entityRenderData[entity.getUUID()].pointCloudBuffer.resize(m_context->swapChainBuffers().size());
        // Create attachable UBO buffers and such
        for (int i = 0; i < m_context->swapChainBuffers().size(); ++i) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].cameraBuffer[i],
                sizeof(GlobalUniformBufferObject), nullptr, "SceneRenderer:PointCloudComponent:Camera",
                m_context->getDebugUtilsObjectNameFunction());
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].modelBuffer[i],
                sizeof(glm::mat4), nullptr, "SceneRenderer:PointCloudComponent:Model",
                m_context->getDebugUtilsObjectNameFunction());
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].pointCloudBuffer[i],
                sizeof(PointCloudUBO), nullptr, "SceneRenderer:PointCloudComponent:PC",
                m_context->getDebugUtilsObjectNameFunction());
        }
    }

    void SceneRenderer::onComponentRemoved(Entity entity, PointCloudComponent& pointCloudComponent) {
    }

    void SceneRenderer::onComponentUpdated(Entity entity, PointCloudComponent& pointCloudComponent) {
    }

    void SceneRenderer::updateGlobalUniformBuffer(uint32_t frameIndex, Entity entity) {
        // Get the active camera entity
        // Compute view and projection matrices
        if (entity.hasComponent<MeshComponent>()) {
            GlobalUniformBufferObject globalUBO = {};
            auto activeCameraPtr = m_activeCamera.lock(); // Lock to get shared_ptr
            if (activeCameraPtr) {
                globalUBO.view = activeCameraPtr->matrices.view;
                globalUBO.projection = activeCameraPtr->matrices.projection;
                globalUBO.cameraPosition = activeCameraPtr->matrices.position;
            }

            // Map and copy data to the global uniform buffer
            void* data;
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].cameraBuffer[frameIndex]->m_memory, 0, sizeof(globalUBO),
                        0,
                        &data);
            memcpy(data, &globalUBO, sizeof(globalUBO));
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          m_entityRenderData[entity.getUUID()].cameraBuffer[frameIndex]->m_memory);

        }
        if (entity.hasComponent<TransformComponent>() && entity.hasComponent<MeshComponent>()) {
            void* data;
            auto& transformComponent = m_activeScene->getRegistry().get<TransformComponent>(entity);
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].modelBuffer[frameIndex]->m_memory, 0, VK_WHOLE_SIZE, 0,
                        &data);
            auto* modelMatrices = reinterpret_cast<glm::mat4*>(data);
            *modelMatrices = transformComponent.getTransform();
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          m_entityRenderData[entity.getUUID()].modelBuffer[frameIndex]->m_memory);
        }
        if (entity.hasComponent<MaterialComponent>() && !m_entityRenderData[entity.getUUID()].materialBuffer.empty()) {
            auto& material = entity.getComponent<MaterialComponent>();
            MaterialBufferObject matUBO = {};
            matUBO.baseColor = material.albedo;
            matUBO.metallic = material.specular;
            matUBO.roughness = material.diffuse;
            matUBO.emissiveFactor = glm::vec4(material.emission);
            void* data;
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].materialBuffer[frameIndex]->m_memory, 0,
                        sizeof(MaterialBufferObject), 0,
                        &data);
            memcpy(data, &matUBO, sizeof(MaterialBufferObject));
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          m_entityRenderData[entity.getUUID()].materialBuffer[frameIndex]->m_memory);
        }
    }


    std::shared_ptr<MaterialInstance> SceneRenderer::initializeMaterial(
        Entity entity, const MaterialComponent& materialComponent) {
        auto materialInstance = std::make_shared<MaterialInstance>();
        if (std::filesystem::exists(materialComponent.albedoTexturePath)) {
            materialInstance->baseColorTexture = EditorUtils::createTextureFromFile(materialComponent.albedoTexturePath,
                m_context);
        }
        else {
            materialInstance->baseColorTexture = EditorUtils::createEmptyTexture(1280, 720, VK_FORMAT_R8G8B8A8_UNORM,
                m_context, VMA_MEMORY_USAGE_GPU_ONLY, true);
        }
        Log::Logger::getInstance()->info("Created Material for Entity: {}", entity.getName());
        return materialInstance;
    }

}
