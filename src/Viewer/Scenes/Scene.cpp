//
// Created by magnus on 8/13/24.
//

#include "Viewer/Scenes/Entity.h"

#include "Viewer/Scenes/Scene.h"

#include <glm/gtx/quaternion.hpp>

#include <Viewer/Rendering/Components/GaussianComponent.h>

#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/Components/MeshComponent.h"
#include "Viewer/Rendering/Components/ImageComponent.h"
#include "Viewer/Application/Application.h"
#include "Viewer/Rendering/Components/PointCloudComponent.h"

namespace VkRender {
    Scene::Scene(VkRender::Application* context) {
        m_context = context;
    }


    void Scene::update() {
        auto cameraView = m_registry.view<CameraComponent>();
        for (auto e : cameraView) {
            auto entity = Entity(e, this);
            auto cameraComponent = entity.getComponent<CameraComponent>();
            auto& transform = entity.getComponent<TransformComponent>();
            cameraComponent.camera->updateViewMatrix(transform.getTransform());
            cameraComponent.camera->updateProjectionMatrix();

            if (entity.hasComponent<MaterialComponent>()) {
                auto& material = entity.getComponent<MaterialComponent>();
                if (cameraComponent.isActiveCamera()) {
                    material.albedo = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
                }
                else {
                    material.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
                }
            }
        }

        auto gaussianView = m_registry.view<GaussianComponent2DGS>();
        for (auto e : gaussianView) {
            // Wrap the entity to use our helper functions.
            Entity gaussianEntity(e, this);
            auto& gaussianComp = gaussianEntity.getComponent<GaussianComponent2DGS>();
            // For this example, we use the first position and normal from the component.
            size_t numGaussians = gaussianComp.size();
            for (size_t i = 0; i < numGaussians; i++) {
                glm::vec3 position = gaussianComp.positions[i];
                glm::vec3 normal = gaussianComp.normals[i];
                glm::vec2 scale = gaussianComp.scales[i];
                // Create a unique name for the mesh entity associated with this gaussian.
                std::string entityName = "GaussianEntity_" + std::to_string(static_cast<uint32_t>(i));
                // Get or create the entity with the given name.
                Entity meshEntity = getOrCreateEntityByName(entityName);
                // Only add the MeshComponent if it doesn't already exist.
                if (!meshEntity.hasComponent<MeshComponent>()) {
                    meshEntity.addComponent<MeshComponent>(
                        OBJ_FILE,
                        "/home/magnus/CLionProjects/multisense_viewer/Resources/models-repository/disk.obj"
                    );
                }
                // Only add the MaterialComponent if it doesn't already exist.
                if (!meshEntity.hasComponent<MaterialComponent>()) {
                    meshEntity.addComponent<MaterialComponent>();
                }
                // Ensure that a TransformComponent exists.
                if (!meshEntity.hasComponent<TransformComponent>()) {
                    meshEntity.addComponent<TransformComponent>();
                }
                if (!meshEntity.hasComponent<TemporaryComponent>()) {
                    meshEntity.addComponent<TemporaryComponent>();
                }

                auto& transform = meshEntity.getComponent<TransformComponent>();
                // Update the transform's position.
                transform.setPosition(position);
                transform.setScale({scale.x, scale.y, 1.0f});
                // Compute the quaternion rotation so that the local up vector (0,1,0)
                // aligns with the Gaussian normal.
                glm::vec3 localUp(0.0f, 0.0f, 1.0f);
                glm::quat rotation = glm::rotation(localUp, normal);
                transform.setRotationQuaternion(rotation);
            }
        }
    }


    CameraComponent* Scene::getActiveCamera() {
        auto view = m_registry.view<CameraComponent>();
        CameraComponent* activeCamera = nullptr;
        // First pass: iterate through all cameras.
        // Keep updating activeCamera so that the last camera found with isActiveCamera() true wins.
        for (auto entityID : view) {
            Entity entity(entityID, this);
            auto& cameraComponent = entity.getComponent<CameraComponent>();
            if (cameraComponent.isActiveCamera()) {
                activeCamera = &cameraComponent;
            }
        }
        return activeCamera;
    }

    Entity Scene::getActiveCameraEntity() {
        Entity entity;
        auto view = m_registry.view<CameraComponent>();
        CameraComponent* activeCamera = nullptr;
        // First pass: iterate through all cameras.
        // Keep updating activeCamera so that the last camera found with isActiveCamera() true wins.
        for (auto entityID : view) {
            auto& cameraComponent = Entity(entityID, this).getComponent<CameraComponent>();
            if (cameraComponent.isActiveCamera()) {
                entity = Entity(entityID, this);
            }
        }
        return entity;
    }

    void Scene::deleteAllEntities() {
        auto view = m_registry.view<IDComponent>();
        for (auto entity : view) {
            // Wrap the registry entity in an Entity object for handling
            Entity e{entity, this};
            destroyEntity(e);
        }
    }

    Entity Scene::createEntityWithUUID(UUID uuid, const std::string& name) {
        Entity entity = {m_registry.create(), this};
        entity.addComponent<IDComponent>(uuid);
        entity.addComponent<TransformComponent>();
        auto& tag = entity.addComponent<TagComponent>();
        tag.Tag = name.empty() ? "Entity" : name;
        Log::Logger::getInstance()->info("Created Entity with UUID: {} and Tag: {}",
                                         entity.getUUID().operator std::string(), entity.getName());

        return entity;
    }


    Entity Scene::createEntity(const std::string& name) {
        return createEntityWithUUID(UUID(), name);
    }

    Entity Scene::getOrCreateEntityByName(const std::string& name) {
        // Check if the entity with the given UUID exists
        auto view = m_registry.view<TagComponent>();
        for (auto entityHandle : view) {
            auto& tagComponent = view.get<TagComponent>(entityHandle);
            if (tagComponent.getTag() == name) {
                // Entity with the given UUID already exists
                Entity existingEntity = {entityHandle, this};
                Log::Logger::getInstance()->trace("Retrieved existing Entity with UUID: {} and Tag: {}",
                                                  existingEntity.getUUID().operator std::string(),
                                                  existingEntity.getName());
                return existingEntity;
            }
        }
        // If not found, create a new entity with the given UUID and name
        return createEntity(name);
    }

    Entity Scene::getEntityByName(const std::string& name) {
        // Check if the entity with the given UUID exists
        auto view = m_registry.view<TagComponent>();
        for (auto entityHandle : view) {
            auto& tagComponent = view.get<TagComponent>(entityHandle);
            if (tagComponent.getTag() == name) {
                // Entity with the given UUID already exists
                Entity existingEntity = {entityHandle, this};
                Log::Logger::getInstance()->trace("Retrieved existing Entity with UUID: {} and Tag: {}",
                                                  existingEntity.getUUID().operator std::string(),
                                                  existingEntity.getName());
                return existingEntity;
            }
        }
        // If not found return empty entity
        return Entity();
    }

    void Scene::destroyEntity(Entity entity) {
        if (!entity) {
            Log::Logger::getInstance()->warning("Attempted to delete an entity that doesn't exist");
            return;
        }
        // Checking if the entity is still valid before attempting to delete
        if (m_registry.valid(entity)) {
            Log::Logger::getInstance()->info("Deleting Entity with UUID: {} and Tag: {}",
                                             entity.getUUID().operator std::string(), entity.getName());
            notifyComponentRemoval(entity);

            // Perform the deletion
            m_registry.destroy(entity);
        }
        else {
            Log::Logger::getInstance()->warning(
                "Attempted to delete an invalid or already deleted entity");
        }
    }

    void Scene::destroyEntityRecursively(Entity entity) {
        // Delete children first
        if (entity.hasChildren()) {
            for (auto& child : entity.getChildren()) {
                destroyEntityRecursively(child);
            }
        }
        // Remove from parent
        if (entity.hasComponent<ParentComponent>()) {
            Entity parent = entity.getParent();
            parent.removeChild(entity);
        }
        // Destroy the entity
        m_context->activeScene()->destroyEntity(entity);
    }

    bool Scene::isDescendantOf(Entity entity, Entity potentialAncestor) {
        Entity currentParent = entity.getParent();
        while (currentParent) {
            if (currentParent == potentialAncestor)
                return true;
            currentParent = currentParent.getParent();
        }
        return false;
    }


    void Scene::notifyComponentRemoval(Entity entity) {
        // Check for each component type, and remove if the entity has the component
        if (entity.hasComponent<MeshComponent>()) {
            entity.removeComponent<MeshComponent>();
        }

        if (entity.hasComponent<MaterialComponent>()) {
            entity.removeComponent<MaterialComponent>();
        }

        if (entity.hasComponent<PointCloudComponent>()) {
            entity.removeComponent<PointCloudComponent>();
        }
        // Repeat for other components, adding more checks for each type of component
        // if (entity.hasComponent<OtherComponent>()) {
        //     entity.removeComponent<OtherComponent>();
        // }
    }

    void Scene::notifyEditorsComponentAdded(Entity entity, MeshComponent& component) {
        for (auto& editor : m_context->m_sceneRenderers) {
            editor.second->onComponentAdded(entity, component);
        }
        for (auto& editor : m_context->m_editors) {
            editor->onComponentAdded(entity, component);
        }
    }

    void Scene::notifyEditorsComponentUpdated(Entity entity, MeshComponent& component) {
        for (auto& editor : m_context->m_sceneRenderers) {
            editor.second->onComponentUpdated(entity, component);
        }
        for (auto& editor : m_context->m_editors) {
            editor->onComponentUpdated(entity, component);
        }
    }

    void Scene::notifyEditorsComponentRemoved(Entity entity, MeshComponent& component) {
        for (auto& editor : m_context->m_sceneRenderers) {
            editor.second->onComponentRemoved(entity, component);
        }
        for (auto& editor : m_context->m_editors) {
            editor->onComponentRemoved(entity, component);
        }
    }

    void Scene::notifyEditorsComponentAdded(Entity entity, MaterialComponent& component) {
        for (auto& editor : m_context->m_sceneRenderers) {
            editor.second->onComponentAdded(entity, component);
        }
        for (auto& editor : m_context->m_editors) {
            editor->onComponentAdded(entity, component);
        }
    }


    void Scene::notifyEditorsComponentUpdated(Entity entity, MaterialComponent& component) {
        for (auto& editor : m_context->m_sceneRenderers) {
            editor.second->onComponentUpdated(entity, component);
        }
        for (auto& editor : m_context->m_editors) {
            editor->onComponentUpdated(entity, component);
        }
    }

    void Scene::notifyEditorsComponentRemoved(Entity entity, MaterialComponent& component) {
        for (auto& editor : m_context->m_sceneRenderers) {
            editor.second->onComponentRemoved(entity, component);
        }
        for (auto& editor : m_context->m_editors) {
            editor->onComponentRemoved(entity, component);
        }
    }

    void Scene::notifyEditorsComponentAdded(Entity entity, PointCloudComponent& component) {
        for (auto& editor : m_context->m_sceneRenderers) {
            editor.second->onComponentAdded(entity, component);
        }
        for (auto& editor : m_context->m_editors) {
            editor->onComponentAdded(entity, component);
        }
    }

    void Scene::notifyEditorsComponentUpdated(Entity entity, PointCloudComponent& component) {
        for (auto& editor : m_context->m_sceneRenderers) {
            editor.second->onComponentUpdated(entity, component);
        }
        for (auto& editor : m_context->m_editors) {
            editor->onComponentUpdated(entity, component);
        }
    }

    void Scene::notifyEditorsComponentRemoved(Entity entity, PointCloudComponent& component) {
        for (auto& editor : m_context->m_sceneRenderers) {
            editor.second->onComponentRemoved(entity, component);
        }
        for (auto& editor : m_context->m_editors) {
            editor->onComponentRemoved(entity, component);
        }
    }


    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER


    /** COMPONENT ADDED **/

    template <>
    void Scene::onComponentAdded<IDComponent>(Entity entity, IDComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<MeshComponent>(Entity entity, MeshComponent& component) {
        notifyEditorsComponentAdded(entity, component);
    }

    template <>
    void Scene::onComponentAdded<MaterialComponent>(Entity entity, MaterialComponent& component) {
        notifyEditorsComponentAdded(entity, component);
    }

    template <>

    void Scene::onComponentAdded<PointCloudComponent>(Entity entity, PointCloudComponent& component) {
        notifyEditorsComponentAdded(entity, component);
    }

    template <>
    void Scene::onComponentAdded<TransformComponent>(Entity entity, TransformComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<CameraComponent>(Entity entity, CameraComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<ScriptComponent>(Entity entity, ScriptComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<TagComponent>(Entity entity, TagComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<TextComponent>(Entity entity, TextComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<ImageComponent>(Entity entity, ImageComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<GaussianComponent>(Entity entity, GaussianComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<ParentComponent>(Entity entity, ParentComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<ChildrenComponent>(Entity entity, ChildrenComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<GroupComponent>(Entity entity, GroupComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<VisibleComponent>(Entity entity, VisibleComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<TemporaryComponent>(Entity entity, TemporaryComponent& component) {
    }

    template <>
    void Scene::onComponentAdded<GaussianComponent2DGS>(Entity entity, GaussianComponent2DGS& component) {
    }

    /** COMPONENT REMOVE **/

    template <>
    void Scene::onComponentRemoved<IDComponent>(Entity entity, IDComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<MeshComponent>(Entity entity, MeshComponent& component) {
        notifyEditorsComponentRemoved(entity, component);
    }

    template <>
    void Scene::onComponentRemoved<MaterialComponent>(Entity entity, MaterialComponent& component) {
        notifyEditorsComponentRemoved(entity, component);
    }

    template <>
    void Scene::onComponentRemoved<PointCloudComponent>(Entity entity, PointCloudComponent& component) {
        notifyEditorsComponentRemoved(entity, component);
    }

    template <>
    void Scene::onComponentRemoved<TransformComponent>(Entity entity, TransformComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<CameraComponent>(Entity entity, CameraComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<ScriptComponent>(Entity entity, ScriptComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<TagComponent>(Entity entity, TagComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<TextComponent>(Entity entity, TextComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<ImageComponent>(Entity entity, ImageComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<GaussianComponent>(Entity entity, GaussianComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<ParentComponent>(Entity entity, ParentComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<ChildrenComponent>(Entity entity, ChildrenComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<GroupComponent>(Entity entity, GroupComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<VisibleComponent>(Entity entity, VisibleComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<TemporaryComponent>(Entity entity, TemporaryComponent& component) {
    }

    template <>
    void Scene::onComponentRemoved<GaussianComponent2DGS>(Entity entity, GaussianComponent2DGS& component) {
    }

    /** COMPONENT UPDATE **/
    template <>
    void Scene::onComponentUpdated<IDComponent>(Entity entity, IDComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<MeshComponent>(Entity entity, MeshComponent& component) {
        notifyEditorsComponentUpdated(entity, component);
    }

    template <>
    void Scene::onComponentUpdated<MaterialComponent>(Entity entity, MaterialComponent& component) {
        notifyEditorsComponentUpdated(entity, component);
    }

    template <>
    void Scene::onComponentUpdated<PointCloudComponent>(Entity entity, PointCloudComponent& component) {
        notifyEditorsComponentUpdated(entity, component);
    }

    template <>
    void Scene::onComponentUpdated<TransformComponent>(Entity entity, TransformComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<CameraComponent>(Entity entity, CameraComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<ScriptComponent>(Entity entity, ScriptComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<TagComponent>(Entity entity, TagComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<TextComponent>(Entity entity, TextComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<ImageComponent>(Entity entity, ImageComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<GaussianComponent>(Entity entity, GaussianComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<ParentComponent>(Entity entity, ParentComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<ChildrenComponent>(Entity entity, ChildrenComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<GroupComponent>(Entity entity, GroupComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<VisibleComponent>(Entity entity, VisibleComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<TemporaryComponent>(Entity entity, TemporaryComponent& component) {
    }

    template <>
    void Scene::onComponentUpdated<GaussianComponent2DGS>(Entity entity, GaussianComponent2DGS& component) {
    }

    DISABLE_WARNING_POP
}
