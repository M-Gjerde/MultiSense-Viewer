//
// Created by magnus on 8/13/24.
//

#include "Viewer/Scenes/Entity.h"

#include "Viewer/Scenes/Scene.h"

#include <Viewer/Rendering/Components/GaussianComponent.h>

#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/Components/MeshComponent.h"
#include "Viewer/Rendering/Components/ImageComponent.h"
#include "Viewer/Application/Application.h"
#include "Viewer/Rendering/Components/PointCloudComponent.h"

namespace VkRender {
    Scene::Scene(VkRender::Application *context) {
        m_context = context;
    }


    void Scene::update() {

        auto cameraView = m_registry.view<CameraComponent>();
        for (auto e: cameraView) {
            auto entity = Entity(e, this);
            auto cameraComponent = entity.getComponent<CameraComponent>();
            auto &transform = entity.getComponent<TransformComponent>();
            cameraComponent.camera->updateViewMatrix(transform.getTransform());
            cameraComponent.camera->updateProjectionMatrix();

            if (entity.hasComponent<MaterialComponent>()) {
                auto &material = entity.getComponent<MaterialComponent>();
                if (cameraComponent.isActiveCamera()) {
                    material.albedo = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
                } else {
                    material.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
                }
            }
        }

        glm::vec3 position(0.0f);
        auto gaussianView = m_registry.view<GaussianComponent2DGS>();
        for (auto e: gaussianView) {
            auto entity = Entity(e, this);
            auto &gaussianComponent = entity.getComponent<GaussianComponent2DGS>();
            position = gaussianComponent.positions.front();
        }

        auto meshView = m_registry.view<MeshComponent>();
        for (auto e: meshView) {
            auto entity = Entity(e, this);
            auto& meshComponent = entity.getComponent<MeshComponent>();
            if (meshComponent.meshDataType() == OBJ_FILE) {
                auto params = std::dynamic_pointer_cast<OBJFileMeshParameters>(meshComponent.meshParameters);
                if (entity.getName() == "Disk"){
                    auto& transformComponent = entity.getComponent<TransformComponent>();
                    transformComponent.setPosition(position);
                }
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

    void Scene::deleteAllEntities() {
        auto view = m_registry.view<IDComponent>();
        for (auto entity: view) {
            // Wrap the registry entity in an Entity object for handling
            Entity e{entity, this};
            destroyEntity(e);
        }
    }

    Entity Scene::createEntityWithUUID(UUID uuid, const std::string &name) {
        Entity entity = {m_registry.create(), this};
        entity.addComponent<IDComponent>(uuid);
        entity.addComponent<TransformComponent>();
        auto &tag = entity.addComponent<TagComponent>();
        tag.Tag = name.empty() ? "Entity" : name;
        Log::Logger::getInstance()->info("Created Entity with UUID: {} and Tag: {}",
                                         entity.getUUID().operator std::string(), entity.getName());

        return entity;
    }


    Entity Scene::createEntity(const std::string &name) {
        return createEntityWithUUID(UUID(), name);
    }

    Entity Scene::getOrCreateEntityByName(const std::string &name) {
        // Check if the entity with the given UUID exists
        auto view = m_registry.view<TagComponent>();
        for (auto entityHandle : view) {
            auto &tagComponent = view.get<TagComponent>(entityHandle);
            if (tagComponent.getTag() == name) {
                // Entity with the given UUID already exists
                Entity existingEntity = {entityHandle, this};
                Log::Logger::getInstance()->trace("Retrieved existing Entity with UUID: {} and Tag: {}",
                                                 existingEntity.getUUID().operator std::string(), existingEntity.getName());
                return existingEntity;
            }
        }
        // If not found, create a new entity with the given UUID and name
        return createEntity(name);
    }
    Entity Scene::getEntityByName(const std::string &name) {
        // Check if the entity with the given UUID exists
        auto view = m_registry.view<TagComponent>();
        for (auto entityHandle : view) {
            auto &tagComponent = view.get<TagComponent>(entityHandle);
            if (tagComponent.getTag() == name) {
                // Entity with the given UUID already exists
                Entity existingEntity = {entityHandle, this};
                Log::Logger::getInstance()->trace("Retrieved existing Entity with UUID: {} and Tag: {}",
                                                 existingEntity.getUUID().operator std::string(), existingEntity.getName());
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
        } else {
            Log::Logger::getInstance()->warning(
                    "Attempted to delete an invalid or already deleted entity");
        }
    }

    void Scene::destroyEntityRecursively(Entity entity) {
        // Delete children first
        if (entity.hasChildren()) {
            for (auto &child: entity.getChildren()) {
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

    void Scene::notifyEditorsComponentAdded(Entity entity, MeshComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentAdded(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentAdded(entity, component);
        }
    }

    void Scene::notifyEditorsComponentUpdated(Entity entity, MeshComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentUpdated(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentUpdated(entity, component);
        }
    }

    void Scene::notifyEditorsComponentRemoved(Entity entity, MeshComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentRemoved(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentRemoved(entity, component);
        }
    }

    void Scene::notifyEditorsComponentAdded(Entity entity, MaterialComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentAdded(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentAdded(entity, component);
        }
    }


    void Scene::notifyEditorsComponentUpdated(Entity entity, MaterialComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentUpdated(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentUpdated(entity, component);
        }
    }

    void Scene::notifyEditorsComponentRemoved(Entity entity, MaterialComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentRemoved(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentRemoved(entity, component);
        }
    }

    void Scene::notifyEditorsComponentAdded(Entity entity, PointCloudComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentAdded(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentAdded(entity, component);
        }
    }

    void Scene::notifyEditorsComponentUpdated(Entity entity, PointCloudComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentUpdated(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentUpdated(entity, component);
        }
    }

    void Scene::notifyEditorsComponentRemoved(Entity entity, PointCloudComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentRemoved(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentRemoved(entity, component);
        }
    }


    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER


    /** COMPONENT ADDED **/

    template<>
    void Scene::onComponentAdded<IDComponent>(Entity entity, IDComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<MeshComponent>(Entity entity, MeshComponent &component) {
        notifyEditorsComponentAdded(entity, component);
    }

    template<>
    void Scene::onComponentAdded<MaterialComponent>(Entity entity, MaterialComponent &component) {
        notifyEditorsComponentAdded(entity, component);
    }

    template<>

    void Scene::onComponentAdded<PointCloudComponent>(Entity entity, PointCloudComponent &component) {
        notifyEditorsComponentAdded(entity, component);
    }

    template<>
    void Scene::onComponentAdded<TransformComponent>(Entity entity, TransformComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<CameraComponent>(Entity entity, CameraComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<ScriptComponent>(Entity entity, ScriptComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<TagComponent>(Entity entity, TagComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<TextComponent>(Entity entity, TextComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<ImageComponent>(Entity entity, ImageComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<GaussianComponent>(Entity entity, GaussianComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<ParentComponent>(Entity entity, ParentComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<ChildrenComponent>(Entity entity, ChildrenComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<GroupComponent>(Entity entity, GroupComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<VisibleComponent>(Entity entity, VisibleComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<TemporaryComponent>(Entity entity, TemporaryComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<GaussianComponent2DGS>(Entity entity, GaussianComponent2DGS &component) {
    }
    /** COMPONENT REMOVE **/

    template<>
    void Scene::onComponentRemoved<IDComponent>(Entity entity, IDComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<MeshComponent>(Entity entity, MeshComponent &component) {
        notifyEditorsComponentRemoved(entity, component);
    }

    template<>
    void Scene::onComponentRemoved<MaterialComponent>(Entity entity, MaterialComponent &component) {
        notifyEditorsComponentRemoved(entity, component);
    }

    template<>
    void Scene::onComponentRemoved<PointCloudComponent>(Entity entity, PointCloudComponent &component) {
        notifyEditorsComponentRemoved(entity, component);
    }

    template<>
    void Scene::onComponentRemoved<TransformComponent>(Entity entity, TransformComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<CameraComponent>(Entity entity, CameraComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<ScriptComponent>(Entity entity, ScriptComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<TagComponent>(Entity entity, TagComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<TextComponent>(Entity entity, TextComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<ImageComponent>(Entity entity, ImageComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<GaussianComponent>(Entity entity, GaussianComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<ParentComponent>(Entity entity, ParentComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<ChildrenComponent>(Entity entity, ChildrenComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<GroupComponent>(Entity entity, GroupComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<VisibleComponent>(Entity entity, VisibleComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<TemporaryComponent>(Entity entity, TemporaryComponent &component) {
    }
    template<>
    void Scene::onComponentRemoved<GaussianComponent2DGS>(Entity entity, GaussianComponent2DGS &component) {
    }
    /** COMPONENT UPDATE **/
    template<>
    void Scene::onComponentUpdated<IDComponent>(Entity entity, IDComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<MeshComponent>(Entity entity, MeshComponent &component) {
        notifyEditorsComponentUpdated(entity, component);
    }

    template<>
    void Scene::onComponentUpdated<MaterialComponent>(Entity entity, MaterialComponent &component) {
        notifyEditorsComponentUpdated(entity, component);
    }

    template<>
    void Scene::onComponentUpdated<PointCloudComponent>(Entity entity, PointCloudComponent &component) {
        notifyEditorsComponentUpdated(entity, component);
    }

    template<>
    void Scene::onComponentUpdated<TransformComponent>(Entity entity, TransformComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<CameraComponent>(Entity entity, CameraComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<ScriptComponent>(Entity entity, ScriptComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<TagComponent>(Entity entity, TagComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<TextComponent>(Entity entity, TextComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<ImageComponent>(Entity entity, ImageComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<GaussianComponent>(Entity entity, GaussianComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<ParentComponent>(Entity entity, ParentComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<ChildrenComponent>(Entity entity, ChildrenComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<GroupComponent>(Entity entity, GroupComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<VisibleComponent>(Entity entity, VisibleComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<TemporaryComponent>(Entity entity, TemporaryComponent &component) {
    }
    template<>
    void Scene::onComponentUpdated<GaussianComponent2DGS>(Entity entity, GaussianComponent2DGS &component) {
    }

    DISABLE_WARNING_POP
}
