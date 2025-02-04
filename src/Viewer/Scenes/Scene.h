//
// Created by mgjer on 14/07/2024.
//

#ifndef MULTISENSE_VIEWER_SCENE_H
#define MULTISENSE_VIEWER_SCENE_H

#include <entt/entt.hpp>
#include <Viewer/Rendering/Components/MaterialComponent.h>
#include <Viewer/Rendering/Components/MeshComponent.h>

#include "Viewer/Rendering/Core/UUID.h"
#include "Viewer/Rendering/Components/PointCloudComponent.h"

namespace VkRender {
    class Application;
    class Entity;

    class Scene {

    public:
        explicit Scene(Application *context);

        void deleteAllEntities();

        ~Scene() {

        }
        void update();

        entt::registry &getRegistry() { return m_registry; };
        const entt::registry &getRegistry() const {return m_registry;}

        Entity createEntity(const std::string &name);
        Entity getOrCreateEntityByName(const std::string &name);
        Entity getEntityByName(const std::string& name);
        Entity createEntityWithUUID(UUID uuid, const std::string &name);
        void destroyEntity(Entity entity);
        void notifyComponentRemoval(Entity entity);

        void destroyEntityRecursively(Entity entity);
        bool isDescendantOf(Entity entity, Entity potentialAncestor);

        CameraComponent* getActiveCamera();
        Entity getActiveCameraEntity();

        template<class T>
        void onComponentUpdated(Entity entity, T &component);
        template<typename T>
        void onComponentAdded(Entity entity, T &component);
        template<class T>
        void onComponentRemoved(Entity entity, T &component);

    private:
        entt::registry m_registry;
        friend class Entity;
        friend class SceneSerializer;
        Application *m_context;

        void notifyEditorsComponentAdded(Entity entity, MaterialComponent &component);
        void notifyEditorsComponentAdded(Entity entity, PointCloudComponent &component);
        void notifyEditorsComponentAdded(Entity entity, MeshComponent &component);
        void notifyEditorsComponentUpdated(Entity entity, PointCloudComponent &component);
        void notifyEditorsComponentUpdated(Entity entity, MaterialComponent &component);
        void notifyEditorsComponentUpdated(Entity entity, MeshComponent &component);
        void notifyEditorsComponentRemoved(Entity entity, PointCloudComponent &component);
        void notifyEditorsComponentRemoved(Entity entity, MaterialComponent &component);
        void notifyEditorsComponentRemoved(Entity entity, MeshComponent &component);

        template<class T>
        void notifyEditorsComponentAdded(Entity entity, T &component);



    };


}


#endif //MULTISENSE_VIEWER_SCENE_H
