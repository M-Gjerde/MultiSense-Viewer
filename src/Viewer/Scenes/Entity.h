//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_ENTITY_H
#define MULTISENSE_VIEWER_ENTITY_H

#include <cassert>
#include <entt/entt.hpp>

#include "Viewer/Tools/Macros.h"
#include "Viewer/Scenes/Scene.h"

namespace VkRender {

    class Entity
    {
    public:
        Entity() = default;
        Entity(entt::entity handle, Scene *scene);
        Entity(const Entity& other) = default;

        template<typename T, typename... Args>
        T& addComponent(Args&&... args)
        {
            VK_ASSERT(!hasComponent<T>(), "Entity already has component!");
            T& component = m_scene->m_registry.emplace<T>(m_entityHandle, std::forward<Args>(args)...);
            m_scene->onComponentAdded<T>(*this, component);
            return component;
        }

        template<typename T, typename... Args>
        T& addOrReplaceComponent(Args&&... args)
        {
            T& component = m_scene->m_registry.emplace_or_replace<T>(m_entityHandle, std::forward<Args>(args)...);
            m_scene->onComponentAdded<T>(*this, component);
            return component;
        }

        template<typename T, typename... Args>
        T& getOrAddComponent(Args&&... args)
        {
            if (hasComponent<T>())
                return getComponent<T>();

            return addComponent<T>(std::forward<Args>(args)...);
        }
        template<typename T>
        T& getComponent()
        {
            VK_ASSERT(hasComponent<T>(), "Entity does not have component!");
            return m_scene->m_registry.get<T>(m_entityHandle);
        }

        template<typename T>
        bool hasComponent()
        {
            return  m_scene->m_registry.any_of<T>(m_entityHandle);
        }
        template<typename T>
        void removeComponent()
        {
            VK_ASSERT(hasComponent<T>(), "Entity does not have component!");
            auto component =  m_scene->m_registry.get<T>(m_entityHandle);
            m_scene->onComponentRemoved<T>(*this, component);
            m_scene->m_registry.remove<T>(m_entityHandle);
        }


        operator bool() const { return m_entityHandle != entt::null; }
        operator entt::entity() const { return m_entityHandle; }
        operator uint32_t() const { return static_cast<uint32_t>(m_entityHandle); }


        const std::string& getName() { return getComponent<TagComponent>().Tag; }
        UUID getUUID() { return getComponent<IDComponent>().ID; }

        // Set this entity's parent
        void setParent(Entity parent) {
            // Remove from previous parent's children list
            if (hasComponent<ParentComponent>()) {
                Entity previousParent = getParent();
                if (previousParent) {
                    previousParent.removeChild(*this);
                }
            }

            addOrReplaceComponent<ParentComponent>().parent = parent;
            parent.addChild(*this);
        }

        // Get this entity's parent
        Entity getParent() {
            if (hasComponent<ParentComponent>()) {
                entt::entity parentHandle = getComponent<ParentComponent>().parent;
                return Entity(parentHandle, m_scene);
            }
            return Entity(); // Null entity
        }

        // Add a child to this entity
        void addChild(Entity child) {
            if (!hasComponent<ChildrenComponent>()) {
                addComponent<ChildrenComponent>();
            }
            getComponent<ChildrenComponent>().children.push_back((entt::entity)child);
        }

        // Remove a child from this entity
        void removeChild(Entity child) {
            if (hasComponent<ChildrenComponent>()) {
                auto& children = getComponent<ChildrenComponent>().children;
                children.erase(std::remove(children.begin(), children.end(), (entt::entity)child), children.end());
            }
        }

        // Get this entity's children
        std::vector<Entity> getChildren() {
            std::vector<Entity> result;
            if (hasComponent<ChildrenComponent>()) {
                auto& childrenHandles = getComponent<ChildrenComponent>().children;
                for (auto childHandle : childrenHandles) {
                    result.emplace_back(childHandle, m_scene);
                }
            }
            return result;
        }

        bool isVisible() {
            if (!hasComponent<VisibleComponent>())
                return true; // Default to visible if no component

            bool visible = getComponent<VisibleComponent>().visible;
            Entity parent = getParent();
            while (parent) {
                if (parent.hasComponent<VisibleComponent>()) {
                    visible = visible && parent.getComponent<VisibleComponent>().visible;
                }
                parent = parent.getParent();
            }
            return visible;
        }

        // Check if this entity has children
        bool hasChildren() {
            return hasComponent<ChildrenComponent>() && !getComponent<ChildrenComponent>().children.empty();
        }

        bool operator==(const Entity& other) const
        {
            return m_entityHandle == other.m_entityHandle && m_scene == other.m_scene;
        }

        bool operator!=(const Entity& other) const
        {
            return !(*this == other);
        }


    private:
        entt::entity m_entityHandle{ entt::null };
        Scene* m_scene = nullptr;

    };

}

#endif //MULTISENSE_VIEWER_ENTITY_H
