//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_COMPONENTS_H
#define MULTISENSE_VIEWER_COMPONENTS_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <filesystem>
#include <entt/entt.hpp>
#include "Viewer/Rendering/Core/UUID.h"

namespace VkRender {

    struct IDComponent {
        UUID ID{};

        IDComponent() = default;

        explicit IDComponent(const UUID &uuid) : ID(uuid) {
        }

    };

    struct TagComponent {
        std::string Tag;
        std::string &getTag() { return Tag; }

        void setTag(const std::string &tag) { Tag = tag; }

        char* getTagForKernel() const { return const_cast<char *>(Tag.c_str()); }

    };


    struct ScriptComponent {
        std::string className;
    };


    struct ParentComponent {
        entt::entity parent = entt::null;
    };

    /** @brief Temporary components are not saved to scene file */
    struct TemporaryComponent {
        entt::entity entity;
    };

    struct ChildrenComponent {
        std::vector<entt::entity> children{};
    };
    struct VisibleComponent {
        bool visible = true;
    };
    struct GroupComponent {
        std::string placeHolder;
    };

    struct TextComponent {
        std::string TextString;
        glm::vec4 Color{1.0f};
    };

}

#endif //MULTISENSE_VIEWER_COMPONENTS_H
