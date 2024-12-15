//
// Created by magnus-desktop on 12/11/24.
//

#ifndef BASECAMERA_H
#define BASECAMERA_H

#include <glm/glm.hpp>

#include "Viewer/Rendering/Components/TransformComponent.h"

namespace VkRender {
    class BaseCamera {
    public:
        virtual ~BaseCamera() = default;
        BaseCamera() = default;

        explicit BaseCamera(float aspect, float fov = 60.0f) : m_aspectRatio(aspect), m_fov(fov) {
            BaseCamera::updateProjectionMatrix();
        }

        struct Matrices {
            glm::mat4 view = glm::mat4(1.0f);
            glm::mat4 projection = glm::mat4(1.0f);
            glm::vec3 position = glm::vec3(0.0f);
        } matrices;

        float m_zNear = 0.1f;
        float m_zFar = 100.0f;
        float m_fov = 60.0f; // FOV in degrees
        float m_aspectRatio = 1.6f; // 16/10 aspect ratio
        bool m_flipYProjection = false;
        // Instead of storing pose in here, just rely on external transforms.

        virtual void updateViewMatrix(const glm::mat4& worldTransform) {
            // The view matrix is typically the inverse of the camera's world transform
            matrices.position = glm::vec3(worldTransform[3]); // Convert vec4 to vec3 (drop the w component)
            matrices.view = glm::inverse(worldTransform);
        }
        virtual void updateViewMatrix(TransformComponent& trans) {
            // The view matrix is typically the inverse of the camera's world transform
            matrices.position = trans.getPosition(); // Convert vec4 to vec3 (drop the w component)
            matrices.view = glm::inverse(trans.getTransform());
        }

        virtual void updateProjectionMatrix() {

            // Guide: https://vincent-p.github.io/posts/vulkan_perspective_matrix/
            float tanHalfFovy = tanf(glm::radians(m_fov) * 0.5f);
            float x = 1 / (tanHalfFovy * m_aspectRatio);
            float y = 1 / tanHalfFovy;
            float A = m_zFar / (m_zNear - m_zFar);
            float B = -(m_zFar * m_zNear) / (m_zFar - m_zNear);
            matrices.projection = glm::mat4(
                    x, 0.0f, 0.0f, 0.0f,
                    0.0f, y, 0.0f, 0.0f,
                    0.0f, 0.0f, A, -1.0f,
                    0.0f, 0.0f, B, 0.0f
            );

            /*
            matrices.projection = glm::perspectiveRH_ZO(
                glm::radians(m_fov),
                m_aspectRatio,
                m_zNear,
                m_zFar
            );
            */

            if (m_flipYProjection) {
                matrices.projection[1][1] *= -1;
            }
        };

        // Movement and rotation inputs now should affect the TransformComponent externally.
        // The camera class just needs to know how to build projection matrices and view matrices.
    };
}
#endif //BASECAMERA_H
