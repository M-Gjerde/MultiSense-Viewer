//
// Created by magnus-desktop on 12/11/24.
//

#ifndef BASECAMERA_H
#define BASECAMERA_H

#include <glm/glm.hpp>

#include "CameraDefinitions.h"
#include "Viewer/Rendering/Components/TransformComponent.h"

namespace VkRender {

    struct ProjectionParameters {
        float near = 0.1f;
        float far = 100.0f;
        float aspect = 1.6f;
        float fov = 60.0f; // FOV in degrees

        // Overload equality operator
        bool operator==(const ProjectionParameters &other) const {
            return near == other.near &&
                   far == other.far &&
                   aspect == other.aspect &&
                   fov == other.fov;
        }

        // Overload inequality operator for convenience
        bool operator!=(const ProjectionParameters &other) const {
            return !(*this == other);
        }
    };

    class BaseCamera {
    public:
        virtual ~BaseCamera() = default;
        BaseCamera() = default;

        explicit BaseCamera(float aspect, float fov = 60.0f) {
            m_parameters.aspect = aspect;
            m_parameters.fov = fov;
            BaseCamera::updateProjectionMatrix();
        }

        BaseCamera(const SharedCameraSettings& sharedSettings, const ProjectionParameters& projectionParameters) : m_settings(sharedSettings), m_parameters(projectionParameters) {

        }

        struct Matrices {
            glm::mat4 view = glm::mat4(1.0f);
            glm::mat4 projection = glm::mat4(1.0f);
            glm::vec3 position = glm::vec3(0.0f);
        } matrices;

        SharedCameraSettings m_settings;
        ProjectionParameters m_parameters;

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
            float tanHalfFovy = tanf(glm::radians(m_parameters.fov) * 0.5f);
            float x = 1 / (tanHalfFovy * m_parameters.aspect);
            float y = 1 / tanHalfFovy;
            float A = m_parameters.far / (m_parameters.near - m_parameters.far);
            float B = -(m_parameters.far * m_parameters.near) / (m_parameters.far - m_parameters.near);
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
            if (m_settings.flipY) {
                matrices.projection[1][1] *= -1;
            }
            if (m_settings.flipX) {
                matrices.projection[0][0] *= -1; // Flip X-axis for left-to-right flipping
            }
        };

        // Movement and rotation inputs now should affect the TransformComponent externally.
        // The camera class just needs to know how to build projection matrices and view matrices.
    };
}
#endif //BASECAMERA_H
