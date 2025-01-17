//
// Created by magnus-desktop on 12/11/24.
//

#ifndef ARCBALLCAMERA_H
#define ARCBALLCAMERA_H
namespace VkRender {
    class ArcballCamera : public BaseCamera {
    public:
        glm::vec2 m_rotation = glm::vec2(0.0f, 0.0f);
        float m_rotationSpeed = 0.20f;
        float m_translationSpeed = 0.005f;
        float m_zoomValue = 1.0f;
        glm::vec3 m_positionOffset = glm::vec3(0.0f);
        explicit ArcballCamera(float aspect)
            : BaseCamera(aspect) {
            rotate(0.0f, 0.0f);
        }
        ArcballCamera() = default;

        void setDefaultPosition(glm::vec2 defaultRotation, float zoomValue = 1.0f){
            m_rotation = defaultRotation;
            m_zoomValue = zoomValue;
            m_positionOffset = glm::vec3(0.0f);
            rotate(0, 0);
        }

        void rotate(float dx, float dy) {
            dx *= m_rotationSpeed;
            dy *= m_rotationSpeed;
            m_rotation.x += dx;
            m_rotation.y += dy;

            glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            // Adjust rotation based on the mouse movement
            glm::quat rotX = glm::angleAxis(glm::radians(m_rotation.x), glm::vec3(0.0f, 0.0f, 1.0f));
            glm::quat rotY = glm::angleAxis(glm::radians(m_rotation.y), glm::vec3(1.0f, 0.0f, 0.0f));
            // Combine rotations in a specific order
            orientation = rotX * orientation;
            orientation = orientation * rotY;
            orientation = glm::normalize(orientation);

            glm::mat4 transMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 3.0f) * m_zoomValue);
            glm::mat4 positionOffsetMatrix = glm::translate(glm::mat4(1.0f), m_positionOffset);
            glm::mat4 rotMatrix = glm::mat4_cast(orientation);
            auto trans = rotMatrix * transMatrix * positionOffsetMatrix;
            m_transform.setPosition(trans[3]);
            m_transform.setRotationQuaternion(glm::quat_cast(trans));
            BaseCamera::updateViewMatrix(trans);
        }

        void translate(float dx, float dy){;
            dx *= m_translationSpeed;
            dy *= m_translationSpeed;
            glm::mat4 view = matrices.view;
            glm::vec3 right = glm::vec3(view[0][0], view[1][0], view[2][0]);   // First column
            glm::vec3 up = glm::vec3(view[0][1], view[1][1], view[2][1]);      // Second column
            glm::vec3 forward = glm::vec3(view[0][2], view[1][2], view[2][2]); // Third column
            glm::vec3 translation = -dx * right - dy * up; // Move left/right (dx) and up/down (dy)
            glm::vec4 worldTranslation = glm::vec4(translation, 1.0f) * m_transform.getTransform();
            m_positionOffset += glm::vec3(worldTranslation);

            rotate(0, 0);
        }

        void zoom(float change) {
            m_zoomValue *= std::abs(change);

            glm::mat4 transMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 3.0f) * m_zoomValue);
            glm::mat4 positionOffsetMatrix = glm::translate(glm::mat4(1.0f), m_positionOffset);
            glm::mat4 rotMatrix = glm::mat4_cast(m_transform.getRotationQuaternion());
            auto trans = rotMatrix * transMatrix * positionOffsetMatrix;
            m_transform.setPosition(trans[3]);
            BaseCamera::updateViewMatrix(trans);

        }



    private:
        TransformComponent m_transform;
    };
}
#endif //ARCBALLCAMERA_H
