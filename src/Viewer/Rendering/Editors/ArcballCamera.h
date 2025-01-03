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
        float m_zoomValue = 1.0f;
        explicit ArcballCamera(float aspect)
            : BaseCamera(aspect) {
            rotate(0.0f, 0.0f);
        }
        ArcballCamera() = default;

        void setDefaultPosition(glm::vec2 defaultRotation, float zoomValue = 1.0f){
            m_rotation = defaultRotation;
            m_zoomValue = zoomValue;
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
            glm::mat4 rotMatrix = glm::mat4_cast(orientation);
            auto trans = rotMatrix * transMatrix;
            m_transform.setPosition(trans[3]);
            m_transform.setRotationQuaternion(glm::quat_cast(trans));
            BaseCamera::updateViewMatrix(trans);
        }

        void zoom(float change) {
            m_zoomValue *= std::abs(change);

            glm::mat4 transMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 3.0f) * m_zoomValue);
            glm::mat4 rotMatrix = glm::mat4_cast(m_transform.getRotationQuaternion());
            auto trans = rotMatrix * transMatrix;
            m_transform.setPosition(trans[3]);
            BaseCamera::updateViewMatrix(trans);

        }



    private:
        TransformComponent m_transform;
    };
}
#endif //ARCBALLCAMERA_H
