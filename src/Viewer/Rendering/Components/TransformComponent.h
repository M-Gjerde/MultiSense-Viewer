//
// Created by magnus-desktop on 12/15/24.
//

#ifndef TRANSFORMCOMPONENT_H
#define TRANSFORMCOMPONENT_H

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace VkRender {
    struct TransformComponent {
        // Flip the up axis option
        bool m_flipUpAxis = false;
        bool m_wasMoved = false;

        // Transformation components
        glm::vec3 translation = {0.0f, 0.0f, 0.0f};
        glm::vec3 rotationEuler = {0.0f, 0.0f, 0.0f}; // Euler angles in degrees
        glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity quaternion
        glm::vec3 scale = {1.0f, 1.0f, 1.0f};

        // Constructors
        TransformComponent() = default;

        // Get the transformation matrix
        [[nodiscard]] glm::mat4 getTransform() const {
            glm::mat4 rotMat = glm::mat4_cast(rotation);

            if (m_flipUpAxis) {
                glm::mat4 flipUpRotationMatrix = glm::rotate(
                        glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                rotMat = flipUpRotationMatrix * rotMat;
            }

            return glm::translate(glm::mat4(1.0f), translation) * rotMat *
                   glm::scale(glm::mat4(1.0f), scale);
        }

        bool moved() const { return m_wasMoved; }

        void setMoving(bool state) {
            m_wasMoved = state;
        }

        // Set rotation using Euler angles (degrees)
        void setRotationEuler(const glm::vec3 &eulerAnglesDegrees) {
            rotationEuler = eulerAnglesDegrees;
            glm::vec3 radians = glm::radians(rotationEuler);
            rotation = glm::quat(radians);
        }

        // Get rotation as Euler angles (degrees)
        glm::vec3 &getRotationEuler() {
            return rotationEuler;
        }

        void updateFromEulerRotation() {
            setRotationEuler(rotationEuler);
        }

        // Set rotation using quaternion
        void setRotationQuaternion(const glm::quat &q) {
            rotation = q;
            rotationEuler = glm::degrees(glm::eulerAngles(rotation));
        }

        // Get rotation as quaternion
        glm::quat &getRotationQuaternion() {
            return rotation;
        }

        // Position setters and getters
        void setPosition(const glm::vec3 &v) {
            translation = v;
        }

        glm::vec3 &getPosition() {
            return translation;
        }

        glm::vec3 getPosition() const {
            return translation;
        }

        // Scale setters and getters
        void setScale(const glm::vec3 &s) {
            scale = s;
        }

        glm::vec3 &getScale() {
            return scale;
        }

        // Flip up axis option setters and getters
        void setFlipUpOption(const bool flipUp) {
            m_flipUpAxis = flipUp;
        }

        bool getFlipUpOption() const {
            return m_flipUpAxis;
        }
    };

}
#endif //TRANSFORMCOMPONENT_H
