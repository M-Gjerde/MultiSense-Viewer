//
// Created by magnus-desktop on 12/11/24.
//

#ifndef PINHOLECAMERA_H
#define PINHOLECAMERA_H

#include <multisense_viewer/src/Viewer/Rendering/Editors/BaseCamera.h>


namespace VkRender {
    struct PinholeParameters {
        float height = 720; // Default image height
        float width = 1280; // Default image width
        float fx = 1280.0f; // Default horizontal focal length (pixels)
        float fy = 720.0f; // Default vertical focal length (pixels)
        float cx = 640.0f; // Default principal point x-coordinate (pixels)
        float cy = 360.0f; // Default principal point y-coordinate (pixels)
        float focalLength = 4; // focal length in mm
        float fNumber = 1.0f / 2.8f; // focal length in mm
        float near = 0.01f;
        float far = 100.0f;
        // Overload equality operator
        bool operator==(const PinholeParameters &other) const {
            return height == other.height &&
                   width == other.width &&
                   fx == other.fx &&
                   fy == other.fy &&
                   cx == other.cx &&
                   focalLength == other.focalLength &&
                   fNumber == other.fNumber &&
                   cy == other.cy;
        }

        // Optional: Overload inequality operator for convenience
        bool operator!=(const PinholeParameters &other) const {
            return !(*this == other);
        }
    };

    class PinholeCamera : public BaseCamera {
    public:

        PinholeCamera() = default;

        PinholeParameters m_parameters{};
        SharedCameraSettings m_settings{};

        [[nodiscard]] const PinholeParameters& parameters() const{return m_parameters;}

        explicit PinholeCamera(const SharedCameraSettings& sharedSettings, const PinholeParameters& pinholeParameters) : m_settings(sharedSettings), m_parameters(pinholeParameters){
            PinholeCamera::updateProjectionMatrix();
        }


        void updateProjectionMatrix() override {
            float A = m_parameters.far / (m_parameters.near - m_parameters.far);
            float B = -(m_parameters.far * m_parameters.near) / (m_parameters.far - m_parameters.near);
            float w = m_parameters.width;
            float h = m_parameters.height;

            float fxNorm = (2.0f * m_parameters.fx) / w;
            float fyNorm = (2.0f * m_parameters.fy) / h;
            float cxNorm = -((2.0f * m_parameters.cx) - w) / w;
            float cyNorm = -((2.0f * m_parameters.cy) - h) / h;
            if (m_settings.flipY) {
                fyNorm *= -1;
            }
            if (m_settings.flipX) {
                fxNorm *= -1;
            }

            matrices.projection = glm::mat4(
                    fxNorm, 0.0f, 0.0f, 0.0f,
                    0.0f, fyNorm, 0.0f, 0.0f,
                    cxNorm, cyNorm, A, -1.0f,
                    0.0f, 0.0f, B, 0.0f
            );
        }
    };
}
#endif //PINHOLECAMERA_H
