//
// Created by magnus-desktop on 12/11/24.
//

#ifndef PINHOLECAMERA_H
#define PINHOLECAMERA_H

#include <multisense_viewer/src/Viewer/Rendering/Editors/BaseCamera.h>


namespace VkRender {
    struct PinholeParameters {
        int height = 720; // Default image height
        int width = 1280; // Default image width
        float fx = 1280.0f; // Default horizontal focal length (pixels)
        float fy = 720.0f; // Default vertical focal length (pixels)
        float cx = 640.0f; // Default principal point x-coordinate (pixels)
        float cy = 360.0f; // Default principal point y-coordinate (pixels)
        float focalLength = 4; // focal length in mm
        float fNumber = 1.0f / 2.8f; // focal length in mm
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

        float m_fx, m_fy, m_cx, m_cy;
        float m_width, m_height;
        float m_focalLength;
        float m_fNumber;

        PinholeCamera(uint32_t width, uint32_t height, float fx, float fy, float cx, float cy,
                      float focalLength = 10.0f, float zNear = 0.1f,
                      float zFar = 100.0f) {
            m_width = static_cast<float>(width);
            m_height = static_cast<float>(height);
            m_fx = fx;
            m_fy = fy;
            m_cx = cx;
            m_cy = cy;
            m_focalLength = focalLength;
            m_zNear = zNear;
            m_zFar = zFar;
            PinholeCamera::updateProjectionMatrix();
        }

        explicit PinholeCamera(PinholeParameters parameters, float zNear = 0.1f,
                               float zFar = 100.0f) {
            m_width = static_cast<float>(parameters.width);
            m_height = static_cast<float>(parameters.height);
            m_fx = parameters.fx;
            m_fy = parameters.fy;
            m_cx = parameters.cx;
            m_cy = parameters.cy;
            m_focalLength = parameters.focalLength;
            m_fNumber = parameters.fNumber;
            m_zNear = zNear;
            m_zFar = zFar;
            PinholeCamera::updateProjectionMatrix();
        }


        void updateProjectionMatrix() override {
            float A = m_zFar / (m_zNear - m_zFar);
            float B = -(m_zFar * m_zNear) / (m_zFar - m_zNear);
            float w = m_width;
            float h = m_height;

            float fxNorm = (2.0f * m_fx) / w;
            float fyNorm = (2.0f * m_fy) / h;
            float cxNorm = -((2.0f * m_cx) - w) / w;
            float cyNorm = -((2.0f * m_cy) - h) / h;
            if (m_flipYProjection) {
                fyNorm *= -1;
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
