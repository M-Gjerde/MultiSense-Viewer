//
// Created by magnus-desktop on 12/16/24.
//

#ifndef CAMERACOMPONENT_H
#define CAMERACOMPONENT_H

#include "Viewer/Tools/Logger.h"
#include "Viewer/Rendering/Editors/BaseCamera.h"
#include "Viewer/Rendering/Editors/PinholeCamera.h"

namespace VkRender {
    struct CameraComponent {
        enum CameraType : uint32_t {
            PERSPECTIVE, // Classical Computer Graphics camera
            ARCBALL, // Classical Computer Graphics camera for orbiting camera around a point
            PINHOLE, // Computer Vision
        };

        // Utility function to convert CameraType to string
        static std::string cameraTypeToString(CameraType cameraType) {
            switch (cameraType) {
            case PERSPECTIVE: return "PERSPECTIVE";
            case PINHOLE: return "PINHOLE";
            case ARCBALL: return "ARCBALL";
            default: throw std::invalid_argument("Invalid CameraType");
            }
        }

        // Utility function to convert string to CameraType
        static CameraType stringToCameraType(const std::string& cameraTypeStr) {
            static const std::unordered_map<std::string, CameraType> stringToEnum = {
                {"PERSPECTIVE", PERSPECTIVE},
                {"PINHOLE", PINHOLE},
                {"ARCBALL", ARCBALL}
            };

            auto it = stringToEnum.find(cameraTypeStr);
            if (it != stringToEnum.end()) {
                return it->second;
            }
            else {
                throw std::invalid_argument("Invalid CameraType string: " + cameraTypeStr);
            }
        }

        // Utility function to get all CameraType values as an array
        static std::array<CameraType, 3> getAllCameraTypes() {
            return {PERSPECTIVE, PINHOLE, ARCBALL};
        }

        std::shared_ptr<PinholeCamera> getPinholeCamera() const {
            return std::dynamic_pointer_cast<PinholeCamera>(camera);
        }

        // we use a shared pointer as storage since most often we need to share this data with the rendering loop.
        std::shared_ptr<BaseCamera> camera = std::make_shared<BaseCamera>();
        // Possibly not required to be a pointer type, but we're passing it quite often so might be beneficial at the risk of safety
        bool render = false;
        bool flipY = false;
        CameraType cameraType = PERSPECTIVE;

        struct PinHoleParameters {
            int height = 720; // Default image height
            int width = 1280; // Default image width
            float fx = 1280.0f; // Default horizontal focal length (pixels)
            float fy = 720.0f; // Default vertical focal length (pixels)
            float cx = 640.0f; // Default principal point x-coordinate (pixels)
            float cy = 360.0f; // Default principal point y-coordinate (pixels)

            // Overload equality operator
            bool operator==(const PinHoleParameters& other) const {
                return height == other.height &&
                    width == other.width &&
                    fx == other.fx &&
                    fy == other.fy &&
                    cx == other.cx &&
                    cy == other.cy;
            }

            // Optional: Overload inequality operator for convenience
            bool operator!=(const PinHoleParameters& other) const {
                return !(*this == other);
            }
        } pinHoleParameters;

        struct ProjectionParameters {
            float near = 0.1f;
            float far = 100.0f;
            float aspect = 1.6f;
            float fov = 60.0f;

            // Overload equality operator
            bool operator==(const ProjectionParameters& other) const {
                return near == other.near &&
                    far == other.far &&
                    aspect == other.aspect &&
                    fov == other.fov;
            }

            // Overload inequality operator for convenience
            bool operator!=(const ProjectionParameters& other) const {
                return !(*this == other);
            }
        } projectionParameters;

        void updateParametersChanged() {
            switch (cameraType) {
            case PERSPECTIVE:
                camera = std::make_shared<BaseCamera>(projectionParameters.aspect, projectionParameters.fov,
                                                      projectionParameters.near, projectionParameters.far);
                camera->m_flipYProjection = flipY;
                break;
            case PINHOLE:
                camera = std::make_shared<PinholeCamera>(pinHoleParameters.width, pinHoleParameters.height,
                                                         pinHoleParameters.fx, pinHoleParameters.fy,
                                                         pinHoleParameters.cx, pinHoleParameters.cy);
                camera->m_flipYProjection = flipY;
                break;
            default:
                Log::Logger::getInstance()->warning("Camera type not implemented in scene. Reverting to Perspective");
                camera = std::make_shared<BaseCamera>();
                break;
            }
            m_updateTrigger = true;
        }

        bool& renderFromViewpoint() { return render; }

        void resetUpdateState() {
            m_updateTrigger = false;
        }

        bool updateTrigger() const { return m_updateTrigger; }

    private:
        bool m_updateTrigger = true;
    };
}
#endif //CAMERACOMPONENT_H
