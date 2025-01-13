//
// Created by magnus-desktop on 12/16/24.
//

#ifndef CAMERACOMPONENT_H
#define CAMERACOMPONENT_H

#include "Viewer/Tools/Logger.h"
#include "Viewer/Rendering/Editors/BaseCamera.h"
#include "Viewer/Rendering/Editors/PinholeCamera.h"
#include "Viewer/Rendering/Editors/CameraDefinitions.h"

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
                case PERSPECTIVE:
                    return "PERSPECTIVE";
                case PINHOLE:
                    return "PINHOLE";
                case ARCBALL:
                    return "ARCBALL";
                default:
                    throw std::invalid_argument("Invalid CameraType");
            }
        }

        // Utility function to convert string to CameraType
        static CameraType stringToCameraType(const std::string &cameraTypeStr) {
            static const std::unordered_map<std::string, CameraType> stringToEnum = {
                    {"PERSPECTIVE", PERSPECTIVE},
                    {"PINHOLE",     PINHOLE},
                    {"ARCBALL",     ARCBALL}
            };

            auto it = stringToEnum.find(cameraTypeStr);
            if (it != stringToEnum.end()) {
                return it->second;
            } else {
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

        std::shared_ptr<BaseCamera> getPerspectiveCamera() const {
            return std::dynamic_pointer_cast<BaseCamera>(camera);
        }

        // we use a shared pointer as storage since most often we need to share this data with the rendering loop.
        std::shared_ptr<BaseCamera> camera = std::make_shared<BaseCamera>();
        // Possibly not required to be a pointer type, but we're passing it quite often so might be beneficial at the risk of safety

        CameraType cameraType = PERSPECTIVE;


        PinholeParameters pinholeParameters;
        ProjectionParameters  baseCameraParameters;
        SharedCameraSettings cameraSettings;

        void updateParametersChanged() {
            switch (cameraType) {
                case PERSPECTIVE:
                    camera = std::make_shared<BaseCamera>(cameraSettings, baseCameraParameters);
                    break;
                case PINHOLE:
                    camera = std::make_shared<PinholeCamera>(cameraSettings, pinholeParameters);
                    break;
                default:
                    Log::Logger::getInstance()->warning(
                            "Camera type not implemented in scene. Reverting to Perspective");
                    camera = std::make_shared<BaseCamera>();
                    break;
            }
            m_updateTrigger = true;
        }
        bool render = false;

        bool &renderFromViewpoint() { return render; }

        void resetUpdateState() {
            m_updateTrigger = false;
        }

        bool updateTrigger() const { return m_updateTrigger; }

    private:
        bool m_updateTrigger = true;
    };
}
#endif //CAMERACOMPONENT_H
