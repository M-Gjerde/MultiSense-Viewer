//
// Created by mgjer on 15/01/2024.
//

#ifndef MULTISENSE_VIEWER_MULTISENSEDEVICE_H
#define MULTISENSE_VIEWER_MULTISENSEDEVICE_H

#include <string>
#include <unordered_map>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <Viewer/Tools/Macros.h>

DISABLE_WARNING_PUSH
DISABLE_WARNING_ALL
#include <imgui.h>
DISABLE_WARNING_POP
#include <json.hpp>

namespace crl {
    namespace multisense {
        typedef int16_t RemoteHeadChannel;
    }
}

namespace VkRender {

    /**
 * @brief Page within pages. Which layout is chosen for previews
 */
    typedef enum PreviewLayout {
        CRL_PREVIEW_LAYOUT_NONE = 0,
        CRL_PREVIEW_LAYOUT_SINGLE = 1,
        CRL_PREVIEW_LAYOUT_DOUBLE = 2,
        CRL_PREVIEW_LAYOUT_DOUBLE_SIDE_BY_SIDE = 3,
        CRL_PREVIEW_LAYOUT_QUAD = 4,
        CRL_PREVIEW_LAYOUT_NINE = 5
    } PreviewLayout;


    /**
 * @brief Identifier for different pages in the GUI.
 */
    typedef enum page {
        CRL_TAB_NONE = 10,
        CRL_TAB_2D_PREVIEW = 11,
        CRL_TAB_3D_POINT_CLOUD = 12,

        CRL_TAB_PREVIEW_CONTROL = 20,
        CRL_TAB_SENSOR_CONFIG = 21
    } Page;


    /**
 * @brief Hardcoded Camera resolutions using descriptive enums
 */
    typedef enum CRLCameraResolution {
        CRL_RESOLUTION_960_600_64 = 0,
        CRL_RESOLUTION_960_600_128 = 1,
        CRL_RESOLUTION_960_600_256 = 2,
        CRL_RESOLUTION_1920_1200_64 = 3,
        CRL_RESOLUTION_1920_1200_128 = 4,
        CRL_RESOLUTION_1920_1200_256 = 5,
        CRL_RESOLUTION_1024_1024_128 = 6,
        CRL_RESOLUTION_2048_1088_256 = 7,
        CRL_RESOLUTION_NONE = 8
    } CRLCameraResolution;


/**
 * @brief What connection state a device seen in the side bar can be in.
 */
    typedef enum ArConnectionState {
        CRL_STATE_CONNECTED = 0,                /// Not implemented cause its the same state as ACTIVE
        CRL_STATE_CONNECTING = 1,               /// Just clicked connect
        CRL_STATE_ACTIVE = 2,                   /// Normal operation
        CRL_STATE_INACTIVE = 3,                 /// Not sure ?
        CRL_STATE_LOST_CONNECTION = 4,          /// Lost connection... Will go into UNAVAILABLE state
        CRL_STATE_RESET = 5,                    /// Reset this device (Put it in disconnected state) if it was clicked while being in ACTIVE state or another device was activated
        CRL_STATE_DISCONNECTED = 6,             /// Not currently active but can be activated by clicking it
        CRL_STATE_UNAVAILABLE = 7,              /// Could not be found on any network adapters
        CRL_STATE_DISCONNECT_AND_FORGET = 8,    /// Device is removed and the profile in crl.ini is deleted
        CRL_STATE_REMOVE_FROM_LIST = 9,         /// Device is removed from sidebar list on next frame
        CRL_STATE_INTERRUPT_CONNECTION = 10,    /// Stop attempting to connect (For closing application quickly and gracefully)
        CRL_STATE_JUST_ADDED = 11,               /// Skip to connection flow, just added into sidebar list
        CRL_STATE_JUST_ADDED_WINDOWS = 12        /// Windows autoconnect needs some extra time to propogate
    } ArConnectionState;


/**
 * @brief Identifier for each preview window and point cloud type. Loosely tied to the preview scripts
 * Note: Numbering matters
 */
    typedef enum StreamWindowIndex {
        // First 0 - 8 elements also correspond to array indices. Check upon this before adding more PREVIEW indices
        CRL_PREVIEW_ONE = 0,
        CRL_PREVIEW_TWO = 1,
        CRL_PREVIEW_THREE = 2,
        CRL_PREVIEW_FOUR = 3,
        CRL_PREVIEW_FIVE = 4,
        CRL_PREVIEW_SIX = 5,
        CRL_PREVIEW_SEVEN = 6,
        CRL_PREVIEW_EIGHT = 7,
        CRL_PREVIEW_NINE = 8,
        CRL_PREVIEW_POINT_CLOUD = 9,
        CRL_PREVIEW_TOTAL_MODES = CRL_PREVIEW_POINT_CLOUD + 1,
        // Other flags
    } StreamWindowIndex;


    /**
     * Shared data for image effects
     */
    struct ImageEffectData {
        float minDisparityValue = 0.0f;
        float maxDisparityValue = 255.0f;

    };

    /**
    * Image effect options for each preview window
    */
    struct ImageEffectOptions {
        bool normalize = false;
        bool interpolation = false;
        bool depthColorMap = false;
        bool magnifyZoomMode = false;
        bool edgeDetection = false;
        bool blur = false;
        bool emboss = false;
        bool sharpening = false;
        VkRender::ImageEffectData data;
    };

    /**
 * @brief Ties together the preview windows for each MultiSense channel
 * Contain possible streams and resolutions for each channel
 */
    struct ChannelInfo {
        crl::multisense::RemoteHeadChannel index = 0;
        std::vector<std::string> availableSources{};
        /** @brief Current connection state for this channel */
        ArConnectionState state = CRL_STATE_DISCONNECTED;
        std::vector<std::string> modes{};
        uint32_t selectedModeIndex = 0;
        CRLCameraResolution selectedResolutionMode = CRL_RESOLUTION_NONE;
        std::vector<std::string> requestedStreams{};
        std::vector<std::string> enabledStreams{};
        bool updateResolutionMode = true;
    };

    /**
 * @brief Information block for each Preview window, should support 1-9 possible previews. Contains stream names and info to presented to the user.
 * Contains a mix of GUI related and MultiSense Device related info.
 */
    struct PreviewWindow {
        std::vector<std::string> availableSources{}; // Human-readable names of camera sources
        std::vector<std::string> availableRemoteHeads{};
        std::string selectedSource = "Idle";
        uint32_t selectedSourceIndex = 0;
        crl::multisense::RemoteHeadChannel selectedRemoteHeadIndex = 0;
        float xPixelStartPos = 0;
        float yPixelStartPos = 0;
        float row = 0;
        float col = 0;

        VkRender::ImageEffectOptions effects;
        bool enableZoom = true;
        bool isHovered = false;
        ImVec2 popupPosition = ImVec2(0.0f,
                                      0.0f); // Position of popup window for image effects. (Used to update popup position when preview window is scrolled)
        ImVec2 popupWindowSize = ImVec2(0.0f,
                                        0.0f); // Position of popup window for image effects. (Used to update popup position when preview window is scrolled)
        bool updatePosition = false;
        std::string name = "None";
    };


/**
 * @brief Adjustable sensor parameters
 */
    struct LightingParams {
        float dutyCycle = 1.0f;
        int selection = -1;
        bool flashing = true;
        float numLightPulses = 3;
        float startupTime = 0;
        bool update = false;
    };

/**
 * @brief Adjustable sensor parameters
 */
    struct ExposureParams {
        bool autoExposure{};
        int exposure = 20000;
        uint32_t autoExposureMax = 20000;
        uint32_t autoExposureDecay = 7;
        float autoExposureTargetIntensity = 0.5f;
        float autoExposureThresh = 0.9f;
        uint16_t autoExposureRoiX = 0;
        uint16_t autoExposureRoiY = 0;
        uint16_t autoExposureRoiWidth = 0; // 0 == crl::multisense::Roi_Full_Image;
        uint16_t autoExposureRoiHeight = 0; //0 == crl::multisense::Roi_Full_Image;
        uint32_t currentExposure = 0;

        bool update = false;
    };

    struct AUXConfig {
        float gain = 1.7f;
        float whiteBalanceBlue = 1.0f;
        float whiteBalanceRed = 1.0f;
        bool whiteBalanceAuto = true;
        uint32_t whiteBalanceDecay = 3;
        float whiteBalanceThreshold = 0.5f;
        bool hdr = false;
        float gamma = 2.0f;
        bool sharpening = false;
        float sharpeningPercentage = 50.0f;
        int sharpeningLimit = 50;

        ExposureParams ep{};
        bool update = false;
    };

    struct ImageConfig {
        float gain = 1.0f;
        float fps = 30.0f;
        float stereoPostFilterStrength = 0.5f;
        int mtu = 1400;
        bool hdrEnabled = false;
        float gamma = 2.0f;
        bool hdr = false;

        ExposureParams ep{};
        bool update = false;

    };

    struct Metadata {
        int custom = 0; // 1 True, 0 False
        char logName[1024] = "Log no. 1";
        char location[1024] = "Test site X";
        char recordDescription[1024] = "Offroad navigation";
        char equipmentDescription[1024] = "MultiSense mounted on X";
        char camera[1024] = "MultiSense S30";
        char collector[1024] = "John Doe";
        char tags[1024] = "self driving, test site x, multisense";

        char customField[1024 * 32] =
                "road_type = Mountain Road\n"
                "weather_conditions = Clear sky in the beginning, started to raind around midday followed by harsh winds\n"
                "project_code = IRAD-2023\n\n\n\n";
        /**@brief JSON object containing the above fields in JSON format if the custom metadata has been set */
        nlohmann::json JSON = nullptr;
        bool parsed = false;
    };

    struct RecordDataInfo {
        Metadata metadata;

        bool showCustomMetaDataWindow = false;
        /**@brief location for which this m_Device should save recorded frames **/
        std::string frameSaveFolder;
        /**@brief location for which this m_Device should save recorded point clouds **/
        std::string pointCloudSaveFolder;
        /**@brief location for which this m_Device should save recorded point clouds **/
        std::string imuSaveFolder;
        /**@brief Flag to decide if user is currently recording frames */
        bool frame = false;
        /**@brief Flag to decide if user is currently recording point cloud */
        bool pointCloud = false;
        /**@brief Flag to decide if user is currently recording IMU data */
        bool imu = false;

        RecordDataInfo() {
            frameSaveFolder.resize(1024);
            pointCloudSaveFolder.resize(1024);
        }
    };

/**
 * @brief Adjustable sensor parameters
 */
    struct CalibrationParams {
        bool update = false;
        bool save = false;
        std::string intrinsicsFilePath = "Path/To/Intrinsics.yml";
        std::string extrinsicsFilePath = "Path/To/Extrinsics.yml";
        std::string saveCalibrationPath = "Path/To/Dir";
        bool updateFailed = false;
        bool saveFailed = false;

        CalibrationParams() {
            intrinsicsFilePath.resize(255);
            extrinsicsFilePath.resize(255);
            saveCalibrationPath.resize(255);
        }
    };

/** @brief  MultiSense Device configurable parameters. Should contain all adjustable parameters through LibMultiSense */
    struct Parameters {
        LightingParams light{};
        CalibrationParams calib{};
        ImageConfig stereo;
        AUXConfig aux;

        bool updateGuiParams = true;
    };


    /**
     * @brief Data block for displaying pixel intensity and depth in textures on mouse hover
     */
    struct CursorPixelInformation {
        uint32_t x = 0, y = 0;
        uint32_t r = 0, g = 0, b = 0;
        uint32_t intensity = 0;
        float depth = 0;
    };


    /**
    * @brief UI Block for a MultiSense Device connection. Contains connection information and user configuration such as selected previews, recording info and more..
    */
    struct Device {
        /** @brief Profile Name information  */
        std::string name = "Profile #1";
        /** @brief Name of the camera connected  */
        std::string cameraName;
        /** @brief Identifier of the camera connected  */
        std::string serialName;
        /** @brief IP of the connected camera  */
        std::string IP;
        /** @brief Default IP of the connected camera  */
        std::string defaultIP = "10.66.171.21";
        /** @brief Name of the network adapter this camera is connected to  */
        std::string interfaceName;
        /** @brief InterFace index of network adapter */
        uint32_t interfaceIndex = 0;
        /** @brief Descriptive text of interface if provided by the OS (Required on windows) */
        std::string interfaceDescription;
        /** @brief Flag for registering if device is clicked in sidebar */
        bool clicked = false;
        /** @brief Current connection state for this profile */
        ArConnectionState state = CRL_STATE_UNAVAILABLE;
        /** @brief is this device a remote head or a MultiSense camera */
        bool isRemoteHead = false;
        /** @brief Order each preview window with a index flag*/
        std::unordered_map<StreamWindowIndex, PreviewWindow> win{};
        /** @brief Put each camera interface channel (crl::multisense::Channel) 0-3 in a vector. Contains GUI info for each channel */
        std::vector<ChannelInfo> channelInfo{};
        /** @brief object containing all adjustable parameters to the camera */
        Parameters parameters{};
        /** @brief Pixel information from renderer, on mouse hover for textures */
        std::unordered_map<StreamWindowIndex, CursorPixelInformation> pixelInfo{};
        /** @brief Pixel information scaled after zoom */
        std::unordered_map<StreamWindowIndex, CursorPixelInformation> pixelInfoZoomed{};
        /** @brief Indices for each remote head if connected.*/
        std::vector<crl::multisense::RemoteHeadChannel> channelConnections{};
        /** @brief Config index for remote head. Presented as radio buttons for remote head selection in the GUI. 0 is for MultiSense */
        crl::multisense::RemoteHeadChannel configRemoteHead = 0;
        /** @brief Flag to signal application to revert network settings on application exit */
        bool systemNetworkChanged = false;
        /** Interrupt connection flow if users exits program. */
        bool interruptConnection = false;
        /** @brief Which compression method to use. Default is tiff which offer no compression */
        std::string saveImageCompressionMethod = "tiff";
        /** @brief Not a physical device just testing the GUI */
        bool simulatedDevice = false;
        /** @brief If possible then use the IMU in the camera */
        bool enableIMU = true;
        /** @brief If possible then use the IMU in the camera */
        int rateTableIndex = 2;

        /** @brief 0 : luma // 1 : Color  */
        int useAuxForPointCloudColor = 1;

        /** @brief Following is UI elements settings for the active device **/
        Page controlTabActive = CRL_TAB_PREVIEW_CONTROL;
        /** @brief Which TAB this preview has selected. 2D or 3D view. */
        Page selectedPreviewTab = CRL_TAB_2D_PREVIEW;
        /** @brief What type of layout is selected for this device*/
        PreviewLayout layout = CRL_PREVIEW_LAYOUT_SINGLE;
        /** @brief IF 3D area should be extended or not */
        bool extend3DArea = true;
        /** @brief If the connected device has a color camera */
        bool hasColorCamera = false;
        bool hasImuSensor = false;

        /** @brief If we managed to update all the device configs */
        bool updateDeviceConfigsSucceeded = false;

        RecordDataInfo record;
    };


}
#endif //MULTISENSE_VIEWER_MULTISENSEDEVICE_H
