/**
 * @file: MultiSense-Viewer/include/Viewer/CRLCamera/MultiSense.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-3-1, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_CRLPHYSICALCAMERA_H
#define MULTISENSE_CRLPHYSICALCAMERA_H

#include <mutex>
#include <bitset>
#include <cstdint>
#include <memory>
#include <glm/ext/matrix_float4x4.hpp>
#include <MultiSense/MultiSenseChannel.hh>

#include "Viewer/Rendering/Core/RenderDefinitions.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Modules/MultiSense/CommonHeader.h"
#include "Viewer/Modules/MultiSense/LibMultiSense/ChannelWrappers.h"
#include "Viewer/Modules/MultiSense/MultiSenseInterface.h"


namespace VkRender::MultiSense {
    /**
     * @brief Links LibMultiSense calls to be used with VkRender
     */
    class LibMultiSenseConnector : public MultiSenseInterface {

    public:
        void getCameraInfo(MultiSenseProfileInfo *profileInfo) override;

    public:
        struct ImuData {
            float x, y, z;
            double time;
            double dTime;

            bool operator==(const ImuData &d) {
                if (d.x == x && d.y == y && d.z == z)
                    return true;
                else
                    return false;
            }
        };

        /**
         * @brief container to keep \refitem crl::multisense::Channel information. This block is closely related to the \refitem Parameters UI block
         */
        struct CameraInfo {
            crl::multisense::system::DeviceInfo devInfo{};
            crl::multisense::image::Config imgConf{};
            crl::multisense::image::AuxConfig auxImgConf{};
            crl::multisense::lighting::Config lightConf{};
            crl::multisense::system::NetworkConfig netConfig{};
            crl::multisense::system::VersionInfo versionInfo{};
            std::vector<crl::multisense::system::DeviceMode> supportedDeviceModes{};
            crl::multisense::DataSource supportedSources{0};
            std::vector<uint8_t *> rawImages{};
            int sensorMTU = 0;
            crl::multisense::image::Calibration calibration{};
            glm::mat4 QMat{};
            glm::mat4 KColorMat{};
            glm::mat4 KColorMatExtrinsic{};
            float focalLength = 0.0f;
            float pointCloudScale = 1.0f;

            // IMU
            std::vector<crl::multisense::imu::Info> imuSensorInfos;
            std::vector<crl::multisense::imu::Config> imuSensorConfigs;
            uint32_t imuMaxTableIndex = 0;
            bool hasIMUSensor = false;
        };


        LibMultiSenseConnector() = default;

        ~LibMultiSenseConnector() override = default;

        void connect(std::string ip) override;
        void update(MultiSenseUpdateData* updateData) override;
        void setup() override;
        void getImage(MultiSenseStreamData* data) override;
        void stopStream(const std::string &source) override;

        void startStreaming(const std::vector<std::string> &streams) override;

        MultiSenseConnectionState connectionState() override;
        /**@brief Starts the desired stream if supported
         *
         * @param[in] stringSrc source described in string (also shown in UI)
         * @param[in] remoteHeadID which remote head to start a stream. ID of 0 can also be a non-remotehead/ VkRender m_Device
         * @return If the requested stream was started
         */
        bool start(const std::string &stringSrc, crl::multisense::RemoteHeadChannel remoteHeadID);

        /**@brief Stops the desired stream.
         *
        * @param[in] stringSrc source described in string (also shown in UI)
        * @param[in] remoteHeadID which remote head to stop a stream. ID of 0 can also be a non-remotehead/ VkRender m_Device
        * @return If the requested stream was stopped. Returns true also if the stream was never started.
        */
        bool stop(const std::string &stringSrc, crl::multisense::RemoteHeadChannel remoteHeadID);


        /**
         *
         * @param[in] channelID Which channel to get IMU data for
         * @param[in] gyro vector to fill with measurements
         * @param[in] accel vector to fill with measurements
         * @return true fetched successfully otherwise false
         */
        bool getIMUData(crl::multisense::RemoteHeadChannel channelID, std::vector<ImuData> *gyro,
                        std::vector<ImuData> *accel) const;


        /**
         * @brief get a status update from the MultiSense m_Device
         * @param[in] channelID Which channel to fetch status for
         * @param[out] msg Parameter to be filled
         * @return
         */
        bool getStatus(crl::multisense::RemoteHeadChannel channelID, crl::multisense::system::StatusMessage *msg);


        /**@brief Sets the desired stereo filter strength
         *
         * @param[in] filterStrength Value to set
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
        */
        bool setPostFilterStrength(float filterStrength, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Sets the desired gamma value to the VkRender m_Device
         *
         * @param[in] gamma Value to set
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
        */
        bool setGamma(float gamma, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Sets the desired framerate of the sensor
         *
         * @param[in] fps Value to set
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
        */
        bool setFps(float fps, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Sets the desired gain
         *
         * @param[in] gain Value to set
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
        */
        bool setGain(float gain, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Enable or disable HDR
         *
         * @param[in] hdr True = Enable // False = Disable
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
        */
        bool setHDR(bool hdr, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Configure the sensor MTU
         *
         * @param[in] mtu Which value to set
         * @return if the value was successfully set
        */
        bool setMtu(uint32_t mtu, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Get a struct of the current camera settings \ref CameraInfo
         *
         * @param[in] idx Which remote head to select
         * @return camera settings for remote head no: 'idx'
         */
        CameraInfo getCameraInfo(crl::multisense::RemoteHeadChannel idx) const;

        /**
         * @brief Sets the sensor calibration using a multisense calibration type
         * @param[in] channelID id of remote head
         * @param[in] intrinsicsFile what calibration to set
         * @param[in] intrinsicsFile what calibration to set
         * @return true successful
         */
        bool setSensorCalibration(const std::string &intrinsicsFile, const std::string &extrinsicsFile,
                                  crl::multisense::RemoteHeadChannel channelID);

        /**
         * @brief saves the sensor calibration to .yml files
         * @param[in] channelID id of remote head
         * @param[in] savePath where to save .yml files
         * @return true successful
         */
        bool saveSensorCalibration(const std::string &savePath, crl::multisense::RemoteHeadChannel channelID);

        bool getExposure(short i, bool b);

        void setIMUConfig(uint32_t tableIndex, crl::multisense::RemoteHeadChannel channelID = 0);

        void disconnect() override;

    private:
        std::unique_ptr<ChannelWrapper> m_channel = nullptr;
        CameraInfo m_channelInfo{};
        std::mutex m_channelMutex;
        MultiSenseConnectionState state;

        /**@brief Boolean to ensure the streamcallbacks called from LibMultiSense threads dont access class data while this class is being destroyed. It does happens once in a while */
        void addCallbacks(crl::multisense::RemoteHeadChannel idx);

        static void remoteHeadCallback(const crl::multisense::image::Header &header, void *userDataP);

        static void imuCallback(const crl::multisense::imu::Header &header, void *userDataP);


        std::ostream &
        writeImageIntrinics(std::ostream &stream, const crl::multisense::image::Calibration &calibration,
                            bool hasAuxCamera);

        std::ostream &
        writeImageExtrinics(std::ostream &stream, const crl::multisense::image::Calibration &calibration,
                            bool hasAuxCamera);

        // For pointclouds
        void updateQMatrix();

        // Helper function to update and log status
        template<typename Func, typename Data>
        bool updateAndLog(Func f, Data &data, const std::string &field) {
            crl::multisense::Status status = f(data);
            if (status != crl::multisense::Status_Ok) {
                Log::Logger::getInstance()->error("Failed to update '{}'. error {}", field,
                                                  crl::multisense::Channel::statusString(status));
                return false;
            }
            return true;
        }


        void getIMUConfig(crl::multisense::RemoteHeadChannel channelID);

        void updateCameraInfo();

    };


}
#endif //MULTISENSE_CRLPHYSICALCAMERA_H
