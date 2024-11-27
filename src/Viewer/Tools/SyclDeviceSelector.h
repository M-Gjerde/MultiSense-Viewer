//
// Created by magnus on 10/21/24.
//

#ifndef MULTISENSE_VIEWER_SYCLDEVICESELECTOR_H
#define MULTISENSE_VIEWER_SYCLDEVICESELECTOR_H

#include <sycl/sycl.hpp>
#include <iostream>

#include "Viewer/Tools/Logger.h"

namespace VkRender {


    class SyclDeviceSelector {
    public:
        // Enum to specify the type of device to select
        enum class DeviceType {
            GPU,
            CPU,
            Default
        };

        // Constructor: Selects the device based on the provided type
        explicit SyclDeviceSelector(DeviceType deviceType = DeviceType::Default) {
            selectDevice(deviceType);
        }

        // Get the selected SYCL queue
        sycl::queue &getQueue() {
            return m_queue;
        }

    private:
        sycl::queue m_queue;

        // Function to select the appropriate device
        void selectDevice(DeviceType deviceType) {
            try {
                sycl::property_list properties{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling()};

                if (deviceType == DeviceType::GPU) {
                    // Select GPU if available
                    m_queue = sycl::queue(sycl::gpu_selector{}, properties);
                    Log::Logger::getInstance()->info("Using GPU: {}",
                                                     m_queue.get_device().get_info<sycl::info::device::name>());
                } else if (deviceType == DeviceType::CPU) {
                    // Select CPU if available
                    m_queue = sycl::queue(sycl::cpu_selector{}, properties);
                    Log::Logger::getInstance()->info("Using CPU: {}",
                                                     m_queue.get_device().get_info<sycl::info::device::name>());
                } else {
                    // Default device selector (picks the best available)
                    m_queue = sycl::queue(sycl::default_selector{}, properties);
                    Log::Logger::getInstance()->info("Using default device: {}",
                                                     m_queue.get_device().get_info<sycl::info::device::name>());
                }


                Log::Logger::getInstance()->info("Using host device: {}",m_queue.get_device().get_info<sycl::info::device::name>());

            } catch (const sycl::exception &e) {
                // Log error and fallback to default device
                Log::Logger::getInstance()->error("Error selecting device: {}", e.what());
                Log::Logger::getInstance()->error("Falling back to default device.");
                sycl::property_list properties{sycl::property::queue::in_order{}};
                // Fallback to default device with in_order property
                m_queue = sycl::queue(sycl::default_selector_v, properties);
            }
        }
    };
}

#endif //MULTISENSE_VIEWER_SYCLDEVICESELECTOR_H
