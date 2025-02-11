//
// Created by magnus on 10/21/24.
//

#ifndef MULTISENSE_VIEWER_SYCLDEVICESELECTOR_H
#define MULTISENSE_VIEWER_SYCLDEVICESELECTOR_H

#ifdef SYCL_ENABLED
#include <sycl/sycl.hpp>
#include "Viewer/Tools/Logger.h"
namespace VkRender {
    // Assuming SYCLDeviceType is defined as in your class
    enum class SYCLDeviceType {
        GPU,
        CPU,
        Default
    };

    class SYCLDeviceSelector {
    public:
        explicit SYCLDeviceSelector(SYCLDeviceType deviceType = SYCLDeviceType::Default) {
            selectDevice(deviceType);
        }

        sycl::queue &getQueue() { return m_queue; }

        ~SYCLDeviceSelector() {
            Log::Logger::getInstance()->info("Destroying Sycl Queue for device: {}",
                                             m_queue.get_device().get_info<sycl::info::device::name>());
            m_queue.wait();
        }

    private:
        sycl::queue m_queue;

        void selectDevice(SYCLDeviceType deviceType) {
            try {
                sycl::property_list properties{sycl::property::queue::in_order{}};
                if (deviceType == SYCLDeviceType::GPU) {
                    m_queue = sycl::queue(sycl::gpu_selector_v, properties);
                    Log::Logger::getInstance()->info("Using GPU: {}",
                                                     m_queue.get_device().get_info<sycl::info::device::name>());
                } else if (deviceType == SYCLDeviceType::CPU) {
                    m_queue = sycl::queue(sycl::cpu_selector_v, properties);
                    Log::Logger::getInstance()->info("Using CPU: {}",
                                                     m_queue.get_device().get_info<sycl::info::device::name>());
                } else {
                    m_queue = sycl::queue(sycl::default_selector_v, properties);
                    Log::Logger::getInstance()->info("Using default device: {}",
                                                     m_queue.get_device().get_info<sycl::info::device::name>());
                }
            } catch (const sycl::exception &e) {
                Log::Logger::getInstance()->error("Error selecting device: {}", e.what());
                Log::Logger::getInstance()->error("Falling back to default device.");
                sycl::property_list properties{sycl::property::queue::in_order{}};
                m_queue = sycl::queue(sycl::default_selector_v, properties);
                Log::Logger::getInstance()->info("Using device: {}",
                                                 m_queue.get_device().get_info<sycl::info::device::name>());
            }

        }
    };

    // Singleton-like manager for device selectors
    class SYCLDeviceManager {
    public:
        static SYCLDeviceManager &getInstance() {
            static SYCLDeviceManager instance;
            return instance;
        }

        // Retrieves a SYCLDeviceSelector by device type.
        std::shared_ptr<SYCLDeviceSelector> getDevice(SYCLDeviceType type) {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_devices.at(type);
        }

        // Prevent copying
        SYCLDeviceManager(const SYCLDeviceManager &) = delete;
        SYCLDeviceManager &operator=(const SYCLDeviceManager &) = delete;

        SYCLDeviceManager() {
            // Create devices once at startup
            m_devices[SYCLDeviceType::GPU]   = std::make_shared<SYCLDeviceSelector>(SYCLDeviceType::GPU);
            m_devices[SYCLDeviceType::CPU]   = std::make_shared<SYCLDeviceSelector>(SYCLDeviceType::CPU);
            m_devices[SYCLDeviceType::Default] = std::make_shared<SYCLDeviceSelector>(SYCLDeviceType::Default);
        }
    private:


        std::mutex m_mutex;
        std::map<SYCLDeviceType, std::shared_ptr<SYCLDeviceSelector>> m_devices;
    };
}
#else
// Singleton-like manager for device selectors
namespace VkRender {
    enum class SYCLDeviceType {
        GPU,
        CPU,
        Default
    };

    class SYCLDeviceSelector;
    class SYCLDeviceManager {
    public:
        SYCLDeviceManager &getInstance() {
            static SYCLDeviceManager instance;
            return instance;
        }

        // Retrieves a SYCLDeviceSelector by device type.
        std::shared_ptr<SYCLDeviceSelector> getDevice(SYCLDeviceType type) {
        return nullptr;
        }

        // Prevent copying
        SYCLDeviceManager(const SYCLDeviceManager &) = delete;
        SYCLDeviceManager &operator=(const SYCLDeviceManager &) = delete;

        SYCLDeviceManager() {
        }
    private:

    };
}
#endif

#endif //MULTISENSE_VIEWER_SYCLDEVICESELECTOR_H
