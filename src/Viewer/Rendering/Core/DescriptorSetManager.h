//
// Created by magnus-desktop on 11/3/24.
//

#ifndef DESCRIPTORSETMANAGER_H
#define DESCRIPTORSETMANAGER_H

#include <vulkan/vulkan_core.h>

#include "Viewer/Rendering/Core/VulkanDevice.h"

namespace VkRender{
    enum class DescriptorManagerType : uint32_t;



    class DescriptorSetManager {
    public:
        DescriptorSetManager(
            VulkanDevice& device,
            const std::vector<VkDescriptorSetLayoutBinding>& bindings,
            DescriptorManagerType descriptorManagerType,
            uint32_t maxDescriptorSets = 1000);
        ~DescriptorSetManager();
        VkDescriptorSetLayout& getDescriptorSetLayout(){return m_descriptorSetLayout;}

        VkDescriptorSet getOrCreateDescriptorSet(const std::vector<VkWriteDescriptorSet>& descriptorWrites);

        void freeDescriptorSets();
        void queryFreeDescriptorSets(const std::vector<VkWriteDescriptorSet>& externalDescriptorSets);
        DescriptorManagerType type() const {return m_descriptorManagerType;}

    private:
        VulkanDevice& m_device;
        VkDescriptorPool m_descriptorPool{};
        VkDescriptorSetLayout m_descriptorSetLayout{};
        std::unordered_map<size_t, VkDescriptorSet> m_descriptorSetCache;
        uint32_t m_maxDescriptorSets;
        DescriptorManagerType m_descriptorManagerType;
        uint32_t m_allocatedDescriptors = 0;

        size_t hashDescriptorImageInfo(const VkDescriptorImageInfo& imageInfo);
        size_t hashDescriptorBufferInfo(const VkDescriptorBufferInfo& bufferInfo);
        size_t hashWriteDescriptorSet(const VkWriteDescriptorSet& write);
        size_t hashDescriptorWrites(const std::vector<VkWriteDescriptorSet>& writes);
    };

}



#endif //DESCRIPTORSETMANAGER_H
