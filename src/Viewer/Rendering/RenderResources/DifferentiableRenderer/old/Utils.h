//
// Created by magnus-desktop on 11/22/24.
//

#ifndef DIFF_RENDER_UTILS_H
#define DIFF_RENDER_UTILS_H

namespace VkRender::DR::Utils {
    inline torch::Tensor L1Loss(const torch::Tensor& network_output, const torch::Tensor& gt) {
        return torch::abs(network_output - gt).mean();
    }


}

#endif //DIFF_RENDER_UTILS_H
