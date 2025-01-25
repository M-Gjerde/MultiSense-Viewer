//
// Created by magnus-desktop on 11/22/24.
//

#include "Viewer/Rendering/RenderResources/DifferentiableRenderer/DiffRenderEntry.h"

#include <cstdint>
#include <torch/torch.h>


namespace VkRender::DR {

    DiffRenderEntry::DiffRenderEntry()
            : renderer(64), device(torch::kCPU) {
        // Device configuration
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
        } else {
            device = torch::Device(torch::kCPU);
        }


        renderer.to(device);

        m_package = std::make_unique<OptimizerPackage>(renderer, device);
        m_package->to(device);

        // Render target and initial images
        m_package->initialImage = renderer.forward(m_package->initialCenter, m_package->initialVariance).to(device);
        m_package->targetImage = renderer.forward(m_package->targetCenter, m_package->targetVariance).to(device);


    }

    void DiffRenderEntry::setup() {
        // Enable anomaly detection for debugging
        torch::autograd::AnomalyMode::set_enabled(true);

        // Additional setup if needed
    }

    void DiffRenderEntry::update() {
        if (iterateOnUpdate) {
            iterate();
        }

    }

    void DiffRenderEntry::iterate() {
        m_package->optimizer->zero_grad();

        // Render the image with the current parameters
        torch::Tensor rendered_image = renderer.forward(m_package->center, m_package->variance);

        // Compute loss (e.g., L1 loss between rendered and target images)
        torch::Tensor loss = torch::abs(rendered_image - m_package->targetImage).mean();
        // Backpropagation
        loss.backward();

        // Update parameters
        m_package->optimizer->step();

        // Clamp variance to prevent negative or zero values
        m_package->variance.data().clamp_(1e-3);


// Debugging output
        std::cout << "Iteration " << m_iteration << ": Loss = " << loss.item<float>() << ", Center = [";

        torch::Tensor centerDataCpu = m_package->center.clone().detach().cpu().contiguous();
        for (int j = 0; j < m_package->center.size(0); ++j) {
            std::cout << centerDataCpu[j].item<float>();
            if (j < m_package->center.size(0) - 1) std::cout << ", ";
        }

        std::cout << "], Variance = [";

        torch::Tensor varianceDataCpu = m_package->variance.clone().detach().cpu().contiguous();
        for (int j = 0; j < m_package->variance.size(0); ++j) {
            std::cout << varianceDataCpu[j].item<float>();
            if (j < m_package->variance.size(0) - 1) std::cout << ", ";
        }

        std::cout << "]" << std::endl;


        // Set learning rates for each parameter group
        auto &param_groups = m_package->optimizer->param_groups();
        // Set learning rate for 'center' parameter group (index 0)
        float mod = 1.0f;
        auto &group0_options = dynamic_cast<torch::optim::AdamOptions &>(param_groups[0].options());
        group0_options.set_lr(1.0f * mod);

        // Set learning rate for 'variance' parameter group (index 1)
        //auto &group1_options = dynamic_cast<torch::optim::AdamOptions &>(param_groups[1].options());
        //group1_options.set_lr(0.25f * mod);


        std::cout << "======================== Optimizer Parameter Groups ========================\n";
        int group_index = 0;
        for (const auto& group : m_package->optimizer->param_groups()) {
            std::cout << "Group " << group_index++ << ":\n";
            std::cout << "  Learning Rate: "
                      << static_cast<const torch::optim::AdamOptions&>(group.options()).lr() << "\n";

            int param_index = 0;
            for (const auto& param : group.params()) {
                std::cout << "  Param " << param_index++ << ":\n";
                std::cout << "    Values: " << param.detach().cpu() << "\n";
                if (param.grad().defined()) {
                    std::cout << "    Gradients: " << param.grad().detach().cpu() << "\n";
                } else {
                    std::cout << "    Gradients: [None]\n";
                }
            }
        }
        std::cout << "=============================================================================\n";

        m_iteration++;
        // Optionally, clamp other parameters or apply constraints
    }

    torch::Tensor DiffRenderEntry::getGradient() const {
        // Return the gradient of the center parameter
        return m_package->center.grad().clone();
    }

    void DiffRenderEntry::setParameters(const torch::Tensor &newCenter, const torch::Tensor &newVariance) {
        m_package->center = newCenter.clone().detach().to(device).requires_grad_(true);
        m_package->variance = newVariance.clone().detach().to(device).requires_grad_(true);

        // Reinitialize the optimizer with the new parameters
        m_package->optimizer = std::make_unique<torch::optim::Adam>(
                torch::optim::Adam({m_package->center, m_package->variance})
        );

        // Optionally set learning rates or other optimizer options
    }

    std::pair<torch::Tensor, torch::Tensor> DiffRenderEntry::getParameters() const {
        return {m_package->center.clone(), m_package->variance.clone()};
    }

    void DiffRenderEntry::setIterateOnUpdate(bool iterate) {
        iterateOnUpdate = iterate;
    }

// Implementation of OptimizerPackage

    DiffRenderEntry::OptimizerPackage::OptimizerPackage(RenderGaussian &renderer, torch::Device device) {
        // Initialize tensors
        initialCenter = torch::tensor({32.0, 32.0}, torch::TensorOptions().device(device).dtype(torch::kFloat));
        initialVariance = torch::tensor({2.0, 15.0}, torch::TensorOptions().device(device).dtype(torch::kFloat));

        // Parameters with requires_grad = true
        center = initialCenter.clone().detach().requires_grad_(true);
        variance = initialVariance.clone().detach().requires_grad_(true);

        // Target position
        targetCenter = torch::tensor({42.0, 48.0}, torch::TensorOptions().device(device).dtype(torch::kFloat));
        targetVariance = torch::tensor({25.0, 4.0}, torch::TensorOptions().device(device).dtype(torch::kFloat));


        optimizer = std::make_unique<torch::optim::Adam>(
                torch::optim::Adam(
                        std::vector<torch::optim::OptimizerParamGroup>{
                                torch::optim::OptimizerParamGroup({center}),  // Group 1 for center
                                torch::optim::OptimizerParamGroup({variance}) // Group 2 for variance
                        }
                )
        );




    }

    void DiffRenderEntry::OptimizerPackage::to(torch::Device device) {
        initialCenter = initialCenter.to(device);
        initialVariance = initialVariance.to(device);
        center = center.to(device).detach().requires_grad_(true);
        variance = variance.to(device).detach().requires_grad_(true);
        targetCenter = targetCenter.to(device);
        targetVariance = targetVariance.to(device);
        //targetImage = targetImage.to(device);
        //initialImage = initialImage.to(device);

        // Update optimizer parameters to the new device
        optimizer = std::make_unique<torch::optim::Adam>(
                torch::optim::Adam(std::vector<torch::Tensor>{center, variance})
        );
    }

} // namespace VkRender::DR
