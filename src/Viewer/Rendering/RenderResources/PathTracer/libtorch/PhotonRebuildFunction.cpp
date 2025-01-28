//
// Created by magnus on 1/27/25.
//

#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildFunction.h"
#include "stb_image_write.h"
#include <random>
#include <glm/gtx/quaternion.hpp>

namespace VkRender::PathTracer {
    static void save_gradient_to_png(torch::Tensor gradient, const std::string& filename) {
        // Ensure the tensor is on CPU and in float32
        gradient = gradient.detach().cpu().to(torch::kFloat32);

        // Normalize to [0, 255]
        auto min = gradient.min().item<float>();
        auto max = gradient.max().item<float>();
        auto normalized = (gradient - min) / (max - min) * 255.0;

        // Convert to uint8
        auto uint8_tensor = normalized.to(torch::kUInt8);

        // Get raw pointer
        uint8_t* data = uint8_tensor.data_ptr<uint8_t>();

        // Get dimensions
        int width = gradient.size(1);
        int height = gradient.size(0);
        // Save as PNG using stb_image_write
        stbi_write_png(filename.c_str(), width, height, 1, data, width);


        std::filesystem::path filenamePath = filename;
        std::ofstream file(filenamePath.replace_extension(".pfm"), std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filenamePath.replace_extension(".pfm").string());
        }
        // Write the PFM header
        // "PF" indicates a color image. Use "Pf" for grayscale.
        file << "Pf\n" << width << " " << height << "\n-1.0\n";

        // PFM expects the data in binary format, row by row from top to bottom
        // Assuming your m_imageMemory is in RGBA format with floats

        // Allocate a temporary buffer for RGB data
        std::vector<float> rgbData(width * height);

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t pixelIndex = (y * width + x);
                uint32_t rgbIndex = (y * width + x);
                rgbData[rgbIndex + 0] = gradient.data_ptr<float>()[pixelIndex] * 255; // R
            }
        }

        // Write the RGB float data
        file.write(reinterpret_cast<const char*>(rgbData.data()), rgbData.size() * sizeof(float));

        if (!file) {
            throw std::runtime_error("Failed to write PFM data to file: " + filenamePath.replace_extension(".pfm").string());
        }

        file.close();
    }

    static float randomFloat() {
        thread_local std::random_device rd; // Seed for random generator
        thread_local std::mt19937 generator(rd()); // Mersenne Twister engine
        thread_local std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        return distribution(generator);
    }

    static glm::vec3 samplePointOnDisk(
        const glm::vec3& center,
        const glm::vec3& normal,
        float radius) {
        // Generate random samples for radius and angle
        float r = radius * sqrt(randomFloat());
        float theta = 2.f * M_PI * randomFloat();

        // Construct orthonormal basis for the disk plane
        glm::vec3 u = normalize(cross(normal, glm::vec3(0.0f, 0.0f, 1.0f)));
        if (length(u) < 1e-5f) {
            u = normalize(cross(normal, glm::vec3(0.0f, 1.0f, 0.0f)));
        }
        glm::vec3 v = cross(normal, u);

        float dx = r * cos(theta);
        float dy = r * sin(theta);

        glm::vec3 offset = dx * u + dy * v;
        return center + offset;
    }


    static glm::vec3 sampleDirectionTowardAperture(
        const glm::vec3& lightPos,
        const glm::vec3& apertureCenter,
        const glm::vec3& apertureNormal,
        float apertureRadius) {
        // Pick a random point on the aperture disk
        glm::vec3 lensHitPoint = samplePointOnDisk(apertureCenter, apertureNormal, apertureRadius);
        // Compute direction from light source to lens point
        glm::vec3 dir = lensHitPoint - lightPos;
        return normalize(dir);
    }


    bool getCameraPlaneIntersection(
        const glm::vec3& rayOriginWorld,
        const glm::vec3& rayDirWorld,
        const glm::mat4& entityTransform,
        glm::vec3& hitPointCam, // out: intersection in camera space
        float& tIntersect, // out: parameter t
        float& contributionScore // out: parameter contributionScore
    ) {
        // Camera plane normal in world space

        // Camera plane normal in world space
        glm::vec3 cameraPlaneNormalWorld = glm::normalize(
            glm::mat3(entityTransform) * glm::vec3(0.0f, 0.0f, -1.0f));
        glm::vec3 cameraPlanePointWorld = glm::vec3(
            entityTransform *
            glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)); // A point on the plane
        // Ray-plane intersection

        // Ray-plane intersection calculation
        float denom = glm::dot(cameraPlaneNormalWorld, rayDirWorld);
        if (std::abs(denom) < 1e-6) {
            return false; // Ray is parallel to the plane
        }
        glm::vec3 p0l0 = cameraPlanePointWorld - rayOriginWorld;
        float t = glm::dot(p0l0, cameraPlaneNormalWorld) / denom;
        if (t < 1e-6) {
            return false; // Intersection is behind the ray origin
        }
        glm::vec3 intersectionPoint = rayOriginWorld + t * rayDirWorld;

        float hitx = intersectionPoint.x;
        float hity = intersectionPoint.y;
        float hitz = intersectionPoint.z;

        glm::vec3 intersectionCamSpace = glm::vec3(
            glm::inverse(entityTransform) * glm::vec4(intersectionPoint, 1.0f));

        float hitCx = intersectionCamSpace.x;
        float hitCy = intersectionCamSpace.y;
        float hitCz = intersectionCamSpace.z;

        // Sensor plane bounds in camera space
        float halfW = (960.0f * 0.5f) / 600.0f;
        float halfH = (600.0f * 0.5f) / 600.0f;

        // Check bounds
        if (intersectionCamSpace.x < -halfW || intersectionCamSpace.x > halfW ||
            intersectionCamSpace.y < -halfH || intersectionCamSpace.y > halfH) {
            return false; // Outside sensor bounds
        }

        float cosAngle = std::abs(glm::dot(cameraPlaneNormalWorld, rayDirWorld)); // Ensure positive cosine
        contributionScore = cosAngle; // Higher cosine means closer to perpendicular


        // If we reach here, the ray intersects the camera plane within bounds
        hitPointCam = intersectionPoint; // Intersection point in camera space
        tIntersect = t; // Distance along the ray to the intersection
        return true;
    }

    // Constructs an orthonormal basis (T, B, N) given a normal N.
    inline void buildTangentBasis(const glm::vec3& N, glm::vec3& T, glm::vec3& B) {
        // Any vector not collinear with N will do for "temp"
        glm::vec3 temp = (fabs(N.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);

        T = glm::normalize(glm::cross(temp, N));
        B = glm::cross(N, T);
        // Now T, B, N is an orthonormal basis
    }

    // ---------------------------------------------------------------------
    //  sampleGaussianPositionAndNormal
    // ---------------------------------------------------------------------
    static void sampleGaussianPositionAndNormal(
        glm::vec3 position,
        glm::vec3 normal,
        glm::vec2 scale,
        glm::vec3& outPos,
        glm::vec3& outNormal,
        float& emissionPower) {
        // ------------------------------------------------------------------
        // 1. Prepare the normal, find two tangent vectors for the plane.
        // ------------------------------------------------------------------
        glm::vec3 n = glm::normalize(normal);
        glm::vec3 t1;
        glm::vec3 t2;
        buildTangentBasis(n, t1, t2);
        // 2. Repeatedly draw samples from a standard normal, then scale
        //    them by (sigma_x, sigma_y), until they fall inside the ellipse.

        float x, y; // final offsets in local 2D coords
        while (true) {
            // (a) Generate two uniform randoms in [0,1)
            float u1 = randomFloat();
            float u2 = randomFloat();

            // (b) Box-Muller transform for standard normal
            float r = sqrtf(-2.0f * logf(u1));
            float theta = 2.0f * M_PIf * u2;
            float z0 = r * cosf(theta); // ~ N(0,1)
            float z1 = r * sinf(theta); // ~ N(0,1)

            // (c) Scale by anisotropic stddev (sigma_x, sigma_y)
            x = z0 * scale.x;
            y = z1 * scale.y;

            // (d) Check elliptical boundary
            //     If scale.x=1 => maximum distance is 1 meter in X
            //     If scale.y=1 => maximum distance is 1 meter in Y
            //     For ellipse: (x/σx)^2 + (y/σy)^2 <= 1
            float ellipseParam = (x * x) / (scale.x * scale.x)
                + (y * y) / (scale.y * scale.y);
            if (ellipseParam <= 1.0f) {
                // Valid truncated sample; break out
                break;
            }
        }
        // ------------------------------------------------------------------
        // 3. Offset the center by (x, y) in the plane spanned by (t1, t2).
        // ------------------------------------------------------------------
        glm::vec3 offset = x * t1 + y * t2;
        outPos = position + offset;

        // Normal remains the same as the Gaussian's normal
        outNormal = n;

        // Emission power (or flux) from your stored value
        emissionPower = 1.0f;
    }

    void printProgressBar(float progress) {
        const int barWidth = 50; // Width of the progress bar
        std::cout << "\r["; // Carriage return to overwrite the line
        int pos = static_cast<int>(barWidth * progress);
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(2) << (progress * 100.0) << "%";
        std::cout.flush();
    }


    torch::Tensor PhotonRebuildFunction::forward(torch::autograd::AutogradContext* ctx,
                                                 PhotonTracer::Settings& settings, PhotonTracer* pathTracer,
                                                 torch::Tensor positions, torch::Tensor scales,
                                                 torch::Tensor normals, torch::Tensor emissions,
                                                 torch::Tensor colors,
                                                 torch::Tensor specular,
                                                 torch::Tensor diffuse) {
        // =================
        // 1) Save for backward any Tensors or scalar values you need
        //    to compute derivatives later. For example:
        ctx->save_for_backward({positions, scales, normals, emissions, colors, specular, diffuse});
        ctx->saved_data["pathTracer"] = reinterpret_cast<int64_t>(pathTracer);

        // If you have non-tensor data you want in backward(), you can store
        // them as attributes:
        ctx->saved_data["settings"] = reinterpret_cast<int64_t>(&settings); // example
        // or store the pointer as a raw pointer or shared pointer if you prefer
        // (but be careful with lifetimes).

        // =================
        // 2) Run your path tracer code that renders an image.

        // Example pseudo-code:

        pathTracer->update(settings);

        // Suppose the path tracer writes out to pathTracer->m_imageMemory,
        // with shape [height * width] or [height * width * channels].
        // We'll build a Torch tensor from that raw memory.

        // For illustration:
        int64_t height = pathTracer->m_height;
        int64_t width = pathTracer->m_width;
        float* rawImage = pathTracer->getImage();
        // e.g. a float[height * width]  (gray) or float[height * width * 3]

        // Wrap it in a Torch tensor.
        // Note from_blob does not take ownership, so we typically clone().
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto output = torch::from_blob(rawImage, {height, width}, options).clone();

        // Return the rendered image
        return output;
    }


    torch::autograd::tensor_list PhotonRebuildFunction::backward(torch::autograd::AutogradContext* ctx,
                                                                 torch::autograd::tensor_list grad_outputs) {
        // Usually, the forward returned 1 tensor => grad_outputs.size() == 1
        // grad_outputs[0] is d(L)/d(output).

        auto dLoss_dRenderedImage = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto positions = saved[0];
        auto scales = saved[1];
        auto normals = saved[2];
        auto emissions = saved[3];
        auto colors = saved[4];
        auto diffuse = saved[5];
        auto specular = saved[6];

        // Retrieve the path tracer pointer
        auto pathTracerRaw = ctx->saved_data["pathTracer"].toInt();
        PhotonTracer* pathTracer = reinterpret_cast<PhotonTracer*>(pathTracerRaw);
        // Retrieve the path tracer pointer
        auto settingsPtr = ctx->saved_data["settings"].toInt();
        PhotonTracer::Settings* settings = reinterpret_cast<PhotonTracer::Settings*>(settingsPtr);

        auto dLossA = dLoss_dRenderedImage.accessor<float, 2>();

        // We'll do a trivial zero gradient for demonstration
        auto grad_positions = torch::zeros_like(positions);
        auto grad_scales = torch::zeros_like(scales);
        auto grad_normals = torch::zeros_like(normals);
        //grad_positions.index_put_({at::indexing::Slice(), 1}, 1.0);
        save_gradient_to_png(dLoss_dRenderedImage,
                             "gradients/gradient_" + std::to_string(pathTracer->m_renderInformation->frameID) + ".png");

        auto gradPosA = grad_positions.accessor<float, 2>(); // [i, 3]
        auto posA = positions.accessor<float, 2>(); // [i, 3]
        auto normalA = normals.accessor<float, 2>(); // [i, 3]
        auto scaleA = scales.accessor<float, 2>(); // [i, 3]
        auto emissionA = emissions.accessor<float, 2>(); // [i, 3]

        pathTracer->m_backwardInfo.gradientImage = dLoss_dRenderedImage.data_ptr<float>();
        uint32_t height = 600;
        uint32_t width = 960;
        auto gradients = pathTracer->backward(*settings);

        glm::vec3 grad = *gradients.sumGradients;

        float grad_x = grad.x;
        float grad_y = grad.y;
        float grad_z = grad.z;

        /*
        for (int py = 0; py < height; ++py) {
            for (int px = 0; px < width; ++px) {
                float dLoss_dThisPixel = dLossA[py][px];
                if (dLoss_dThisPixel == 0)
                    continue;
                grad_x += dLoss_dThisPixel * gradients.sumGradients[0].x;
                grad_y += dLoss_dThisPixel * gradients.sumGradients[0].y;
                grad_z += dLoss_dThisPixel * gradients.sumGradients[0].z;
            }
        }
        */

        gradPosA[0][0] = grad_x;
        gradPosA[0][1] = grad_y;
        gradPosA[0][2] = grad_z;
        //glm::vec4 gradient_world = pathTracer->m_cameraTransform.getTransform() * glm::vec4(I_pred_d, 1.0f);

        /*
        {
            float etmin = FLT_MAX;
            float cosAngleHit = 0.0f;
            glm::vec3 cameraHitPointWorld(0.0f);
            getCameraPlaneIntersection(emissionOrigin, emissionDirection,
                                       pathTracer->m_cameraTransform.getTransform(),
                                       cameraHitPointWorld, etmin, cosAngleHit);
            glm::mat4 worldToCamera = glm::inverse(pathTracer->m_cameraTransform.getTransform());
            glm::vec4 hitPointCam = worldToCamera * glm::vec4(cameraHitPointWorld, 1.0f);
            // Pixel
            float Px = hitPointCam.x * 1000.0f;
            float Py = hitPointCam.y * 1000.0f;
            float Pz = hitPointCam.z * 1000.0f;
            float xSensor_mm = 10.0f * (Px / Pz);
            float ySensor_mm = 10.0f * (Py / Pz);
            float xPitchSize = ((10.0f) / 600.0f);
            float yPitchSize = ((10.0f) / 600.0f);
            float xPixel = (xSensor_mm / xPitchSize) + 300.0f;
            float yPixel = (ySensor_mm / yPitchSize) + 300.0f;
            int px = static_cast<int>(std::round(xPixel));
            int py = static_cast<int>(std::round(yPixel));
            if (px < 0 && px > static_cast<int>(pathTracer->m_camera.parameters().width) &&
                py < 0 && py > static_cast<int>(pathTracer->m_camera.parameters().height))
                continue;
        }
        */
        // contribution

        // Return them in the same order as forward inputs
        return {
            torch::Tensor(), // wrt settings (not a Tensor)
            torch::Tensor(), // wrt pathTracer (not a Tensor)
            grad_positions, // wrt positions
            grad_scales, // wrt scales
            grad_normals, // wrt normals
            torch::Tensor(), // emission
            torch::Tensor(), // colors
            torch::Tensor(), // specular
            torch::Tensor(), // diffuse
        };
    }
}
