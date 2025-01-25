#version 450

layout (location = 0) in vec2 inUV;
layout (binding = 0) uniform sampler2D samplerColorMap;
layout (location = 0) out vec4 outColor;

layout (binding = 1) uniform INFO {
    int selection;
    float gammaCorrection;
} info;


void main() {
    //vec3 value = texture(samplerColorMap, vec2(1.0 - inUV.x, 1.0 - inUV.y)).rgb;
    vec3 value = texture(samplerColorMap, inUV).rgb;

    // Apply gamma correction (assuming gamma = 2.2)
    //vec3 correctedValue = pow(value, vec3(1.0 / info.gammaCorrection)); // Correct each channel

    outColor = vec4(value, 1.0f);
}