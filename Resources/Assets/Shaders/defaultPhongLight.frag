#version 450

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec4 fragPos;
layout (location = 2) in vec3 inNormal;

layout (binding = 0) uniform CameraUBO {
    mat4 projection;
    mat4 view;
    vec3 position; // renamed to match fragment shader
} camera;

layout (set = 1, binding = 0) uniform Info {
    vec4 baseColor;
    float specular;
    float diffuse;
    vec2 _pad0;        // Ensure 16-byte alignment
    vec4 emissiveFactor;
    float numLightSources;
    vec4 lightPosition[32]; // Expanded vec3 -> vec4 for alignment
    vec4 lightNormal[32];   // Expanded vec3 -> vec4 for alignment
} info;

layout (set = 1, binding = 1) uniform sampler2D samplerColorMap;

layout (location = 0) out vec4 outColor;

// Calculates diffuse and specular contributions for one light (ambient is added separately)
vec3 calculatePhongLighting(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 specularColor, float shininess) {
    // Diffuse term
    float diff = max(dot(-normal, lightDir), 0.0);
    vec3 diffuseComponent = diff * vec3(1.0); // white light assumed

    // Specular term
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specularComponent = spec * specularColor;

    return diffuseComponent + specularComponent;
}

void main()
{
    // Normalize the input normal
    vec3 norm = normalize(inNormal);

    // World-space fragment position
    vec3 fragPosWorld = fragPos.xyz;

    // Compute view direction from fragment to camera
    vec3 viewDir = normalize(camera.position - fragPosWorld);

    // Determine specular color by mixing white and the base color
    vec3 specularColor = mix(vec3(1.0), info.baseColor.rgb, info.specular);

    // Compute shininess: higher diffuse weight gives lower shininess
    float shininess = mix(256.0, 32.0, info.diffuse);

    // Define ambient lighting (applied once)
    vec3 ambient = vec3(0.0);

    vec3 lighting = ambient;
    vec3 lightDir = normalize(info.lightPosition[0].rgb - fragPosWorld);
    lighting += calculatePhongLighting(norm, viewDir, lightDir, specularColor, shininess);

    // Sample texture color
    vec3 texColor = texture(samplerColorMap, inUV).rgb;

    // Combine the material color and lighting. The mix here is based on the specular factor.
    vec3 finalColor = mix(texColor * info.baseColor.rgb, lighting * texColor, info.specular);

    // Add emissive component
    finalColor += info.emissiveFactor.rgb;

    outColor = vec4(finalColor, 1.0);
}
