#version 450

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec4 fragPos;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec4 vertexColor; // World space position

layout (binding = 0) uniform CameraUBO
{
    mat4 projection;
    mat4 view;
    vec3 position;
} camera;

layout (set = 1, binding = 0) uniform Info {
    vec4 baseColor;      // Base color of the material
    float metallic;      // Metallic factor (0.0 to 1.0)
    float roughness;     // Roughness factor (0.0 to 1.0)
    float isDisparity;   // Unused in Phong, potentially for stereo
    vec4 emissiveFactor; // Emissive color
} info;

layout (set = 1, binding = 1) uniform sampler2D samplerColorMap;

layout (location = 0) out vec4 outColor;

// Phong lighting calculation function
vec3 calculatePhongLighting(vec3 normal, vec3 fragPos, vec3 viewDir, vec3 lightDir, vec3 specularColor, float shininess) {
    // Phong lighting components
    vec3 ambientColor = vec3(0.2, 0.2, 0.2); // Ambient color
    vec3 diffuseColor = vec3(1.0, 1.0, 1.0); // Diffuse color

    // Ambient
    vec3 ambient = ambientColor;

    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;

    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * specularColor;

    // Combine results
    return (ambient + diffuse + specular);
}

void main()
{
    // Define light direction (assuming light.direction points towards the scene)
    vec3 lightDir = normalize(vec3(0, 0, 1)); // Adjust based on your scene setup

    // Normalize normal vector
    vec3 norm = normalize(inNormal);

    // World space fragment position
    vec3 fragPosWorld = fragPos.xyz;

    // Compute view direction
    vec3 viewDir = normalize(camera.position - fragPosWorld);

    // Determine specular color based on metallic factor
    vec3 specularColor = mix(vec3(1.0), info.baseColor.rgb, info.metallic);

    // Adjust shininess based on roughness
    float shininess = mix(256.0, 32.0, info.roughness); // Higher roughness -> lower shininess

    // Calculate Phong lighting
    vec3 phongLighting = calculatePhongLighting(norm, fragPosWorld, viewDir, lightDir, specularColor, shininess);

    // Sample texture color
    vec3 texColor = texture(samplerColorMap, inUV).rgb;

    // Combine lighting with texture and base color
    vec3 finalColor = mix(texColor * info.baseColor.rgb, phongLighting * texColor, info.metallic);

    // Apply emissive factor
    finalColor += info.emissiveFactor.rgb;

    // Assign to output color
    outColor = vec4(finalColor, 1.0);
    //vec3 normalColor = normalize(inNormal) * 0.5 + 0.5; // Map from [-1,1] to [0,1]
    //outColor = vec4(normalColor, 1.0); // Output as color
    // Uncomment the following lines for depth debugging (ensure it's commented out in the final shader)
    // float depth = (camera.projection * camera.view * fragPos).z / (camera.projection * camera.view * fragPos).w;
    // outColor = vec4(vec3(depth), 1.0);
}
