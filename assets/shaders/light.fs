#version 330 core
out vec4 FragColor;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    float specular;
    vec3 emission;
    sampler2D diffuse_map;
    sampler2D specular_map;
    sampler2D emission_map;
    float shininess;
};

uniform Material material;

void main() {
    vec3 result = material.diffuse;
    // result = vec3(1,1,1);
    // vec3 result = (ambient) * objectColor;
    FragColor = vec4(result, 1.0);
}