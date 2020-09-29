#version 450

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SceneParameter {
  mat4 view;
  mat4 proj;
  vec4 lightDirection;
  vec4 eyePosition;
};

layout(set = 0, binding = 2) uniform MaterialParameter {
  vec4 diffuse;
  vec4 ambient;
  vec4 specular;
  uint useTexture;
};

layout(set = 0, binding = 3) uniform sampler2D diffuseTex;

void main() {
  vec3 color = diffuse.xyz;
  if (useTexture != 0) {
    color *= texture(diffuseTex, inUV.xy).xyz;
  }
  outColor = vec4(color * 0.1, 1);
}
