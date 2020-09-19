#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;

layout(binding = 0, set = 0) uniform UniformBufferObject {
  mat4 world;
  mat4 view;
  mat4 proj;
}
ubo;

void main() {
  mat4 mvp = ubo.proj * ubo.view * ubo.world;
  gl_Position = mvp * vec4(inPos, 1.0);
  outColor = vec4(inColor, 1.0);
  outUV = inUV;
}
