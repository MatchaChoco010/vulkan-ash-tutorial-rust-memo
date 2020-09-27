#version 450

layout(location = 0) in vec2 inTexcoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D tex;
layout(set = 0, binding = 1) uniform UniformBufferObject {
  vec2 windowSize;
  float blockSize;
}
ubo;

void main() {
  vec2 uv = inTexcoord * ubo.windowSize;
  uv /= ubo.blockSize;
  uv = floor(uv) * ubo.blockSize;
  uv /= ubo.windowSize;
  outColor = texture(tex, uv);
}
