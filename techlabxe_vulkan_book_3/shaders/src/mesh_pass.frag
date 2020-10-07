#version 460

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inWorldNormal;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraUniformBufferObject {
  mat4 view;
  mat4 proj;
  mat4 rotation;
  vec4 eye;
  float fovy;
  float aspect;
};
layout(set = 0, binding = 1) uniform samplerCube cubemap;

void main() {
  vec3 V = normalize(inWorldPosition - eye.xyz / eye.w);
  vec3 N = normalize(inWorldNormal);
  vec3 direction = reflect(V, N);
  direction.xy *= -1;
  outColor = texture(cubemap, direction);
}
