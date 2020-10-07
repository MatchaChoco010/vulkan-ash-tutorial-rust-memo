#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outWorldNormal;

layout(set = 0, binding = 0) uniform CameraUniformBufferObject {
  mat4 view;
  mat4 proj;
  mat4 rotation;
  vec4 eye;
  float fovy;
  float aspect;
};

layout(push_constant) uniform PushConstants { mat4 modelMatrix; };

out gl_PerVertex { vec4 gl_Position; };

void main() {
  gl_Position = proj * view * modelMatrix * vec4(inPosition, 1.0);
  outWorldPosition = (modelMatrix * vec4(inPosition, 1.0)).xyz;
  outWorldNormal = transpose(inverse(mat3(modelMatrix))) * inNormal;
}
