#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform PushConstants {
  mat4 modelMatrix;
  mat4 viewMatrix;
  mat4 projMatrix;
}
pushConstants;

layout(location = 0) out vec4 fragPosition;
layout(location = 1) out vec3 fragNormal;

out gl_PerVertex { vec4 gl_Position; };

void main() {
  vec4 pos = pushConstants.modelMatrix * vec4(inPosition, 1.0);
  fragPosition = pos;
  fragNormal = transpose(inverse(mat3(pushConstants.modelMatrix))) * inNormal;
  gl_Position = pushConstants.projMatrix * pushConstants.viewMatrix * pos;
}
