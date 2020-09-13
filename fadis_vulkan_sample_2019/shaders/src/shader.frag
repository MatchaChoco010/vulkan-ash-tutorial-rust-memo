#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 fragPosition;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
  mat4 worldMatrix;
  mat4 viewMatrix;
  mat4 projMatrix;
  vec3 eye;
  vec3 lightPos;
}
pushConstants;

void main() {
  vec3 pos = fragPosition.xyz / fragPosition.w;
  vec3 N = normalize(fragNormal);
  vec3 V = normalize(pushConstants.eye - pos);
  vec3 L = normalize(pushConstants.lightPos - pos);
  vec3 H = normalize(L + V);

  vec3 lambert = vec3(1, 1, 1) * max(dot(N, L), 0);
  vec3 blinnPhong = vec3(1, 1, 1) * pow(max(dot(N, H), 0), 50);

  outColor = vec4(pow(lambert + blinnPhong, vec3(2.2)), 1);
}
