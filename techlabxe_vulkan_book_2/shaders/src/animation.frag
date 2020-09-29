#version 450

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec4 inWorldPosition;
layout(location = 4) in vec4 inShadowPosition;
layout(location = 5) in vec4 inShadowPosUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SceneParameter {
  mat4 view;
  mat4 proj;
  vec4 lightDirection;
  vec4 eyePosition;
  mat4 lightViewProj;
  mat4 lightViewProjBias;
  vec2 resolution;
};

layout(set = 0, binding = 2) uniform MaterialParameter {
  vec4 diffuse;
  vec4 ambient;
  vec4 specular;
  uint useTexture;
};

layout(set = 0, binding = 3) uniform sampler2D diffuseTex;

layout(set = 0, binding = 4) uniform sampler2D shadowTex;

void main() {
  vec4 color = diffuse;
  vec3 normal = normalize(inNormal);
  vec3 toLightDirection = normalize(lightDirection.xyz);
  float lmb = clamp(dot(toLightDirection, normalize(inNormal)), 0, 1);
  float isShade = step(lmb, 0.5);

  float z = inShadowPosition.z / inShadowPosition.w;
  vec4 fetchUV = inShadowPosUV / inShadowPosUV.w;
  float depthFromLight = texture(shadowTex, fetchUV.xy).r + 0.0002;
  float isShadow = float(depthFromLight < z);

  vec2 st =
      (gl_FragCoord.xy * 2.0 - resolution) / min(resolution.x, resolution.y);
  vec2 uv = mod(st * 80, 1.0) * 2.0 - 1.0;
  float c = ceil(0.6 - length(uv));

  if (useTexture != 0) {
    color *= texture(diffuseTex, inUV.xy);
  }
  vec3 baseColor = color.xyz;
  color.rgb = mix(baseColor, baseColor * 0.6 + vec3(0.0, 0.1, 0.15),
                  min(c, max(isShade, isShadow)));

  outColor = color;
}
