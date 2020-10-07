#version 460

vec3 positions[6] =
    vec3[](vec3(-1.0, -1.0, 0.0), vec3(-1.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0),
           vec3(-1.0, -1.0, 0.0), vec3(1.0, 1.0, 0.0), vec3(1.0, -1.0, 0.0));

layout(location = 0) out vec3 direction;

layout(set = 0, binding = 0) uniform CameraUniformBufferObject {
  mat4 view;
  mat4 proj;
  mat4 rotation;
  vec4 eye;
  float fovy;
  float aspect;
};

out gl_PerVertex { vec4 gl_Position; };

void main() {
  float yTan = tan(fovy / 2.0);
  float xTan = yTan * aspect;
  direction = (rotation * vec4(xTan * positions[gl_VertexIndex].x,
                               yTan * positions[gl_VertexIndex].y, -1.0, 0.0))
                  .xyz;
  direction.xy *= -1;
  gl_Position = vec4(positions[gl_VertexIndex], 1.0);
}
