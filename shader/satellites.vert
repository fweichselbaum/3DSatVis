#version 450

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
in vec4 p3d_Color;

uniform float point_size;

out vec4 v_color;
flat out uint vertex_id;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    gl_PointSize = max(2.0, point_size / gl_Position.w);
    v_color = p3d_Color;
    vertex_id = gl_VertexID;
}