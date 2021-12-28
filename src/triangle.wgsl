
[[block]]
struct Camera {
  view_pos: vec4<f32>;
  view_proj: mat4x4<f32>;
};

[[block]]
struct Light {
  position: vec3<f32>;
  color: vec3<f32>;
};

[[group(1), binding(0)]] var<uniform> camera: Camera;
[[group(2), binding(0)]] var<uniform> light: Light;

struct VertexInput {
  [[location(0)]] position: vec3<f32>;
  [[location(1)]] uvs: vec2<f32>;
  [[location(2)]] normal: vec3<f32>;
  [[location(3)]] tangent: vec3<f32>;
  [[location(4)]] bitangent: vec3<f32>;
};

struct InstanceInput {
  [[location(5)]] model_matrix_0: vec4<f32>;
  [[location(6)]] model_matrix_1: vec4<f32>;
  [[location(7)]] model_matrix_2: vec4<f32>;
  [[location(8)]] model_matrix_3: vec4<f32>;

  [[location(9)]] normal_matrix_0: vec3<f32>;
  [[location(10)]] normal_matrix_1: vec3<f32>;
  [[location(11)]] normal_matrix_2: vec3<f32>;
};

struct VertexOutput {
  [[builtin(position)]] frag_pos: vec4<f32>;
  [[location(0)]] uvs: vec2<f32>;
  [[location(1)]] tangent_position: vec3<f32>;
  [[location(2)]] tangent_light_position: vec3<f32>;
  [[location(3)]] tangent_view_position: vec3<f32>;
};

[[stage(vertex)]]
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
  let model_matrix = mat4x4<f32>(
    instance.model_matrix_0,
    instance.model_matrix_1,
    instance.model_matrix_2,
    instance.model_matrix_3,
  );
  let normal_matrix = mat3x3<f32>(
    instance.normal_matrix_0,
    instance.normal_matrix_1,
    instance.normal_matrix_2,
  );

  let world_normal = normalize(normal_matrix * vertex.normal);
  let world_tangent = normalize(normal_matrix * vertex.tangent);
  let world_bitangent = normalize(normal_matrix * vertex.bitangent);
  let tangent_matrix = transpose(mat3x3<f32>(
    world_bitangent,
    world_tangent,
    world_normal,
  ));

  let world_pos = model_matrix * vec4<f32>(vertex.position, 1.0);
  let clip_pos = camera.view_proj * world_pos;

  var out: VertexOutput;
  out.frag_pos = clip_pos;
  out.uvs = vertex.uvs;
  out.tangent_position = tangent_matrix * world_pos.xyz;
  out.tangent_view_position = tangent_matrix * camera.view_pos.xyz;
  out.tangent_light_position = tangent_matrix * light.position;
  return out;
}

[[group(0), binding(0)]] var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]] var s_diffuse: sampler;
[[group(0), binding(2)]] var t_normal: texture_2d<f32>;
[[group(0), binding(3)]] var s_normal: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
  let object = textureSample(t_diffuse, s_diffuse, in.uvs);
  let normal = textureSample(t_normal, s_normal, in.uvs);
  
  let tangent_normal = normal.xyz * 2.0 - 1.0;
  // vector from vertex to light
  let light_vec = normalize(in.tangent_light_position - in.tangent_position);
  // vector from vertex to camera
  let view_vec = normalize(in.tangent_view_position - in.tangent_position);
  // vector "half-way" between the light and view vectors
  let half_vec = normalize(view_vec + light_vec);

  // ambient component
  // - how much the scene is lit as a whole
  let ambient = light.color * 0.1;

  // diffuse component
  // - the light reflected by the object
  // - the closer `light_vec` is to `normal`, the more light it reflects from the light source
  let diffuse = light.color * max(dot(tangent_normal, light_vec), 0.0);

  // specular component
  // - highlights on shiny objects
  // - the closer `half_vec` gets to `normal`, the stronger the highlight gets
  let specular = light.color * pow(max(dot(tangent_normal, half_vec), 0.0), 32.0);
  
  let result = (ambient + diffuse + specular) * object.rgb;
  return vec4<f32>(result, object.a);
}