mod camera;
mod light;
mod model;
mod texture;

use std::time::{Duration, Instant};

use anyhow::Result;
use cgmath::{prelude::*, Matrix3, Matrix4, Quaternion};
use wgpu::util::DeviceExt;
use winit::{
  event::*,
  event_loop::{ControlFlow, EventLoop},
  window::{Window, WindowBuilder},
};

use camera::{Camera, CameraUniform};
use light::LightUniform;
use model::{Model, ModelVertex, Vertex};
use texture::Texture;

// continue:
// https://sotrh.github.io/learn-wgpu/beginner/tutorial7-instancing/#the-instance-buffer

struct Instance {
  position: cgmath::Vector3<f32>,
  rotation: cgmath::Quaternion<f32>,
}

impl Instance {
  pub fn data(&self) -> InstanceData {
    let translation = Matrix4::from_translation(self.position);
    let rotation = Matrix4::from(self.rotation);
    InstanceData {
      model: (translation * rotation).into(),
      normal: Matrix3::from(self.rotation).into(),
    }
  }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
  model: [[f32; 4]; 4],
  normal: [[f32; 3]; 3],
}

impl Vertex for InstanceData {
  fn descriptor<'a>() -> wgpu::VertexBufferLayout<'a> {
    use std::mem;
    wgpu::VertexBufferLayout {
      array_stride: mem::size_of::<InstanceData>() as wgpu::BufferAddress,
      step_mode: wgpu::VertexStepMode::Instance,
      attributes: &[
        // model
        wgpu::VertexAttribute {
          offset: 0,
          shader_location: 5,
          format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
          shader_location: 6,
          format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
          shader_location: 7,
          format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
          shader_location: 8,
          format: wgpu::VertexFormat::Float32x4,
        },
        // normal
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
          shader_location: 9,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
          shader_location: 10,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
          shader_location: 11,
          format: wgpu::VertexFormat::Float32x3,
        },
      ],
    }
  }
}

const NUM_INSTANCES_PER_ROW: u32 = 10;
const ANGULAR_VELOCITY: cgmath::Rad<f32> = cgmath::Rad(0.0); //cgmath::Rad(std::f32::consts::PI / 144.0);
const SPACE_BETWEEN: f32 = 3.0;

struct State {
  surface: wgpu::Surface,
  device: wgpu::Device,
  queue: wgpu::Queue,
  config: wgpu::SurfaceConfiguration,
  size: winit::dpi::PhysicalSize<u32>,
  camera: Camera,
  camera_uniform: CameraUniform,
  camera_buffer: wgpu::Buffer,
  camera_controller: camera::Controller,
  camera_bind_group: wgpu::BindGroup,
  light_uniform: LightUniform,
  light_buffer: wgpu::Buffer,
  light_bind_group: wgpu::BindGroup,
  render_pipeline: wgpu::RenderPipeline,
  light_render_pipeline: wgpu::RenderPipeline,
  model: Model,
  depth_texture: Texture,
  instances: Vec<Instance>,
  instance_buffer: wgpu::Buffer,
  mouse_pressed: bool,
}

fn create_render_pipeline(
  label: &str,
  device: &wgpu::Device,
  layout: &wgpu::PipelineLayout,
  color_format: wgpu::TextureFormat,
  depth_format: Option<wgpu::TextureFormat>,
  vertex_layouts: &[wgpu::VertexBufferLayout],
  shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
  let shader = device.create_shader_module(&shader);

  device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    label: Some(label),
    layout: Some(layout),
    vertex: wgpu::VertexState {
      module: &shader,
      entry_point: "vs_main",
      buffers: vertex_layouts,
    },
    fragment: Some(wgpu::FragmentState {
      module: &shader,
      entry_point: "fs_main",
      targets: &[wgpu::ColorTargetState {
        format: color_format,
        blend: Some(wgpu::BlendState {
          alpha: wgpu::BlendComponent::REPLACE,
          color: wgpu::BlendComponent::REPLACE,
        }),
        write_mask: wgpu::ColorWrites::ALL,
      }],
    }),
    primitive: wgpu::PrimitiveState {
      topology: wgpu::PrimitiveTopology::TriangleList,
      strip_index_format: None,
      front_face: wgpu::FrontFace::Ccw,
      cull_mode: Some(wgpu::Face::Back),
      // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
      polygon_mode: wgpu::PolygonMode::Fill,
      // Requires Features::DEPTH_CLAMPING
      clamp_depth: false,
      // Requires Features::CONSERVATIVE_RASTERIZATION
      conservative: false,
    },
    depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
      format,
      depth_write_enabled: true,
      depth_compare: wgpu::CompareFunction::Less,
      stencil: wgpu::StencilState::default(),
      bias: wgpu::DepthBiasState::default(),
    }),
    multisample: wgpu::MultisampleState {
      count: 1,
      mask: !0,
      alpha_to_coverage_enabled: false,
    },
  })
}

impl State {
  async fn new(window: &Window) -> Result<Self> {
    let size = window.inner_size();

    // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(window) };
    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
      })
      .await
      .unwrap();

    let (device, queue) = adapter
      .request_device(
        &wgpu::DeviceDescriptor {
          features: wgpu::Features::empty(),
          limits: wgpu::Limits::default(),
          label: None,
        },
        None,
      )
      .await?;

    let config = wgpu::SurfaceConfiguration {
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
      format: surface.get_preferred_format(&adapter).unwrap(),
      width: size.width,
      height: size.height,
      present_mode: wgpu::PresentMode::Fifo,
    };
    surface.configure(&device, &config);

    let camera = Camera::new(
      (0.0, 5.0, 10.0),
      cgmath::Deg(-90.0),
      cgmath::Deg(-20.0),
      config.width,
      config.height,
      cgmath::Deg(45.0),
      0.1,
      100.0,
    );
    let camera_controller = camera::Controller::new(4.0, 0.4);

    let mut camera_uniform = CameraUniform::new();
    camera_uniform.update_view_proj(&camera);

    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Camera Buffer"),
      contents: bytemuck::cast_slice(&[camera_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let camera_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        }],
      });
    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("camera_bind_group"),
      layout: &camera_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: camera_buffer.as_entire_binding(),
      }],
    });

    let light_uniform = LightUniform::new([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]);
    let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Light"),
      contents: bytemuck::cast_slice(&[light_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let light_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        }],
      });
    let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &light_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: light_buffer.as_entire_binding(),
      }],
    });

    let texture_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("texture_bind_group_layout"),
        entries: &[
          // diffuse texture
          wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              multisampled: false,
              view_dimension: wgpu::TextureViewDimension::D2,
              sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler {
              comparison: false,
              filtering: true,
            },
            count: None,
          },
          // normal map
          wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              multisampled: false,
              sample_type: wgpu::TextureSampleType::Float { filterable: true },
              view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler {
              comparison: false,
              filtering: true,
            },
            count: None,
          },
        ],
      });

    let depth_texture = Texture::create_depth_texture("depth_texture", &device, &config);

    let render_pipeline = {
      let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Triangle Pipeline Layout"),
        bind_group_layouts: &[
          &texture_bind_group_layout,
          &camera_bind_group_layout,
          &light_bind_group_layout,
        ],
        push_constant_ranges: &[],
      });
      create_render_pipeline(
        "Render Pipeline",
        &device,
        &layout,
        config.format,
        Some(Texture::DEPTH_FORMAT),
        &[ModelVertex::descriptor(), InstanceData::descriptor()],
        wgpu::ShaderModuleDescriptor {
          label: Some("Triangle Shader"),
          source: wgpu::ShaderSource::Wgsl(include_str!("triangle.wgsl").into()),
        },
      )
    };
    let light_render_pipeline = {
      let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Light Pipeline Layout"),
        bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
        push_constant_ranges: &[],
      });
      create_render_pipeline(
        "Light Render Pipeline",
        &device,
        &layout,
        config.format,
        Some(Texture::DEPTH_FORMAT),
        &[ModelVertex::descriptor()],
        wgpu::ShaderModuleDescriptor {
          label: Some("Light Shader"),
          source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
        },
      )
    };

    let model = Model::load("res/cube.obj", &device, &queue, &texture_bind_group_layout)?;

    let instances = (0..NUM_INSTANCES_PER_ROW)
      .flat_map(|z| {
        (0..NUM_INSTANCES_PER_ROW).map(move |x| {
          let position = cgmath::Vector3 {
            x: SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0),
            y: 0.0,
            z: SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0),
          };
          Instance {
            position,
            rotation: if position.is_zero() {
              cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
            } else {
              cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
            },
          }
        })
      })
      .collect::<Vec<_>>();
    let instance_data = instances.iter().map(Instance::data).collect::<Vec<_>>();
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Instance Buffer"),
      contents: bytemuck::cast_slice(&instance_data),
      usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    Ok(Self {
      surface,
      device,
      queue,
      config,
      size,
      camera,
      camera_uniform,
      camera_buffer,
      camera_bind_group,
      camera_controller,
      light_uniform,
      light_buffer,
      light_bind_group,
      render_pipeline,
      light_render_pipeline,
      model,
      depth_texture,
      instances,
      instance_buffer,
      mouse_pressed: false,
    })
  }

  fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    if new_size.width > 0 && new_size.height > 0 {
      self.size = new_size;
      self.config.width = new_size.width;
      self.config.height = new_size.height;
      self.surface.configure(&self.device, &self.config);

      self.depth_texture =
        Texture::create_depth_texture("depth_texture", &self.device, &self.config);

      self.camera.resize(new_size.width, new_size.height);
    }
  }

  fn input(&mut self, event: &DeviceEvent) -> bool {
    match event {
      DeviceEvent::Key(KeyboardInput {
        virtual_keycode: Some(key),
        state,
        ..
      }) => self.camera_controller.process_keyboard(*key, *state),
      DeviceEvent::Button {
        button: 1, // Left Mouse Button
        state,
      } => {
        self.mouse_pressed = *state == ElementState::Pressed;
        true
      }
      DeviceEvent::MouseMotion { delta } => {
        if self.mouse_pressed {
          self.camera_controller.process_mouse(delta.0, delta.1);
        }
        true
      }
      _ => false,
    }
  }

  fn update(&mut self, dt: Duration) {
    self.camera_controller.update_camera(&mut self.camera, dt);
    self.camera_uniform.update_view_proj(&self.camera);
    self.queue.write_buffer(
      &self.camera_buffer,
      0,
      bytemuck::cast_slice(&[self.camera_uniform]),
    );

    for instance in &mut self.instances {
      instance.rotation = cgmath::Quaternion::from_angle_y(ANGULAR_VELOCITY) * instance.rotation;
    }
    let instance_data = self
      .instances
      .iter()
      .map(Instance::data)
      .collect::<Vec<_>>();
    self.queue.write_buffer(
      &self.instance_buffer,
      0,
      bytemuck::cast_slice(&instance_data),
    );

    let rotation =
      Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(60.0 * dt.as_secs_f32()));
    self
      .light_uniform
      .set_position(rotation * self.light_uniform.position());
    self.queue.write_buffer(
      &self.light_buffer,
      0,
      bytemuck::cast_slice(&[self.light_uniform]),
    );
  }

  fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
    let output = self.surface.get_current_texture()?;

    let view = output
      .texture
      .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = self
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Render Encoder"),
      });

    {
      let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Triangle Pass"),
        color_attachments: &[wgpu::RenderPassColorAttachment {
          view: &view,
          resolve_target: None,
          ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
              r: 0.1,
              g: 0.2,
              b: 0.3,
              a: 1.0,
            }),
            store: true,
          },
        }],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
          view: &self.depth_texture.view,
          depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: true,
          }),
          stencil_ops: None,
        }),
      });

      render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

      use light::DrawLight;
      render_pass.set_pipeline(&self.light_render_pipeline);
      render_pass.draw_light_model(&self.model, &self.camera_bind_group, &self.light_bind_group);

      use model::DrawModel;
      render_pass.set_pipeline(&self.render_pipeline);
      render_pass.draw_model_instanced(
        &self.model,
        &self.camera_bind_group,
        &self.light_bind_group,
        0..self.instances.len() as u32,
      );
    }

    self.queue.submit(std::iter::once(encoder.finish()));
    output.present();

    Ok(())
  }
}

fn main() -> Result<()> {
  std::env::set_var(
    "RUST_LOG",
    std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
  );
  let event_loop = EventLoop::new();
  let window = WindowBuilder::new().build(&event_loop).unwrap();
  window.set_title("wgpu-book");

  let mut state = pollster::block_on(State::new(&window))?;
  let mut last_render_time = Instant::now();

  event_loop.run(move |event, _, control_flow| match event {
    Event::DeviceEvent { ref event, .. } => {
      state.input(event);
    }
    Event::WindowEvent {
      ref event,
      window_id,
    } if window_id == window.id() => match event {
      WindowEvent::CloseRequested
      | WindowEvent::KeyboardInput {
        input:
          KeyboardInput {
            state: ElementState::Pressed,
            virtual_keycode: Some(VirtualKeyCode::Escape),
            ..
          },
        ..
      } => *control_flow = ControlFlow::Exit,
      WindowEvent::Resized(physical_size) => {
        state.resize(*physical_size);
      }
      WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
        state.resize(**new_inner_size);
      }
      _ => {}
    },
    Event::RedrawRequested(_) => {
      let now = Instant::now();
      let dt = now - last_render_time;
      last_render_time = now;

      state.update(dt);
      match state.render() {
        Ok(_) => {}
        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
        Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
        Err(e) => eprintln!("{:?}", e),
      }
    }
    Event::MainEventsCleared => {
      window.request_redraw();
    }
    _ => {}
  });
}
