use bytemuck::{Pod, Zeroable};
use cgmath::{InnerSpace, Matrix4, Point3, Rad, Vector3};
use std::{f32::consts::FRAC_PI_2, time::Duration};
use winit::event::{ElementState, VirtualKeyCode};

pub struct Camera {
  pub position: Point3<f32>,
  yaw: Rad<f32>,
  pitch: Rad<f32>,

  aspect: f32,
  fovy: Rad<f32>,
  near: f32,
  far: f32,
}

impl Camera {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    position: impl Into<Point3<f32>>,
    yaw: impl Into<Rad<f32>>,
    pitch: impl Into<Rad<f32>>,
    width: u32,
    height: u32,
    fovy: impl Into<Rad<f32>>,
    near: f32,
    far: f32,
  ) -> Self {
    Self {
      position: position.into(),
      yaw: yaw.into(),
      pitch: pitch.into(),
      aspect: width as f32 / height as f32,
      fovy: fovy.into(),
      near,
      far,
    }
  }

  pub fn view(&self) -> Matrix4<f32> {
    Matrix4::look_to_rh(
      self.position,
      Vector3::new(self.yaw.0.cos(), self.pitch.0.sin(), self.yaw.0.sin()).normalize(),
      Vector3::unit_y(),
    )
  }

  pub fn resize(&mut self, width: u32, height: u32) {
    self.aspect = width as f32 / height as f32;
  }

  pub fn projection(&self) -> Matrix4<f32> {
    OPENGL_TO_WGPU_MATRIX * cgmath::perspective(self.fovy, self.aspect, self.near, self.far)
  }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
  1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 0.5, 0.0,
  0.0, 0.0, 0.5, 1.0,
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
  view_pos: [f32; 4],
  view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
  pub fn new() -> Self {
    use cgmath::SquareMatrix;
    Self {
      view_pos: [0.0; 4],
      view_proj: cgmath::Matrix4::identity().into(),
    }
  }

  pub fn update_view_proj(&mut self, camera: &Camera) {
    self.view_pos = camera.position.to_homogeneous().into();
    self.view_proj = (camera.projection() * camera.view()).into();
  }
}

pub struct Controller {
  amount_left: f32,
  amount_right: f32,
  amount_forward: f32,
  amount_backward: f32,
  amount_up: f32,
  amount_down: f32,
  rotate_horizontal: f32,
  rotate_vertical: f32,
  speed: f32,
  fast: bool,
  sensitivity: f32,
}

impl Controller {
  pub fn new(speed: f32, sensitivity: f32) -> Self {
    Self {
      amount_left: 0.0,
      amount_right: 0.0,
      amount_forward: 0.0,
      amount_backward: 0.0,
      amount_up: 0.0,
      amount_down: 0.0,
      rotate_horizontal: 0.0,
      rotate_vertical: 0.0,
      speed,
      fast: false,
      sensitivity,
    }
  }

  pub fn process_keyboard(&mut self, key: VirtualKeyCode, state: ElementState) -> bool {
    let amount = if state == ElementState::Pressed {
      1.0
    } else {
      0.0
    };
    match key {
      VirtualKeyCode::W | VirtualKeyCode::Up => {
        self.amount_forward = amount;
        true
      }
      VirtualKeyCode::S | VirtualKeyCode::Down => {
        self.amount_backward = amount;
        true
      }
      VirtualKeyCode::A | VirtualKeyCode::Left => {
        self.amount_left = amount;
        true
      }
      VirtualKeyCode::D | VirtualKeyCode::Right => {
        self.amount_right = amount;
        true
      }
      VirtualKeyCode::Space => {
        self.amount_up = amount;
        true
      }
      VirtualKeyCode::LControl => {
        self.amount_down = amount;
        true
      }
      VirtualKeyCode::LShift => {
        self.fast = state == ElementState::Pressed;
        true
      }
      _ => false,
    }
  }

  pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
    self.rotate_horizontal = mouse_dx as f32;
    self.rotate_vertical = mouse_dy as f32;
  }

  pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
    let dt = dt.as_secs_f32();

    let speed = if self.fast {
      self.speed * 2.0
    } else {
      self.speed
    };

    // Move forward/backward and left/right
    let (yaw_sin, yaw_cos) = camera.yaw.0.sin_cos();
    let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
    let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
    camera.position += forward * (self.amount_forward - self.amount_backward) * speed * dt;
    camera.position += right * (self.amount_right - self.amount_left) * speed * dt;

    // Move up/down. Since we don't use roll, we can just
    // modify the y coordinate directly.
    camera.position.y += (self.amount_up - self.amount_down) * speed * dt;

    // Rotate
    camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
    camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

    // If process_mouse isn't called every frame, these values
    // will not get set to zero, and the camera will rotate
    // when moving in a non cardinal direction.
    self.rotate_horizontal = 0.0;
    self.rotate_vertical = 0.0;

    // Keep the camera's angle from going too high/low.
    if camera.pitch < -Rad(SAFE_FRAC_PI_2) {
      camera.pitch = -Rad(SAFE_FRAC_PI_2);
    } else if camera.pitch > Rad(SAFE_FRAC_PI_2) {
      camera.pitch = Rad(SAFE_FRAC_PI_2);
    }
  }
}
