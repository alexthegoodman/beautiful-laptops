use beautiful_laptops::inference::infer_from_file;
// use beautiful_laptops::model::LaptopClassifier;
use beautiful_laptops::training::{train, TrainingConfig};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::prelude::*;

use burn::{
    backend::Autodiff,
    optim::{momentum::MomentumConfig, SgdConfig},
};

pub fn run() {
    train::<Autodiff<Wgpu>>(
        TrainingConfig::new(SgdConfig::new().with_momentum(Some(MomentumConfig {
            momentum: 0.9,
            dampening: 0.,
            nesterov: false,
        }))),
        WgpuDevice::DiscreteGpu(0),
    );
}

fn main() {
    // println!("Intializing...");
    // run();
    println!("Infering...");
    let device = WgpuDevice::DiscreteGpu(0);
    infer_from_file::<Wgpu>(
        "/tmp/custom-image-dataset",
        &device,
        "laptop_dataset_224/val/laptop_1530.jpg",
    );
}
