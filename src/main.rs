// use beautiful_laptops::model::LaptopClassifier;
use beautiful_laptops::training::{train, TrainingConfig};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::prelude::*;
// use burn::data::dataloader::DataLoaderBuilder;
use burn::record::Recorder;
use burn::tensor::backend::AutodiffBackend;

// use burn::loss::BCELoss;
use burn::module::Module;
use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
    Linear, LinearConfig, Relu,
};
use burn::optim::{Adam, AdamConfig, Optimizer};
use burn::tensor::Tensor;

// // Example usage:
// fn main() -> Result<(), CompileError> {
//     // Initialize your backend
//     // type Backend = burn::backend::Autodiff<burn::backend::Cuda>;
//     type Backend = Wgpu;

//     let device = WgpuDevice::BestAvailable;

//     // Create model
//     let model = LaptopClassifier::<Backend>::new(&device);

//     // Create config
//     let config = TrainingConfig::default();

//     // Assuming you have your datasets ready
//     // train_data and val_data should implement Dataset trait
//     let trained_model = train(model, train_data, val_data, config, device)?;

//     Ok(())
// }

use burn::{
    backend::Autodiff,
    optim::{momentum::MomentumConfig, SgdConfig},
};
// use custom_image_dataset::training::{train, TrainingConfig};

pub fn run() {
    train::<Autodiff<Wgpu>>(
        TrainingConfig::new(SgdConfig::new().with_momentum(Some(MomentumConfig {
            momentum: 0.9,
            dampening: 0.,
            nesterov: false,
        }))),
        WgpuDevice::default(),
    );
}

fn main() {
    // wgpu::run();
    run();
}
