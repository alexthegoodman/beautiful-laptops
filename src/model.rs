use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::prelude::*;
// use burn::data::dataloader::DataLoaderBuilder;
use burn::record::Recorder;
use burn::tensor::activation::sigmoid;
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
use nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig};

// #[derive(Module, Debug)]
// pub struct LaptopClassifier<B: Backend> {
//     conv1: Conv2d<B>,
//     conv2: Conv2d<B>,
//     pool: MaxPool2d,
//     fc1: Linear<B>,
//     fc2: Linear<B>,
//     relu: Relu,
// }

// impl<B: Backend> LaptopClassifier<B> {
//     pub fn new(device: &WgpuDevice) -> Self {
//         let conv1 = Conv2dConfig::new([3, 16], [3, 3])
//             .with_padding(nn::PaddingConfig2d::Explicit(1, 1))
//             .init(device);

//         let conv2 = Conv2dConfig::new([16, 32], [3, 3])
//             .with_padding(nn::PaddingConfig2d::Explicit(1, 1))
//             .init(device);

//         let pool = MaxPool2dConfig::new([2, 2]).init();

//         let fc1 = LinearConfig::new(32 * 56 * 56, 120).init(device);
//         let fc2 = LinearConfig::new(120, 1).init(device);

//         Self {
//             conv1,
//             conv2,
//             pool,
//             fc1,
//             fc2,
//             relu: Relu::new(),
//         }
//     }

//     pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
//         let x = self.conv1.forward(input);
//         let x = self.relu.forward(x);
//         let x = self.pool.forward(x);

//         let x = self.conv2.forward(x);
//         let x = self.relu.forward(x);
//         let x = self.pool.forward(x);

//         let batch_size = x.shape()[0];
//         let x = x.reshape([batch_size, 32 * 56 * 56]);

//         let x = self.fc1.forward(x);
//         let x = self.relu.forward(x);
//         let x = self.fc2.forward(x);

//         // x.sigmoid()
//         sigmoid(x)
//     }
// }

// #[derive(Debug)]
// pub struct TrainingConfig {
//     pub learning_rate: f32,
//     pub batch_size: usize,
//     pub epochs: usize,
//     pub beta1: f32,
//     pub beta2: f32,
//     pub epsilon: f32,
// }

// impl Default for TrainingConfig {
//     fn default() -> Self {
//         Self {
//             learning_rate: 0.001,
//             batch_size: 32,
//             epochs: 10,
//             beta1: 0.9,
//             beta2: 0.999,
//             epsilon: 1e-8,
//         }
//     }
// }

// impl<B: Backend> LaptopClassifier<B> {
//     pub fn training_step(
//         &self,
//         batch: (Tensor<B, 4>, Tensor<B, 2>),
//         device: &WgpuDevice,
//     ) -> Tensor<B, 0> {
//         let (images, targets) = batch;
//         let predictions = self.forward(images);
//         // BinaryCrossEntropyLoss::new().forward(predictions, targets)
//         let test = BinaryCrossEntropyLossConfig::new().init(device);
//         test.forward(predictions, targets)
//     }
// }

// pub fn create_optimizer_config<B: Backend>(
//     config: &TrainingConfig,
//     model: LaptopClassifier<B>,
//     // ) -> Adam<LaptopClassifier<B>> {
// ) -> AdamConfig {
//     AdamConfig::new()
//         // .with_learning_rate(config.learning_rate)
//         // .with_betas([config.beta1, config.beta2])
//         .with_beta_1(config.beta1)
//         .with_beta_2(config.beta2)
//         .with_epsilon(config.epsilon)
// }

use burn::{
    nn::{Dropout, DropoutConfig, PaddingConfig2d},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct LaptopClassifier<B: Backend> {
    relu: Relu,
    dropout: Dropout,
    pool: MaxPool2d,
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    // conv3: Conv2d<B>,
    // conv4: Conv2d<B>,
    // conv5: Conv2d<B>,
    // conv6: Conv2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> LaptopClassifier<B> {
    // pub fn new(num_classes: usize, device: &Device<B>) -> Self {
    //     let conv1 = Conv2dConfig::new([3, 32], [3, 3])
    //         .with_padding(PaddingConfig2d::Same)
    //         .init(device);
    //     let conv2 = Conv2dConfig::new([32, 32], [3, 3])
    //         .with_padding(PaddingConfig2d::Same)
    //         .init(device);

    //     // let conv3 = Conv2dConfig::new([32, 64], [3, 3])
    //     //     .with_padding(PaddingConfig2d::Same)
    //     //     .init(device);
    //     // let conv4 = Conv2dConfig::new([64, 64], [3, 3])
    //     //     .with_padding(PaddingConfig2d::Same)
    //     //     .init(device);

    //     // let conv5 = Conv2dConfig::new([64, 128], [3, 3])
    //     //     .with_padding(PaddingConfig2d::Same)
    //     //     .init(device);
    //     // let conv6 = Conv2dConfig::new([128, 128], [3, 3])
    //     //     .with_padding(PaddingConfig2d::Same)
    //     //     .init(device);

    //     let pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

    //     // TODO: adjust input and output sizes?

    //     let fc1 = LinearConfig::new(2048, 128).init(device);
    //     let fc2 = LinearConfig::new(128, num_classes).init(device);

    //     let dropout = DropoutConfig::new(0.3).init();

    //     Self {
    //         relu: Relu::new(),
    //         dropout,
    //         pool,
    //         conv1,
    //         conv2,
    //         // conv3,
    //         // conv4,
    //         // conv5,
    //         // conv6,
    //         fc1,
    //         fc2,
    //     }
    // }

    pub fn new(num_classes: usize, device: &Device<B>) -> Self {
        let conv1 = Conv2dConfig::new([3, 8], [3, 3])
            .with_stride([8, 8])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let conv2 = Conv2dConfig::new([8, 16], [3, 3])
            .with_stride([8, 8])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        // // Conv3: 200x200x32 -> 200x200x64
        // let conv3 = Conv2dConfig::new([32, 64], [3, 3])
        //     .with_padding(PaddingConfig2d::Same)
        //     .init(device);
        // // After Pool: 100x100x64

        // // Conv4: 100x100x64 -> 100x100x64
        // let conv4 = Conv2dConfig::new([64, 64], [3, 3])
        //     .with_padding(PaddingConfig2d::Same)
        //     .init(device);
        // // After Pool: 50x50x64

        // // Conv5: 50x50x64 -> 50x50x128
        // let conv5 = Conv2dConfig::new([64, 128], [3, 3])
        //     .with_padding(PaddingConfig2d::Same)
        //     .init(device);
        // // After Pool: 25x25x128

        // // Conv6: 25x25x128 -> 25x25x128
        // let conv6 = Conv2dConfig::new([128, 128], [3, 3])
        //     .with_padding(PaddingConfig2d::Same)
        //     .init(device);
        // // After Pool: 13x13x128 = 21,632 features

        let pool = MaxPool2dConfig::new([8, 8]).with_strides([8, 8]).init();

        let fc1 = LinearConfig::new(144, 128).init(device);
        let fc2 = LinearConfig::new(128, num_classes).init(device);

        let dropout = DropoutConfig::new(0.3).init();

        Self {
            relu: Relu::new(),
            dropout,
            pool,
            conv1,
            conv2,
            // conv3,
            // conv4,
            // conv5,
            // conv6,
            fc1,
            fc2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.relu.forward(x);
        let x = self.conv2.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);
        let x = self.dropout.forward(x);

        // let x = self.conv3.forward(x);
        // let x = self.relu.forward(x);
        // let x = self.conv4.forward(x);
        // let x = self.relu.forward(x);
        // let x = self.pool.forward(x);
        // let x = self.dropout.forward(x);

        // let x = self.conv5.forward(x);
        // let x = self.relu.forward(x);
        // let x = self.conv6.forward(x);
        // let x = self.relu.forward(x);
        // let x = self.pool.forward(x);
        // let x = self.dropout.forward(x);

        // extra pooling for memory management
        let x = self.pool.forward(x);

        let x = x.flatten(1, 3);

        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);

        self.fc2.forward(x)
    }
}
