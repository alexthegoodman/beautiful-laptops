use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::prelude::*;
// use burn::data::dataloader::DataLoaderBuilder;
use burn::record::Recorder;
use burn::serde::de::value::Error;
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
use burn::train::MultiLabelClassificationOutput;
use nn::loss::BinaryCrossEntropyLossConfig;

// use crate::model::{create_optimizer, LaptopClassifier, TrainingConfig};

// pub fn train<B: AutodiffBackend>(
//     model: LaptopClassifier<B>,
//     train_data: impl Dataset,
//     val_data: impl Dataset,
//     config: TrainingConfig,
//     device: WgpuDevice,
// ) -> Result<LaptopClassifier<B>, Error> {
//     // Initialize optimizer
//     let mut optimizer = create_optimizer(&config, model);

//     // Create data loaders
//     let train_loader = DataLoaderBuilder::new()
//         .batch_size(config.batch_size)
//         .shuffle(42)
//         .build(train_data);

//     let val_loader = DataLoaderBuilder::new()
//         .batch_size(config.batch_size)
//         .build(val_data);

//     // Training loop
//     for epoch in 0..config.epochs {
//         // Training phase
//         model.train();
//         let mut train_loss = 0.0;

//         for (batch_idx, batch) in train_loader.iter().enumerate() {
//             // Forward pass and compute loss
//             let loss = model.training_step(batch, &device);

//             // Backward pass
//             optimizer.backward_step(&loss);

//             train_loss += loss.to_scalar();

//             // if batch_idx % 10 == 0 {
//             //     println!(
//             //         "Epoch [{}/{}], Batch [{}/{}], Loss: {:.4}",
//             //         epoch + 1,
//             //         config.epochs,
//             //         batch_idx + 1,
//             //         train_loader.len(),
//             //         loss.to_scalar()
//             //     );
//             // }
//         }

//         // Validation phase
//         model.eval();
//         let mut val_loss = 0.0;
//         let mut correct = 0;
//         let mut total = 0;

//         for batch in val_loader.iter() {
//             let (images, targets) = batch;
//             let predictions = optimizer.model.forward(images);

//             // Compute validation metrics
//             val_loss += optimizer.model.training_step((images, targets)).to_scalar();

//             // Convert predictions to binary decisions (threshold at 0.5)
//             let pred_classes = predictions.map(|x| if x > 0.5 { 1.0 } else { 0.0 });
//             correct += pred_classes.eq(&targets).sum().to_scalar() as usize;
//             total += targets.shape()[0];
//         }

//         // println!(
//         //     "Epoch [{}/{}] complete. \
//         //     Training Loss: {:.4}, \
//         //     Validation Loss: {:.4}, \
//         //     Validation Accuracy: {:.2}%",
//         //     epoch + 1,
//         //     config.epochs,
//         //     train_loss / train_loader.len() as f64,
//         //     val_loss / val_loader.len() as f64,
//         //     100.0 * correct as f64 / total as f64
//         // );
//     }

//     Ok(optimizer.into_model())
// }

use std::time::Instant;

use crate::{
    data::{ClassificationBatch, ClassificationBatcher},
    dataset::LaptopDatasetLoader,
    model::LaptopClassifier,
};
use burn::{
    data::dataset::vision::ImageFolderDataset,
    nn::loss::CrossEntropyLossConfig,
    optim::SgdConfig,
    prelude::*,
    record::CompactRecorder,
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};

const NUM_CLASSES: u8 = 2;
const ARTIFACT_DIR: &str = "/tmp/custom-image-dataset";

impl<B: Backend> LaptopClassifier<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);

        let loss = CrossEntropyLossConfig::new()
            .with_weights(Some(Vec::from([1.0, 3.0])))
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

// impl<B: Backend> LaptopClassifier<B> {
//     pub fn forward_classification(
//         &self,
//         images: Tensor<B, 4>,
//         targets: Tensor<B, 2, Int>,
//     ) -> MultiLabelClassificationOutput<B> {
//         let output = self.forward(images);
//         let dims = targets.dims();

//         let targets_new = targets.reshape([dims[0], 1]);

//         let output_device = output.device();

//         let output_binary = if output.dims()[1] == 2 {
//             let batch_size = output.dims()[0] as i64;
//             // Use explicit end value from dimensions
//             output.slice([(0, batch_size), (1, 2)])
//         } else {
//             output
//         };

//         let loss = BinaryCrossEntropyLossConfig::new()
//             .init(&output_device)
//             .forward(output_binary.clone(), targets_new.clone());

//         MultiLabelClassificationOutput::new(loss, output_binary, targets_new)
//     }
// }

impl<B: AutodiffBackend> TrainStep<ClassificationBatch<B>, ClassificationOutput<B>>
    for LaptopClassifier<B>
{
    fn step(&self, batch: ClassificationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ClassificationBatch<B>, ClassificationOutput<B>>
    for LaptopClassifier<B>
{
    fn step(&self, batch: ClassificationBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: SgdConfig,
    #[config(default = 30)]
    pub num_epochs: usize,
    #[config(default = 16)] // 128?
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.02)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) {
    println!("Running train...");
    create_artifact_dir(ARTIFACT_DIR);

    config
        .save(format!("{ARTIFACT_DIR}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    // Dataloaders
    let batcher_train = ClassificationBatcher::<B>::new(device.clone());
    let batcher_valid = ClassificationBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::laptop_train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::laptop_val());

    // Learner config
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            LaptopClassifier::new(NUM_CLASSES.into(), &device),
            config.optimizer.init(),
            config.learning_rate,
        );

    // Training
    println!("Running fit...");
    let now = Instant::now();
    let model_trained = learner.fit(dataloader_train, dataloader_test);
    let elapsed = now.elapsed().as_secs();
    println!("Training completed in {}m{}s", (elapsed / 60), elapsed % 60);

    model_trained
        .save_file(format!("{ARTIFACT_DIR}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
