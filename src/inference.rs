// use burn::{
//     data::{
//         dataloader::batcher::Batcher,
//         dataset::vision::{Annotation, ImageDatasetItem},
//     },
//     module::Module,
//     record::{CompactRecorder, Recorder},
//     tensor::backend::Backend,
// };

// use crate::{data::ClassificationBatcher, model::LaptopClassifier};

const NUM_CLASSES: u8 = 2;

// pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: ImageDatasetItem) {
//     let record = CompactRecorder::new()
//         .load(format!("{artifact_dir}/model").into(), &device)
//         .expect("Trained model should exist");

//     let model: LaptopClassifier<B> =
//         LaptopClassifier::new(NUM_CLASSES.into(), &device).load_record(record);

//     let mut label = 0;
//     if let Annotation::Label(category) = item.annotation {
//         label = category;
//     };
//     let batcher = ClassificationBatcher::new(device);
//     let batch = batcher.batch(vec![item]);
//     let output = model.forward(batch.images);
//     let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();
//     println!("Predicted {} Expected {:?}", predicted, label);
// }

use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Shape, Tensor, TensorData},
};
use image::io::Reader as ImageReader;
use std::path::Path;

use crate::{data::Normalizer, model::LaptopClassifier};

pub fn infer_from_file<B: Backend>(artifact_dir: &str, device: &B::Device, image_path: &str) {
    // Load the model
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist");

    let model: LaptopClassifier<B> =
        LaptopClassifier::new(NUM_CLASSES.into(), device).load_record(record);

    // Load and preprocess the image
    let img = ImageReader::open(image_path)
        .expect("Couldn't open image path")
        .decode()
        .expect("Couldn't decode image")
        .resize_exact(800, 800, image::imageops::FilterType::Lanczos3);

    // Convert to RGB if not already
    let img_rgb = img.to_rgb8();

    // Convert to tensor [1, 3, 800, 800]
    // let tensor_data: Vec<f32> = img_rgb
    //     .pixels()
    //     .flat_map(|p| {
    //         [
    //             p[0] as f32 / 255.0,
    //             p[1] as f32 / 255.0,
    //             p[2] as f32 / 255.0,
    //         ]
    //     })
    //     .collect();

    // println!("creat tensor data...");

    // let tensor_data = TensorData::new(tensor_data, Shape::new([1, 3, 800, 800]));

    // Create input tensor with correct shape
    // let input = Tensor::from_data(tensor_data, device);

    // println!("forward... {:?}", input.shape());
    let tensor_data: Vec<f32> = img_rgb
        .pixels()
        .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();

    let new_tensor = TensorData::new(tensor_data, Shape::new([32, 32, 3]));

    let new_data = Tensor::<B, 3>::from_data(new_tensor.convert::<B::FloatElem>(), &device)
        // permute(2, 0, 1)
        .swap_dims(2, 1) // [H, C, W]
        .swap_dims(1, 0); // [C, H, W]

    let new_data = new_data / 255;

    let mut images = Vec::new();
    images.push(new_data);

    let images: Tensor<B, 4> = Tensor::stack(images, 0);

    let normalizer = Normalizer::from_dataset(&images, &device);
    let images = normalizer.normalize(images);

    // Run inference
    let output = model.forward(images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted class: {}", predicted);
    // Ok(predicted)
}

// Alternative version that also normalizes using the same normalizer as training
pub fn infer_from_file_with_normalizer<B: Backend>(
    artifact_dir: &str,
    device: &B::Device,
    image_path: &str,
    normalizer: &Normalizer<B>,
) {
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist");

    let model: LaptopClassifier<B> =
        LaptopClassifier::new(NUM_CLASSES.into(), device).load_record(record);

    // Load and preprocess the image
    let img = ImageReader::open(image_path)
        .expect("Couldn't open image path")
        .decode()
        .expect("Couldn't decode image")
        .resize_exact(800, 800, image::imageops::FilterType::Lanczos3);

    let img_rgb = img.to_rgb8();

    // Convert to tensor
    let tensor_data: Vec<f32> = img_rgb
        .pixels()
        .flat_map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    // let input = Tensor::from_vec(tensor_data, (1, 3, 800, 800), device);

    let tensor_data = TensorData::new(tensor_data, Shape::new([1, 3, 800, 800]));

    // Create input tensor with correct shape
    let input = Tensor::from_data(tensor_data, device);

    // Apply same normalization as training
    let normalized_input = normalizer.normalize(input);

    // Run inference
    let output = model.forward(normalized_input);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted class: {}", predicted);
    // Ok(predicted)
}

// // Usage example in main or wherever:
// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     let device = burn::backend::wgpu::Wgpu::default().device();

//     // Basic usage
//     let prediction = infer_from_file(
//         "/tmp/custom-image-dataset",
//         &device,
//         "path/to/your/image.jpg"
//     )?;

//     // Or with normalizer if you want to use the same normalization as training
//     let normalizer = Normalizer::load_from_file("/path/to/saved/normalizer")?;
//     let prediction = infer_from_file_with_normalizer(
//         "/tmp/custom-image-dataset",
//         &device,
//         "path/to/your/image.jpg",
//         &normalizer
//     )?;

//     Ok(())
// }
