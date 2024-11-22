const NUM_CLASSES: u8 = 2;

use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, backend::Backend, Shape, Tensor, TensorData},
};
use image::io::Reader as ImageReader;
use std::path::Path;

use crate::{data::Normalizer, model::LaptopClassifier};

pub fn infer_from_file<B: Backend>(artifact_dir: &str, device: &B::Device, image_path: &str) {
    // Load the model
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist");

    println!("Model loaded...");

    let model: LaptopClassifier<B> =
        LaptopClassifier::new(NUM_CLASSES.into(), device).load_record(record);

    // Load and preprocess the image
    let img = ImageReader::open(image_path)
        .expect("Couldn't open image path")
        .decode()
        .expect("Couldn't decode image")
        .resize_exact(224, 224, image::imageops::FilterType::Lanczos3);

    // Convert to RGB if not already
    let img_rgb = img.to_rgb8();

    // println!("forward... {:?}", input.shape());
    let tensor_data: Vec<f32> = img_rgb
        .pixels()
        .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();

    let new_tensor = TensorData::new(tensor_data, Shape::new([224, 224, 3]));

    let new_data = Tensor::<B, 3>::from_data(new_tensor.convert::<B::FloatElem>(), &device)
        // permute(2, 0, 1)
        .swap_dims(2, 1) // [H, C, W]
        .swap_dims(1, 0); // [C, H, W]

    let new_data = new_data / 255;

    let mut images = Vec::new();
    images.push(new_data);

    let images: Tensor<B, 4> = Tensor::stack(images, 0);

    let normalizer = Normalizer::new(device);
    let images = normalizer.normalize(images);

    // Run inference
    let output = model.forward(images);
    let output_data = output.to_data();

    // Convert bytes to f32 values
    let values: Vec<f32> = output_data
        .bytes
        .chunks(4) // f32 is 4 bytes
        .map(|bytes| {
            let arr = [bytes[0], bytes[1], bytes[2], bytes[3]];
            f32::from_le_bytes(arr)
        })
        .collect();

    println!("Raw output logits: {:?}", values);

    // If you want to see them paired (for 2 classes)
    let predictions: Vec<_> = values.chunks(2).map(|chunk| (chunk[0], chunk[1])).collect();
    println!("Predictions (class0, class1): {:?}", predictions);

    // Rest of your code...
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();
    println!("Predicted class: {}", predicted);

    // Ok(predicted)
}

// pub fn infer_from_file<B: Backend>(artifact_dir: &str, device: &B::Device, image_path: &str) {
//     // Load the model
//     let record = CompactRecorder::new()
//         .load(format!("{artifact_dir}/model").into(), device)
//         .expect("Trained model should exist");

//     let model: LaptopClassifier<B> =
//         LaptopClassifier::new(NUM_CLASSES.into(), device).load_record(record);

//     // First load all training images just to compute normalizer
//     let train_path = Path::new("laptop_dataset_normal").join("train");
//     let mut training_tensors = Vec::new();

//     for entry in std::fs::read_dir(train_path)
//         .expect("Failed to read train directory")
//         .filter_map(Result::ok)
//         .filter(|e| e.path().extension().map_or(false, |ext| ext != "csv"))
//     {
//         let img = ImageReader::open(entry.path())
//             .expect("Couldn't open image")
//             .decode()
//             .expect("Couldn't decode image")
//             .resize_exact(32, 32, image::imageops::FilterType::Lanczos3)
//             .to_rgb8();

//         let tensor_data: Vec<f32> = img
//             .pixels()
//             .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
//             .collect();

//         let tensor = TensorData::new(tensor_data, Shape::new([32, 32, 3]));
//         let tensor = Tensor::<B, 3>::from_data(tensor.convert::<B::FloatElem>(), device)
//             .swap_dims(2, 1)
//             .swap_dims(1, 0);

//         training_tensors.push(tensor / 255.0);
//     }

//     let training_data: Tensor<B, 4> = Tensor::stack(training_tensors, 0);
//     let normalizer = Normalizer::from_dataset(&training_data, device);

//     // Now only process and infer on the single image
//     let img = ImageReader::open(image_path)
//         .expect("Couldn't open image path")
//         .decode()
//         .expect("Couldn't decode image")
//         .resize_exact(32, 32, image::imageops::FilterType::Lanczos3)
//         .to_rgb8();

//     let tensor_data: Vec<f32> = img
//         .pixels()
//         .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
//         .collect();

//     let new_tensor = TensorData::new(tensor_data, Shape::new([32, 32, 3]));
//     let new_data = Tensor::<B, 3>::from_data(new_tensor.convert::<B::FloatElem>(), device)
//         .swap_dims(2, 1)
//         .swap_dims(1, 0);

//     let new_data = new_data / 255.0;

//     let mut image_vec = Vec::new();
//     image_vec.push(new_data);

//     // Only stack, normalize and forward the single inference image
//     let single_image = Tensor::stack(image_vec, 0);
//     let normalized_image = normalizer.normalize(single_image);

//     let output = model.forward(normalized_image);
//     let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

//     println!("Predicted class: {}", predicted);
// }
