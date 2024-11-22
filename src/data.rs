use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::vision::{Annotation, ImageDatasetItem, PixelDepth},
    },
    prelude::*,
};
use image::{DynamicImage, RgbImage};

#[derive(Clone)]
pub struct Normalizer<B: Backend> {
    pub mean: Tensor<B, 4>,
    pub std: Tensor<B, 4>,
}

// impl<B: Backend> Normalizer<B> {
//     /// Creates a new normalizer by computing statistics from the dataset tensor
//     /// Expects input tensor of shape [batch_size, channels, height, width]
//     pub fn from_dataset(images: &Tensor<B, 4>, device: &Device<B>) -> Self {
//         let dims = images.dims();
//         let n_images = dims[0] as f32;
//         let n_pixels = (dims[2] * dims[3]) as f32;

//         // Calculate mean per channel
//         let means = images
//             .clone()
//             .sum_dim(0) // Sum over batch
//             .sum_dim(1) // Sum over height
//             .sum_dim(1) // Sum over width
//             .clone()
//             / (n_images * n_pixels);

//         // Calculate squared values and their mean
//         let squared = images.clone() * images.clone(); // Element-wise multiplication instead of powf
//         let squared_means =
//             squared.sum_dim(0).sum_dim(1).sum_dim(1).clone() / (n_images * n_pixels);

//         // Calculate std: sqrt(E[X²] - E[X]²)
//         let variances = squared_means - (means.clone() * means.clone());
//         let stds = variances.sqrt();

//         // Reshape to [1, C, 1, 1] for broadcasting
//         let mean = means.reshape([1, 3, 1, 1]);
//         let std = stds.reshape([1, 3, 1, 1]);

//         Self { mean, std }
//     }

//     /// Alternative constructor with custom mean and std values
//     pub fn with_values(mean: [f32; 3], std: [f32; 3], device: &Device<B>) -> Self {
//         let mean = Tensor::<B, 1>::from_floats(mean, device).reshape([1, 3, 1, 1]);
//         let std = Tensor::<B, 1>::from_floats(std, device).reshape([1, 3, 1, 1]);
//         Self { mean, std }
//     }

//     /// Normalizes the input image.
//     pub fn normalize(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
//         (input - self.mean.clone()) / self.std.clone()
//     }
// }

// use image::GenericImageView;
// const TARGET_SIZE: (u32, u32) = (800, 800);

// impl<B: Backend> Normalizer<B> {
//     /// Creates a new normalizer by computing statistics from the dataset tensor
//     pub fn from_dataset(images: &Tensor<B, 4>, device: &Device<B>) -> Self {
//         let dims = images.dims();
//         let n_images = dims[0] as f32;
//         let n_pixels = (dims[2] * dims[3]) as f32;

//         // Calculate mean per channel
//         let means = images
//             .clone()
//             .sum_dim(0) // Sum over batch
//             .sum_dim(1) // Sum over height
//             .sum_dim(1) // Sum over width
//             .clone()
//             / (n_images * n_pixels);

//         // Calculate squared values and their mean
//         let squared = images.clone() * images.clone();
//         let squared_means =
//             squared.sum_dim(0).sum_dim(1).sum_dim(1).clone() / (n_images * n_pixels);

//         // Calculate std: sqrt(E[X²] - E[X]²)
//         let variances = squared_means - (means.clone() * means.clone());
//         let stds = variances.sqrt();

//         // Reshape to [1, C, 1, 1] for broadcasting
//         let mean = means.reshape([1, 3, 1, 1]);
//         let std = stds.reshape([1, 3, 1, 1]);

//         Self { mean, std }
//     }

//     /// Normalizes the input image tensor
//     pub fn normalize(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
//         (input - self.mean.clone()) / self.std.clone()
//     }
// }

const MEAN: [f32; 3] = [0.618738, 0.61940384, 0.6039725];
const STD: [f32; 3] = [0.36482072, 0.3620487, 0.36871767];

impl<B: Backend> Normalizer<B> {
    /// Creates a new normalizer.
    pub fn new(device: &Device<B>) -> Self {
        let mean = Tensor::<B, 1>::from_floats(MEAN, device).reshape([1, 3, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(STD, device).reshape([1, 3, 1, 1]);
        Self { mean, std }
    }

    /// Normalizes the input image according to the CIFAR-10 dataset.
    ///
    /// The input image should be in the range [0, 1].
    /// The output image will be in the range [-1, 1].
    ///
    /// The normalization is done according to the following formula:
    /// `input = (input - mean) / std`
    pub fn normalize(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        (input - self.mean.clone()) / self.std.clone()
    }
}

#[derive(Clone)]
pub struct ClassificationBatcher<B: Backend> {
    normalizer: Normalizer<B>,
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ClassificationBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            normalizer: Normalizer::<B>::new(&device),
            device,
        }
    }
}

impl<B: Backend> Batcher<ImageDatasetItem, ClassificationBatch<B>> for ClassificationBatcher<B> {
    fn batch(&self, items: Vec<ImageDatasetItem>) -> ClassificationBatch<B> {
        fn image_as_vec_u8(item: ImageDatasetItem) -> Vec<u8> {
            // Convert Vec<PixelDepth> to Vec<u8> (we know that CIFAR images are u8)
            item.image
                .into_iter()
                .map(|p: PixelDepth| -> u8 { p.try_into().unwrap() })
                .collect::<Vec<u8>>()
        }

        let targets = items
            .iter()
            .map(|item| {
                if let Annotation::Label(y) = item.annotation {
                    Tensor::<B, 1, Int>::from_data(
                        TensorData::new(vec![y as i32], Shape::new([1])),
                        &self.device,
                    )
                } else {
                    panic!("Invalid target type")
                }
            })
            .collect();

        let images = items
            .into_iter()
            .map(|item| TensorData::new(image_as_vec_u8(item), Shape::new([224, 224, 3])))
            .map(|data| {
                Tensor::<B, 3>::from_data(data.convert::<B::FloatElem>(), &self.device)
                    // permute(2, 0, 1)
                    .swap_dims(2, 1) // [H, C, W]
                    .swap_dims(1, 0) // [C, H, W]
            })
            .map(|tensor| tensor / 255) // normalize between [0, 1]
            .collect();

        let images: Tensor<B, 4> = Tensor::stack(images, 0);
        let targets = Tensor::cat(targets, 0);

        let images = self.normalizer.normalize(images);

        // println!("batch dims {:?}", images.dims());

        ClassificationBatch { images, targets }
    }
}

// impl<B: Backend> Batcher<ImageDatasetItem, ClassificationBatch<B>> for ClassificationBatcher<B> {
//     fn batch(&self, items: Vec<ImageDatasetItem>) -> ClassificationBatch<B> {
//         fn image_as_vec_u8(item: ImageDatasetItem) -> Vec<u8> {
//             // Convert Vec<PixelDepth> to Vec<u8> (we know that CIFAR images are u8)
//             item.image
//                 .into_iter()
//                 .map(|p: PixelDepth| -> u8 { p.try_into().unwrap() })
//                 .collect::<Vec<u8>>()
//         }

//         // Print a sample of original pixel values from first image
//         if let Some(first_item) = items.first() {
//             println!("Original first few pixels: {:?}", &first_item.image[..10]);
//             println!("Annotation: {:?}", first_item.annotation);
//         }

//         let targets = items
//             .iter()
//             .map(|item| {
//                 if let Annotation::Label(y) = item.annotation {
//                     println!("Target label: {}", y); // Log each target
//                     Tensor::<B, 1, Int>::from_data(
//                         TensorData::new(vec![y as i32], Shape::new([1])),
//                         &self.device,
//                     )
//                 } else {
//                     panic!("Invalid target type")
//                 }
//             })
//             .collect();

//         let images = items
//             .into_iter()
//             .enumerate() // Add enumerate to track image index
//             .map(|(i, item)| {
//                 let vec_u8 = image_as_vec_u8(item);
//                 if i == 0 {
//                     println!("After u8 conversion (first few): {:?}", &vec_u8[..10]);
//                 }

//                 let data = TensorData::new(vec_u8, Shape::new([800, 800, 3]));
//                 let tensor =
//                     Tensor::<B, 3>::from_data(data.convert::<B::FloatElem>(), &self.device);

//                 if i == 0 {
//                     // Log tensor data after conversion to float
//                     let tensor_data = tensor.to_data();
//                     let values: Vec<f32> = tensor_data
//                         .bytes
//                         .chunks(4)
//                         .take(10) // First few values
//                         .map(|bytes| {
//                             let arr = [bytes[0], bytes[1], bytes[2], bytes[3]];
//                             f32::from_le_bytes(arr)
//                         })
//                         .collect();
//                     println!("After float conversion (first few): {:?}", values);
//                 }

//                 let permuted = tensor.swap_dims(2, 1).swap_dims(1, 0);

//                 let normalized = permuted / 255.0;

//                 if i == 0 {
//                     // Log after normalization
//                     let norm_data = normalized.to_data();
//                     let norm_values: Vec<f32> = norm_data
//                         .bytes
//                         .chunks(4)
//                         .take(10)
//                         .map(|bytes| {
//                             let arr = [bytes[0], bytes[1], bytes[2], bytes[3]];
//                             f32::from_le_bytes(arr)
//                         })
//                         .collect();
//                     println!("After [0,1] normalization (first few): {:?}", norm_values);
//                 }

//                 normalized
//             })
//             .collect();

//         let images: Tensor<B, 4> = Tensor::stack(images, 0);

//         // Log final normalized batch
//         let batch_data = images.to_data();
//         let batch_values: Vec<f32> = batch_data
//             .bytes
//             .chunks(4)
//             .take(20) // First few values from batch
//             .map(|bytes| {
//                 let arr = [bytes[0], bytes[1], bytes[2], bytes[3]];
//                 f32::from_le_bytes(arr)
//             })
//             .collect();
//         println!("Final batch values (first few): {:?}", batch_values);

//         // Log after final normalization
//         let images = self.normalizer.normalize(images);
//         let final_data = images.to_data();
//         let final_values: Vec<f32> = final_data
//             .bytes
//             .chunks(4)
//             .take(20)
//             .map(|bytes| {
//                 let arr = [bytes[0], bytes[1], bytes[2], bytes[3]];
//                 f32::from_le_bytes(arr)
//             })
//             .collect();
//         println!("After final normalization: {:?}", final_values);

//         let targets = Tensor::cat(targets, 0);
//         ClassificationBatch { images, targets }
//     }
// }
