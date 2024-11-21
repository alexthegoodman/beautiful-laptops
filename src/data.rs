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

use image::GenericImageView;
const TARGET_SIZE: (u32, u32) = (800, 800);

impl<B: Backend> Normalizer<B> {
    /// Creates a new normalizer by computing statistics from the dataset tensor
    pub fn from_dataset(images: &Tensor<B, 4>, device: &Device<B>) -> Self {
        let dims = images.dims();
        let n_images = dims[0] as f32;
        let n_pixels = (dims[2] * dims[3]) as f32;

        // Calculate mean per channel
        let means = images
            .clone()
            .sum_dim(0) // Sum over batch
            .sum_dim(1) // Sum over height
            .sum_dim(1) // Sum over width
            .clone()
            / (n_images * n_pixels);

        // Calculate squared values and their mean
        let squared = images.clone() * images.clone();
        let squared_means =
            squared.sum_dim(0).sum_dim(1).sum_dim(1).clone() / (n_images * n_pixels);

        // Calculate std: sqrt(E[X²] - E[X]²)
        let variances = squared_means - (means.clone() * means.clone());
        let stds = variances.sqrt();

        // Reshape to [1, C, 1, 1] for broadcasting
        let mean = means.reshape([1, 3, 1, 1]);
        let std = stds.reshape([1, 3, 1, 1]);

        Self { mean, std }
    }

    /// Normalizes the input image tensor
    pub fn normalize(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        (input - self.mean.clone()) / self.std.clone()
    }
}

#[derive(Clone)]
pub struct ClassificationBatcher<B: Backend> {
    // normalizer: Normalizer<B>,
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> ClassificationBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            // normalizer: Normalizer::<B>::new(&device),
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
                    // Create a tensor with shape [1, 1] for each item
                    Tensor::<B, 2, Int>::from_data(
                        TensorData::new(vec![y as i32], Shape::new([1, 1])),
                        &self.device,
                    )
                } else {
                    panic!("Invalid target type")
                }
            })
            .collect();

        let images = items
            .into_iter()
            // TODO: adjust 32x32 to correctness?
            .map(|item| TensorData::new(image_as_vec_u8(item), Shape::new([32, 32, 3])))
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

        let normalizer = Normalizer::from_dataset(&images, &self.device);
        let images = normalizer.normalize(images);

        ClassificationBatch { images, targets }
    }
}
