use image::io::Reader as ImageReader;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize)]
struct NormalizationStats {
    mean: Vec<f32>,
    std: Vec<f32>,
}

fn main() {
    let dataset_path = Path::new("laptop_dataset_normal/train");
    println!("Loading images from {:?}...", dataset_path);

    let entries: Vec<_> = fs::read_dir(dataset_path)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();

    // Process images in parallel to get per-image sums
    let image_sums: Vec<_> = entries
        .par_iter()
        .map(|entry| {
            let img = ImageReader::open(entry.path())
                .unwrap()
                .decode()
                .unwrap()
                .to_rgb8();

            // Calculate sum for each channel
            let mut sums = vec![0.0; 3];
            for pixel in img.pixels() {
                for c in 0..3 {
                    sums[c] += pixel[c] as f32 / 255.0;
                }
            }
            sums
        })
        .collect();

    // Calculate means
    let num_pixels = (800 * 800 * entries.len()) as f32;
    let means: Vec<f32> = (0..3)
        .map(|c| image_sums.iter().map(|sums| sums[c]).sum::<f32>() / num_pixels)
        .collect();

    println!("Calculated means, now computing standard deviations...");

    // Calculate squared differences in parallel
    let squared_diff_sums: Vec<_> = entries
        .par_iter()
        .map(|entry| {
            let img = ImageReader::open(entry.path())
                .unwrap()
                .decode()
                .unwrap()
                .to_rgb8();

            let mut squared_diffs = vec![0.0; 3];
            for pixel in img.pixels() {
                for c in 0..3 {
                    let diff = pixel[c] as f32 / 255.0 - means[c];
                    squared_diffs[c] += diff * diff;
                }
            }
            squared_diffs
        })
        .collect();

    // Calculate standard deviations
    let stds: Vec<f32> = (0..3)
        .map(|c| {
            let sum_squared_diff: f32 = squared_diff_sums.iter().map(|diffs| diffs[c]).sum();
            (sum_squared_diff / num_pixels).sqrt()
        })
        .collect();

    let stats = NormalizationStats {
        mean: means,
        std: stds,
    };

    // Save statistics to JSON file
    let json = serde_json::to_string_pretty(&stats).unwrap();
    fs::write("normalization_stats.json", json).unwrap();

    println!("Statistics saved to normalization_stats.json");
    println!("Mean: {:?}", stats.mean);
    println!("Std: {:?}", stats.std);
}
