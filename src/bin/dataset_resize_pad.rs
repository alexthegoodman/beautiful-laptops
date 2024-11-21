use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

const TARGET_SIZE: (u32, u32) = (800, 800);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_path = Path::new("laptop_dataset");
    let output_base = Path::new("laptop_dataset_normal");

    // Create output directories
    for dir in &["train", "val"] {
        fs::create_dir_all(output_base.join(dir))?;
    }

    // Process both train and val directories
    for dir in &["train", "val"] {
        let input_dir = base_path.join(dir);
        let output_dir = output_base.join(dir);

        println!("Processing {} directory...", dir);

        // Get all image files
        let entries: Vec<_> = fs::read_dir(&input_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .map(|ext| ext.to_str().unwrap_or("").to_lowercase())
                    .map(|ext| vec!["jpg", "jpeg", "png"].contains(&ext.as_str()))
                    .unwrap_or(false)
            })
            .collect();

        // Process images in parallel using rayon
        entries.par_iter().for_each(|entry| {
            let path = entry.path();
            if let Err(e) = process_image(&path, &output_dir) {
                eprintln!("Error processing {}: {}", path.display(), e);
            }
        });
    }

    println!("Dataset preprocessing complete!");
    Ok(())
}

fn resize_and_pad(image: &DynamicImage) -> DynamicImage {
    let (width, height) = image.dimensions();

    // Calculate scaling factor to fit within 800x800 while maintaining aspect ratio
    let scale = f32::min(
        TARGET_SIZE.0 as f32 / width as f32,
        TARGET_SIZE.1 as f32 / height as f32,
    );

    // New dimensions after scaling
    let new_width = (width as f32 * scale) as u32;
    let new_height = (height as f32 * scale) as u32;

    // Resize image maintaining aspect ratio
    let resized = image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);

    // Create new image with padding
    let mut padded = RgbImage::new(TARGET_SIZE.0, TARGET_SIZE.1);

    // Calculate padding
    let pad_x = (TARGET_SIZE.0 - new_width) / 2;
    let pad_y = (TARGET_SIZE.1 - new_height) / 2;

    // Fill with white background
    for y in 0..TARGET_SIZE.1 {
        for x in 0..TARGET_SIZE.0 {
            padded.put_pixel(x, y, Rgb([255, 255, 255]));
        }
    }

    // Copy resized image onto padded background
    image::imageops::overlay(&mut padded, &resized.to_rgb8(), pad_x.into(), pad_y.into());

    DynamicImage::ImageRgb8(padded)
}

fn process_image(input_path: &Path, output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Load image
    let img = image::open(input_path)?;

    // Resize and pad
    let processed = resize_and_pad(&img);

    // Create output path with same filename
    let output_path = output_dir.join(input_path.file_name().unwrap());

    // Save processed image
    processed.save(&output_path)?;

    println!("Processed: {}", input_path.display());
    Ok(())
}
