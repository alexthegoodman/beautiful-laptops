use image::{self, ImageBuffer, Rgb};
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub fn resize_dataset_images(source_dir: &Path, target_dir: &Path) -> std::io::Result<()> {
    // Create target directory if it doesn't exist
    std::fs::create_dir_all(target_dir)?;

    // Collect all image paths first
    let image_paths: Vec<_> = std::fs::read_dir(source_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();

            // Filter for image files
            let is_image = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ["jpg", "jpeg", "png"].contains(&ext))
                .unwrap_or(false);

            if is_image {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    let total_images = image_paths.len();
    let processed_count = Arc::new(AtomicUsize::new(0));
    let target_dir = Arc::new(target_dir.to_path_buf());

    // Process images in parallel
    image_paths.par_iter().for_each(|path| {
        let target_dir = Arc::clone(&target_dir);

        // Load and resize image
        if let Ok(img) = image::open(path) {
            let resized = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);

            // Create target path with same filename
            let target_path = target_dir.join(path.file_name().unwrap());

            // Save resized image
            if let Err(e) = resized.save(&target_path) {
                eprintln!("Failed to save {:?}: {}", path, e);
            }
        } else {
            eprintln!("Failed to open {:?}", path);
        }

        // Update and print progress
        let completed = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
        println!("Processed {}/{} images", completed, total_images);
    });

    println!("Finished resizing {} images!", total_images);
    Ok(())
}

fn main() -> std::io::Result<()> {
    let source = Path::new("laptop_dataset_normal/val");
    let target = Path::new("laptop_dataset_224/val");

    println!("Starting parallel image resizing...");
    resize_dataset_images(source, target)?;

    Ok(())
}
