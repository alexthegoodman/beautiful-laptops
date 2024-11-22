use burn::data::{dataset::vision::ImageFolderDataset, network::downloader};
use reqwest;
use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const DATASET_API: &str = "https://datasets-server.huggingface.co/rows?dataset=amaye15%2Flaptop&config=default&split=train";
const BATCH_SIZE: usize = 100;

#[derive(Debug, Deserialize)]
pub struct HuggingFaceResponse {
    pub features: Vec<Feature>,
    pub rows: Vec<Row>,
}

#[derive(Debug, Deserialize)]
pub struct Feature {
    pub feature_idx: usize,
    pub name: String,
    #[serde(rename = "type")]
    pub feature_type: FeatureType,
}

#[derive(Debug, Deserialize)]
pub struct FeatureType {
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(rename = "_type")]
    pub type_name: String,
    #[serde(default)]
    pub names: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct Row {
    pub row_idx: usize,
    pub row: RowData,
    pub truncated_cells: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct RowData {
    pub pixel_values: ImageData,
    pub label: usize,
}

#[derive(Debug, Deserialize)]
pub struct ImageData {
    pub src: String,
    pub height: usize,
    pub width: usize,
}

pub fn download_dataset(force: bool, limit: Option<usize>) -> PathBuf {
    let client = reqwest::blocking::Client::new();
    let mut offset = 0;
    let batch_size = 100;

    let test_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
    let dataset_dir = test_dir.join("laptop_dataset");
    let train_dir = dataset_dir.join("train");
    let val_dir = dataset_dir.join("val");

    // Check if dataset is already downloaded
    if dataset_dir.exists() {
        return dataset_dir;
    }

    // Create necessary directories
    fs::create_dir_all(&train_dir).unwrap();
    fs::create_dir_all(&val_dir).unwrap();

    loop {
        let url = format!(
            "{}&offset={}&length={}",
            "https://datasets-server.huggingface.co/rows?dataset=amaye15%2Flaptop&config=default&split=train",
            offset,
            batch_size
        );

        println!("url {:?}", url);

        let response: HuggingFaceResponse = client
            .get(&url)
            .send()
            .expect("Failed to fetch from HuggingFace API")
            .json()
            .expect("Failed to parse JSON response");

        // Process features if it's the first batch
        if offset == 0 {
            println!("Dataset features:");
            for feature in &response.features {
                println!("- {} ({})", feature.name, feature.feature_type.type_name);
                if let Some(names) = &feature.feature_type.names {
                    println!("  Labels: {:?}", names);
                }
            }
        }

        if response.rows.is_empty() {
            break Default::default();
        }

        for row in response.rows {
            if let Some(limit) = limit {
                if offset >= limit {
                    break;
                }
            }

            let image = row.row.pixel_values;
            println!(
                "Downloading image {} ({}x{})",
                row.row_idx, image.width, image.height
            );

            let image_bytes = downloader::download_file_as_bytes(&image.src, "temp.jpg");

            // Determine if this should go to train or val (80/20 split)
            let target_dir = if (offset + row.row_idx) % 5 == 0 {
                &val_dir
            } else {
                &train_dir
            };

            // Save image
            let image_path = target_dir.join(format!("laptop_{}.jpg", offset + row.row_idx));
            let mut file = fs::File::create(image_path).unwrap();
            file.write_all(&image_bytes).unwrap();

            // Log progress
            println!("Downloaded image {}", image.src);

            offset += 1;
        }
    }

    return dataset_dir;
}

use std::fs::File;
use std::io::{BufRead, BufReader};

pub trait LaptopDatasetLoader {
    fn laptop_train() -> Self;
    fn laptop_val() -> Self;
}

impl LaptopDatasetLoader for ImageFolderDataset {
    fn laptop_train() -> Self {
        let test_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
        let dataset_dir = test_dir.join("laptop_dataset_224");
        let annotations_path = dataset_dir.join("train").join("annotations.csv");

        let reader = BufReader::new(File::open(annotations_path).unwrap());
        let mut items = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            if i == 0 {
                // skip header
                continue;
            }

            let line = line.unwrap();
            let parts: Vec<&str> = line.split(',').collect();

            if parts.len() != 2 {
                panic!("Invalid annotation format at line {}", i + 1);
            }

            let filename = parts[0];
            let label = parts[1];

            let file_path = dataset_dir.join(filename);
            items.push((file_path, label.to_string()));
        }

        let classes = &["false", "true"];
        Self::new_classification_with_items(items, classes).unwrap()
    }

    fn laptop_val() -> Self {
        let test_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
        let dataset_dir = test_dir.join("laptop_dataset_224");
        let annotations_path = dataset_dir.join("val").join("annotations.csv");

        let reader = BufReader::new(File::open(annotations_path).unwrap());
        let mut items = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            if i == 0 {
                // skip header
                continue;
            }

            let line = line.unwrap();
            let parts: Vec<&str> = line.split(',').collect();

            if parts.len() != 2 {
                panic!("Invalid annotation format at line {}", i + 1);
            }

            let filename = parts[0];
            let label = parts[1];

            let file_path = dataset_dir.join(filename);
            items.push((file_path, label.to_string()));
        }

        let classes = &["false", "true"];
        Self::new_classification_with_items(items, classes).unwrap()
    }
}

// pub fn download_dataset() -> PathBuf {
//     println!("Download Dataset");
//     let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
//     let dataset_dir = example_dir.join("laptop_dataset");
//     let train_dir = dataset_dir.join("train");
//     let val_dir = dataset_dir.join("val");

//     // Check if dataset is already downloaded
//     if dataset_dir.exists() {
//         return dataset_dir;
//     }

//     // Create necessary directories
//     fs::create_dir_all(&train_dir).unwrap();
//     fs::create_dir_all(&val_dir).unwrap();

//     // Initialize HTTP client
//     let client = reqwest::blocking::Client::new();
//     let mut offset = 0;

//     loop {
//         let url = format!("{}&offset={}&length={}", DATASET_API, offset, BATCH_SIZE);
//         let response: HuggingFaceResponse = client.get(&url).send().unwrap().json().unwrap();

//         if response.rows.is_empty() {
//             break;
//         }

//         for (idx, row) in response.rows.iter().enumerate() {
//             let image_url = &row.row.image.src;
//             let laptop_name = &row.row.laptop_name;

//             // Download image
//             let image_bytes = downloader::download_file_as_bytes(image_url, "temp.jpg");

//             // Determine if this should go to train or val (80/20 split)
//             let target_dir = if (offset + idx) % 5 == 0 {
//                 &val_dir
//             } else {
//                 &train_dir
//             };

//             // Save image
//             let image_path = target_dir.join(format!("laptop_{}.jpg", offset + idx));
//             let mut file = fs::File::create(image_path).unwrap();
//             file.write_all(&image_bytes).unwrap();

//             // Log progress
//             println!("Downloaded image {} - {}", offset + idx, laptop_name);
//         }

//         offset += BATCH_SIZE;

//         if offset >= response.num_rows_total {
//             break;
//         }
//     }

//     // Create labels file
//     let labels_file = dataset_dir.join("labels.txt");
//     fs::write(labels_file, "attractive\nnot_attractive").unwrap();

//     dataset_dir
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_dataset_download() {
//         let dataset_path = download_dataset();
//         assert!(dataset_path.exists());
//         assert!(dataset_path.join("train").exists());
//         assert!(dataset_path.join("val").exists());
//         assert!(dataset_path.join("labels.txt").exists());
//     }
// }
