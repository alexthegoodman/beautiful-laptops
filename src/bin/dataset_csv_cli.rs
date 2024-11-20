use std::{fs, path::PathBuf};

fn generate_template(output_path: PathBuf, limit: Option<usize>) -> std::io::Result<()> {
    let dataset_dir = PathBuf::from("laptop_dataset");
    let mut content = String::from("filename,is_attractive\n");
    let mut count = 0;

    fn visit_dirs(
        dir: &PathBuf,
        content: &mut String,
        count: &mut usize,
        limit: Option<usize>,
    ) -> std::io::Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_dir() {
                    visit_dirs(&path, content, count, limit)?;
                } else if path.extension().map_or(false, |ext| ext == "jpg") {
                    if let Some(filename) = path.strip_prefix("laptop_dataset").unwrap().to_str() {
                        // Remove leading slash if present
                        let filename = filename.trim_start_matches('/');
                        content.push_str(&format!("{},\n", filename));
                        *count += 1;

                        if let Some(limit) = limit {
                            if *count >= limit {
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    visit_dirs(&dataset_dir, &mut content, &mut count, limit)?;

    fs::write(output_path, content)?;
    println!("Generated template with {} entries", count);
    Ok(())
}

fn main() {
    let mut output = PathBuf::from("annotations_template.csv");
    generate_template(output, None).expect("Couldn't generate csv template");
}
