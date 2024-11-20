use beautiful_laptops::dataset::download_dataset;
use clap::{command, value_parser, Arg, ArgAction, Command};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::PathBuf,
};

fn main() {
    let matches = cli().get_matches();
    let values = Value::from_matches(&matches);
    println!("{values:#?}");
}

fn cli() -> Command {
    command!().next_help_heading("DATASET COMMANDS").args([
        Arg::new("download")
            .long("download")
            .action(ArgAction::SetTrue)
            .help("Download dataset from HuggingFace"),
        Arg::new("force")
            .long("force")
            .action(ArgAction::SetTrue)
            .help("Force redownload of existing dataset"),
        Arg::new("init")
            .long("init")
            .action(ArgAction::SetTrue)
            .help("Initialize annotations file"),
        Arg::new("yes")
            .long("yes")
            .action(ArgAction::SetTrue)
            .help("Skip confirmation prompts"),
        Arg::new("stats")
            .long("stats")
            .action(ArgAction::SetTrue)
            .help("Show dataset statistics"),
        Arg::new("limit")
            .long("limit")
            .value_parser(value_parser!(usize))
            .help("Limit number of images to download"),
    ])
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Value {
    Bool(bool),
    Number(usize),
}

impl Value {
    pub fn from_matches(matches: &clap::ArgMatches) -> Vec<(clap::Id, Self)> {
        let mut values = BTreeMap::new();
        for id in matches.ids() {
            if matches.try_get_many::<clap::Id>(id.as_str()).is_ok() {
                continue;
            }
            let value_source = matches
                .value_source(id.as_str())
                .expect("id came from matches");
            if value_source != clap::parser::ValueSource::CommandLine {
                continue;
            }
            if Self::extract::<bool>(matches, id, &mut values) {
                continue;
            }
            if Self::extract::<usize>(matches, id, &mut values) {
                continue;
            }
            unimplemented!("unknown type for {id}: {matches:?}");
        }

        let values = values.into_values().collect::<Vec<_>>();

        // Handle the commands based on arguments
        if let Some(values) = handle_commands(&values) {
            values
        } else {
            println!("No valid command combination provided");
            std::process::exit(1);
        }
    }

    fn extract<T: Clone + Into<Value> + Send + Sync + 'static>(
        matches: &clap::ArgMatches,
        id: &clap::Id,
        output: &mut BTreeMap<usize, (clap::Id, Self)>,
    ) -> bool {
        match matches.try_get_many::<T>(id.as_str()) {
            Ok(Some(values)) => {
                for (value, index) in values.zip(
                    matches
                        .indices_of(id.as_str())
                        .expect("id came from matches"),
                ) {
                    output.insert(index, (id.clone(), value.clone().into()));
                }
                true
            }
            Ok(None) => {
                unreachable!("ids only reports what is present")
            }
            Err(clap::parser::MatchesError::UnknownArgument { .. }) => {
                unreachable!("id came from matches")
            }
            Err(clap::parser::MatchesError::Downcast { .. }) => false,
            Err(_) => {
                unreachable!("id came from matches")
            }
        }
    }
}

impl From<bool> for Value {
    fn from(other: bool) -> Self {
        Self::Bool(other)
    }
}

impl From<usize> for Value {
    fn from(other: usize) -> Self {
        Self::Number(other)
    }
}

fn handle_commands(values: &[(clap::Id, Value)]) -> Option<Vec<(clap::Id, Value)>> {
    let mut download = false;
    let mut force = false;
    let mut init = false;
    // let mut template = false;
    let mut yes = false;
    let mut stats = false;
    let mut limit = None;

    for (id, value) in values {
        match (id.as_str(), value) {
            ("download", Value::Bool(true)) => download = true,
            ("force", Value::Bool(true)) => force = true,
            ("init", Value::Bool(true)) => init = true,
            // ("template", Value::Bool(true)) => template = true,
            ("yes", Value::Bool(true)) => yes = true,
            ("stats", Value::Bool(true)) => stats = true,
            ("limit", Value::Number(n)) => limit = Some(*n),
            _ => {}
        }
    }

    if download {
        println!("downlading");
        if force {
            println!("Forcing redownload of dataset...");
        }
        // download_dataset(force, limit);
        download_dataset(false, None);
        Some(values.to_vec())
    } else if init {
        // if !yes {
        //     println!("This will reset all annotations. Continue? [y/N]");
        //     let mut input = String::new();
        //     std::io::stdin().read_line(&mut input).unwrap();
        //     if !input.trim().eq_ignore_ascii_case("y") {
        //         return None;
        //     }
        // }
        // create_sample_annotations();
        // println!("Created new annotations file");
        Some(values.to_vec())
    } else if stats {
        // let annotations = Annotations::load();
        // println!("Dataset statistics:");
        // println!("Total images: {}", annotations.metadata.total_images);
        // println!("Attractive: {}", annotations.metadata.attractive_count);
        // println!(
        //     "Not attractive: {}",
        //     annotations.metadata.not_attractive_count
        // );
        Some(values.to_vec())
    } else {
        None
    }
}
