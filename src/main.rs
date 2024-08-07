use std::fs::{create_dir_all, File};
use std::env::{current_dir};
use std::error::Error;
use std::io;
use std::io::{Stdout, Write};
use std::path::{PathBuf};
use std::process::Stdio;
use std::sync::{Arc};
use futures_util::StreamExt;

use clap::{CommandFactory, Parser};
use clap::error::ErrorKind;
use futures_util::future::join_all;
use reqwest::{Url, Client, ClientBuilder, get};
use tokio::process::Command;
use tokio::io::AsyncWriteExt;
use tokio::join;
use tokio::sync::Mutex;
use tokio::sync::Barrier;
use indicatif::{ProgressBar, ProgressStyle};

const DEFAULT_ENDPOINT: &str = "https://hf-mirror.com/";
const DEFAULT_PROXY: &str = "https://hg.whl.moe/";

const ORIGIN_ENDPOINT: &str = "https://huggingface.co/";

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// HuggingFace Dataset Or Model to Download, use format like `google/gemma-2-2b-it`
    repo_id: String,

    /// Local directory path where the model or dataset will be stored, default is `pwd`. Note that a folder named model or dataset will be created, such as `<your_dir>/gemma-2-2b-it`.
    #[arg(short, long, value_name = "PATH")]
    local_dir: Option<PathBuf>,

    /// HuggingFace Endpoint, default is https://hf-mirror.com/
    #[arg(short, long, value_name = "URL")]
    endpoint_url: Option<String>,

    /// Large file proxy url, default is https://hg.whl.moe/
    #[arg(short, long, value_name = "URL")]
    proxy_url: Option<String>,

    /// Flag to specify a string pattern to include files for downloading.
    #[arg(long, value_name = "PATTERN")]
    include: Option<String>,

    /// Flag to specify a string pattern to exclude files from downloading.
    /// include/exclude_pattern The pattern to match against filenames, supports wildcard characters. e.g., '--exclude *.safetensor', '--include vae/*'.
    #[arg(long, value_name = "PATTERN")]
    exclude: Option<String>,

    ///Hugging Face username for authentication. **NOT EMAIL**.
    #[arg(long)]
    hf_username: Option<String>,

    ///Hugging Face token for authentication.
    #[arg(long)]
    hf_token: Option<String>,
}


async fn check_args(cli: Cli) -> Result<(Url, Url, PathBuf, String), Box<dyn std::error::Error>> {
    let endpoint_url;
    let proxy_url;
    let file_path;
    let save_path;


    let splits: Vec<&str> = cli.repo_id.trim().split("/").collect();

    // println!("{splits:?}");

    // let y = &splits[..];

    if let [author, item, ..] = splits[..] {
        println!("Parsing {author}:{item}...");


        let endpoint = (cli.endpoint_url
            .unwrap_or(DEFAULT_ENDPOINT.to_string()
            ));
        let proxy = (cli.proxy_url
            .unwrap_or(DEFAULT_PROXY.to_string()
            ));
        file_path = format!("{author}/{item}");

        endpoint_url = Url::parse(&
        if endpoint.ends_with("/") {
            endpoint
        } else {
            endpoint + "/"
        }
        ).unwrap_or_else(|e| {
            let mut cmd = Cli::command();
            cmd.error(
                ErrorKind::InvalidValue,
                format!("Error while parse url: {}", e),
            )
                .exit()
        })
            .join(&(file_path.clone() + "/"))?

        ;

        proxy_url = Url::parse(&
        if proxy.ends_with("/") {
            proxy
        } else {
            proxy + "/"
        }
        ).unwrap_or_else(|e| {
            let mut cmd = Cli::command();
            cmd.error(
                ErrorKind::InvalidValue,
                format!("Error while parse url: {}", e),
            )
                .exit()
        });


        println!("Target url is {}, proxy url is {}", endpoint_url.to_string(), proxy_url.to_string());
        println!("Checking endpoint url...");
        if !(check_url_status(&endpoint_url)
            .await?
        ) {
            let mut cmd = Cli::command();
            cmd.error(
                ErrorKind::ValueValidation,
                format!("{} not return 200, please check network!", endpoint_url.to_string()),
            )
                .exit();
        }

        println!("Checking proxy url...");
        if !(check_url_status(&proxy_url)
            .await?
        ) {
            let mut cmd = Cli::command();
            cmd.error(
                ErrorKind::ValueValidation,
                format!("{} not return 200, please check network!", proxy_url.to_string()),
            )
                .exit();
        }

        save_path = cli.local_dir
            .unwrap_or(current_dir().unwrap())
            .join(item);

        if !save_path.exists() {
            println!("Path {} does not exist. Creating it now.", save_path.to_str().unwrap());
            let _ = create_dir_all(&save_path).unwrap_or_else(|e| {
                let mut cmd = Cli::command();
                cmd.error(
                    ErrorKind::InvalidValue,
                    format!("Error creating path: {}", e),
                )
                    .exit()
            });
            println!("Path created successfully.");
        } else {
            println!("Path {} already exists.", save_path.to_str().unwrap());
        }
    } else {
        let mut cmd = Cli::command();
        cmd.error(
            ErrorKind::InvalidValue,
            format!("{} is not a valid repo id!", cli.repo_id),
        )
            .exit();
    }
    Ok((endpoint_url, proxy_url, save_path, file_path))
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let (endpoint, proxy, save_path, file_path): (Url, Url, PathBuf, String) = check_args(cli).await?;

    println!("Check git and lfs...");
    check_command_exists("git").await;
    check_command_exists("git-lfs").await;
    check_repo_authority(&endpoint, None, None).await.expect("Check authority fail!");

    let ret = String::from_utf8(if save_path
        .join(".git")
        .exists()
    {
        println!("Executing `git pull`...");
        Command::new(r"git")
            .current_dir(&save_path)
            .env("GIT_LFS_SKIP_SMUDGE", "1")
            .arg("pull")
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .output()
            .await
            .expect("git pull fail!")
            .stderr
    } else {
        println!("Executing `git clone {}`...", endpoint.to_string());
        Command::new(r"git")
            .env("GIT_LFS_SKIP_SMUDGE", "1")
            .arg("clone")
            .arg(endpoint.to_string())
            .arg(save_path.to_str().expect("Save path is not a Valid utf8 path"))
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .output()
            .await
            .expect("git clone fail!")
            .stderr
    })?;
    println!("{ret}");
    let output = Command::new("git")
        .current_dir(&save_path)
        .env("GIT_LFS_SKIP_SMUDGE", "1")
        .arg("lfs")
        .arg("ls-files")
        .output()
        .await?;

    let lfs = String::from_utf8(output.stdout)
        .expect("Read utf8 output fail!");

    println!("{lfs}");

    let lfs_vec: Vec<_> = lfs.lines().collect();
    let files_count = lfs_vec.len();
    let bar = Arc::new(indicatif::MultiProgress::with_draw_target(
        indicatif::ProgressDrawTarget::stderr_with_hz(5)
    ));

    let tasks:Vec<_> = lfs_vec.iter().enumerate().map(|(i, line)| {
        let file_name = line
            .split_once("-")
            .expect(&format!("Cant parse lfs list:{line}"))
            .1
            .trim()
            .to_string()
            ;
        let proxy = proxy.clone();
        let endpoint = endpoint.clone();
        let save_path = save_path.clone();
        let bar = Arc::clone(&bar);
        let url = format!("{}{}{}/resolve/main/{}",
                          proxy.to_string(),
                          ORIGIN_ENDPOINT,
                          endpoint.path()
                              .strip_prefix("/")
                              .unwrap()
                              .strip_suffix("/")
                              .unwrap()
                          ,
                          file_name
        );
        tokio::spawn(async move {
            download_files(&url, &save_path.join(file_name), i, files_count, bar)
                .await
                .expect("Download fail...");
        })
    }).collect();


    for task in tasks {
        task.await.unwrap();
    }
    Ok(())
}


async fn download_files(url: &str, path: &PathBuf, task_count: usize, total_task: usize, bar_m: Arc<indicatif::MultiProgress>) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = tokio::fs::File::create(path).await?;
    let resp = get(url).await?;

    if !resp.status().is_success() {
        println!("Cant download {} with status {}", url, resp.status().to_string());
        return Ok(());
    }

    let total_bytes: u64 = resp.content_length().unwrap_or(10485760);
    let mut count_bytes: f64 = 0.;

    let mut stream = resp.bytes_stream();
    let bar = bar_m.add(ProgressBar::new(total_bytes));
    bar.set_style(ProgressStyle::with_template( &(format!("{}",path.file_name().unwrap().to_str().unwrap()) +" {bar:70.green/red} {binary_bytes:>7}/{binary_total_bytes:7} {bytes_per_sec} [{elapsed_precise}/{eta_precise}] {msg}"))
        .unwrap()
        );


    println!("\r[{task_count}] Start downloading {url}...");
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        file.write_all(&chunk).await?;
        count_bytes += chunk.len() as f64;
        //进度条？
        bar.inc(chunk.len() as u64);

    }

    file.flush().await?;

    println!("[{task_count}] Downloaded {}", url);
    Ok(())
}

async fn check_url_status(url: &Url) -> Result<bool, Box<dyn std::error::Error>> {
    let success = get(
        url.clone()
    )
        .await?
        .status()
        .is_success();
    Ok(success)
}
async fn check_repo_authority(endpoint: &Url, hf_name: Option<String>, hf_token: Option<String>) -> Result<bool, Box<dyn std::error::Error>> {
    let ref_url = endpoint.join("info/refs?service=git-upload-pack").unwrap();
    Ok(check_url_status(&ref_url).await.expect(&format!("Cant authority target repo {}", ref_url.to_string())))
}

async fn check_command_exists(command: &str) -> bool {
    let check_command = if cfg!(target_os = "windows") {
        Command::new(r"C:/Windows/system32/where.exe")
            .arg(command)
            .output()
            .await
            .expect(&format!("{command} not exist!"))
    } else {
        Command::new("which")
            .arg(command)
            .output()
            .await
            .expect(&format!("{command} not exist!"))
    };


    check_command.status.success()
}

fn save(text: &str, file_path: &str) {
    // let file_path = "output.txt";

    // 创建并打开文件
    let mut file = match File::create(file_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error creating file: {}", e);
            return;
        }
    };

    // 将字符串写入文件
    match file.write_all(text.as_bytes()) {
        Ok(_) => println!("Text successfully written to {}", file_path),
        Err(e) => eprintln!("Error writing to file: {}", e),
    }
}

#[test]
fn verify_cli() {
    use clap::CommandFactory;
    Cli::command().debug_assert();
}

#[tokio::test]
async fn test_download() {
    let urls = vec![
        "https://speed.cloudflare.com/__down?during=download&bytes=10485760",
    ];
    // let barrier = Arc::new(Barrier::new(urls.len()));
    let bar = Arc::new(indicatif::MultiProgress::new());

    let tasks: Vec<_> = urls.iter().enumerate().map(
        |(i, url)| {
            let url = url.to_string();
            let bar = Arc::clone(&bar);
            tokio::spawn(async move {
                download_files(
                    &url,
                    &PathBuf::from(&format!("./tmp_{i}")),
                    i,
                    5,
                    bar,
                ).await.unwrap();
            })
        }
    ).collect();
    for task in tasks {
        task.await.unwrap();
    }
    println!("All tasks completed");
}