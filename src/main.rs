use anyhow::{anyhow, Result};
use glob::glob;
use tokenizers::{EncodeInput, Tokenizer};

fn exec(dst: &str) -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("Qwen/Qwen3-4B", None).map_err(|e| anyhow!(e))?;

    let mut text_list: Vec<EncodeInput> = Vec::new();
    let mut total_tokens = 0;

    for entry in glob(&format!("{}/**/*.md", dst)).unwrap() {
        let path = match entry {
            Ok(path) => path,
            Err(_) => continue
        };

        let r = std::fs::read_to_string(path)?;
        text_list.push(r.into());

        if text_list.len() == 200 {
            let out = tokenizer.encode_batch_fast(std::mem::take(&mut text_list), false).map_err(|e| anyhow!(e))?;

            for x in out {
                total_tokens += x.get_ids().len();
            }
        }
    }

    println!("Total tokens: {}", total_tokens);
    Ok(())
}

fn main() {
    let mut args = std::env::args();
    let dst = args.next().unwrap();

    exec(&dst).unwrap();
}
