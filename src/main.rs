use glob::glob;
use tokenizers::{tokenizer, EncodeInput, Tokenizer};
use anyhow::{anyhow, Result};

fn exec(dst: &str) -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("Qwen/Qwen3-4B", None).map_err(|e| anyhow!(e))?;

    let mut text_list: Vec<EncodeInput> = Vec::new();

    for entry in glob(&format!("{}/**/*.md", dst)).unwrap().filter_map(Result::ok) {
        let r = std::fs::read_to_string(entry)?;
        text_list.push(r.into());
    }

    let out = tokenizer.encode_batch_fast(text_list, false)?;
    let mut total_tokens = 0;

    for x in out {
        total_tokens += x.get_ids().len();
    }
    println!("Total tokens: {}", total_tokens);
    Ok(())
}

fn main() {
    let mut args = std::env::args();
    let dst = args.next().unwrap();

    exec(&dst).unwrap();
}
