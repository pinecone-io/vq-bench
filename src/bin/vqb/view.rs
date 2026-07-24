//! `vqb view`: render a results JSON into a standalone HTML by injecting it into
//! the site template's embedded-run slot (the same page vq-bench.com serves).

use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};

/// The site dashboard. Its `__EMBED__` slot, when filled with a run's JSON, makes
/// the page render that single run instead of fetching the published manifest.
const TEMPLATE: &str = include_str!("../../../docs/index.html");
const SLOT: &str = "__EMBED__";

/// Where generated dashboards land by default (a gitignored working dir).
const HTML_DIR: &str = "results/html";

/// Render `results` to a standalone HTML and, unless `no_open`, open it in the
/// default browser. Output defaults to `results/html/<stem>.html`.
pub fn write(results: &Path, out: Option<&Path>, no_open: bool) -> Result<()> {
    if !results.is_file() {
        bail!("no such results file: {}", results.display());
    }
    if !TEMPLATE.contains(SLOT) {
        bail!("template docs/index.html is missing the {SLOT} slot");
    }
    let json =
        std::fs::read_to_string(results).with_context(|| format!("reading {}", results.display()))?;
    // Break any literal `</` (e.g. inside a method label) so it can't close the
    // <script> block early; `<\/` is an equivalent escape inside JSON.
    let embedded = json.replace("</", "<\\/");
    let html = TEMPLATE.replace(SLOT, &embedded);

    let out_path = match out {
        Some(p) => p.to_path_buf(),
        None => {
            std::fs::create_dir_all(HTML_DIR).context("create results/html")?;
            let stem = results.file_stem().and_then(|s| s.to_str()).unwrap_or("run");
            Path::new(HTML_DIR).join(format!("{stem}.html"))
        }
    };
    std::fs::write(&out_path, html).with_context(|| format!("writing {}", out_path.display()))?;
    println!("wrote {}", out_path.display());

    if !no_open {
        open_in_browser(&out_path);
    }
    Ok(())
}

/// Open a path in the default browser. Best-effort: a failure to launch (e.g. a
/// headless box) is reported but not fatal — the HTML is already written.
fn open_in_browser(path: &Path) {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };
    match Command::new(opener).arg(path).spawn() {
        Ok(_) => {}
        Err(e) => eprintln!("could not open a browser ({opener}: {e}); open {} manually", path.display()),
    }
}
