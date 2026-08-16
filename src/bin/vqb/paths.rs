//! Where the harness reads and writes.
//!
//! Every output directory resolves in one place, so no command carries a path literal of
//! its own. Defaults are relative to the current directory; set them explicitly to work
//! from anywhere.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const DATA: &str = "data";
const RESULTS: &str = "results";
/// Under the results dir, so moving the results moves the store with it.
const CODES: &str = "codes";
const RAW: &str = "raw";
const HTML: &str = "html";
const DOCS: &str = "docs";

/// Where the harness writes, as the CLI states it. Each is `None` unless the user named
/// it, so the default only applies where nothing was asked for. Set them once in the
/// environment and the current directory stops mattering.
#[derive(clap::Args)]
pub struct Dirs {
    /// Directory holding downloaded datasets (default: `./data`).
    #[arg(long, global = true, env = "VQB_DATA_DIR")]
    data_dir: Option<PathBuf>,
    /// Directory for results JSON, `raw/` captures, and `html/` dashboards
    /// (default: `./results`).
    #[arg(long, global = true, env = "VQB_RESULTS_DIR")]
    results_dir: Option<PathBuf>,
    /// Directory for the per-method code stores (default: `<results-dir>/codes`). Point
    /// it at a roomy volume — one dataset's stores run to tens of GB.
    #[arg(long, global = true, env = "VQB_CODES_DIR")]
    codes_dir: Option<PathBuf>,
    /// Directory `publish` copies into and `index` scans, served as the site
    /// (default: `./docs/results`).
    #[arg(long, global = true, env = "VQB_PUBLISH_DIR")]
    publish_dir: Option<PathBuf>,
}

/// Resolved output roots: one per kind of artifact the harness emits.
pub struct Paths {
    data: PathBuf,
    results: PathBuf,
    codes: PathBuf,
    publish: PathBuf,
}

impl Paths {
    /// Resolve every root, letting each supplied override win over the default. Clap has
    /// already applied the flag-beats-environment half of the precedence.
    pub fn resolve(dirs: Dirs) -> Result<Self> {
        let cwd = std::env::current_dir().context("reading the current directory")?;
        Ok(Self::from_parts(
            cwd,
            dirs.data_dir,
            dirs.results_dir,
            dirs.codes_dir,
            dirs.publish_dir,
        ))
    }

    /// The precedence rules on their own, so they can be tested without a real current
    /// directory.
    fn from_parts(
        cwd: PathBuf,
        data: Option<PathBuf>,
        results: Option<PathBuf>,
        codes: Option<PathBuf>,
        publish: Option<PathBuf>,
    ) -> Self {
        let data = data.unwrap_or_else(|| cwd.join(DATA));
        let results = results.unwrap_or_else(|| cwd.join(RESULTS));
        let codes = codes.unwrap_or_else(|| results.join(CODES));
        let publish = publish.unwrap_or_else(|| cwd.join(DOCS).join(RESULTS));
        Self {
            data,
            results,
            codes,
            publish,
        }
    }

    /// Downloaded dataset files.
    pub fn data(&self) -> &Path {
        &self.data
    }

    /// Aggregated results JSON, and the parent of `raw` and `html`.
    pub fn results(&self) -> &Path {
        &self.results
    }

    /// Per-method code stores. Not necessarily under `results` — a large dataset's
    /// stores run to tens of GB and often belong on another volume.
    pub fn codes(&self) -> &Path {
        &self.codes
    }

    /// Raw per-run captures, which `vqb eval` reads back.
    pub fn raw(&self) -> PathBuf {
        self.results.join(RAW)
    }

    /// Rendered dashboards.
    pub fn html(&self) -> PathBuf {
        self.results.join(HTML)
    }

    /// The published site directory, whose tracked copy the site is served from.
    pub fn publish(&self) -> &Path {
        &self.publish
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_hang_off_the_current_directory() {
        let p = Paths::from_parts(PathBuf::from("/scratch"), None, None, None, None);
        assert_eq!(p.data(), Path::new("/scratch/data"));
        assert_eq!(p.results(), Path::new("/scratch/results"));
        assert_eq!(p.codes(), Path::new("/scratch/results/codes"));
        assert_eq!(p.raw(), Path::new("/scratch/results/raw"));
        assert_eq!(p.html(), Path::new("/scratch/results/html"));
        assert_eq!(p.publish(), Path::new("/scratch/docs/results"));
    }

    #[test]
    fn an_override_wins_over_the_default() {
        let p = Paths::from_parts(
            PathBuf::from("/repo"),
            Some(PathBuf::from("/mnt/data")),
            None,
            Some(PathBuf::from("/mnt/codes")),
            Some(PathBuf::from("/srv/site")),
        );
        assert_eq!(p.data(), Path::new("/mnt/data"));
        assert_eq!(p.codes(), Path::new("/mnt/codes"));
        assert_eq!(p.publish(), Path::new("/srv/site"));
        assert_eq!(p.results(), Path::new("/repo/results"), "left alone");
    }

    /// `publish` is the one root that does not follow `results` — the site is served
    /// from a tracked directory, so moving the run outputs must not move it.
    #[test]
    fn publish_is_independent_of_the_results_dir() {
        let p = Paths::from_parts(
            PathBuf::from("/repo"),
            None,
            Some(PathBuf::from("/mnt/run")),
            None,
            None,
        );
        assert_eq!(p.publish(), Path::new("/repo/docs/results"));
    }

    /// Moving the results dir takes the code store with it, unless the store is
    /// separately named — that pairing is what makes one flag move a whole working set.
    #[test]
    fn codes_default_under_an_overridden_results_dir() {
        let moved = Paths::from_parts(
            PathBuf::from("/repo"),
            None,
            Some(PathBuf::from("/mnt/run")),
            None,
            None,
        );
        assert_eq!(moved.codes(), Path::new("/mnt/run/codes"));
        assert_eq!(moved.raw(), Path::new("/mnt/run/raw"));

        let split = Paths::from_parts(
            PathBuf::from("/repo"),
            None,
            Some(PathBuf::from("/mnt/run")),
            Some(PathBuf::from("/ssd/codes")),
            None,
        );
        assert_eq!(split.codes(), Path::new("/ssd/codes"));
    }
}
