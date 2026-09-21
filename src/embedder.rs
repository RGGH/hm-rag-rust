// embedder.rs
use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// `fastembed` ships BGE-small-en-v1.5 as its default model, so the
/// vectors it produces live in the same 384-dim space as the ones already
/// stored in the `hm_products` Qdrant collection.
/// 
/// Note: the stored HF H&M `dense_embedding` vectors are L2-normalized
/// (confirmed empirically). fastembed's `embed()` also always L2-normalizes
/// its output, so no explicit normalization step is needed here — the two
/// sides already agree.
/// 
pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// Downloads (first run only, cached after) and loads the model.
    pub fn new() -> Result<Self> {
        let start = std::time::Instant::now();

        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
        )?;

        println!("⏱️ BGE model init: {:.3}s", start.elapsed().as_secs_f64());

        Ok(Self { model })
    }

    /// Embed a single query string.

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        let mut embeddings = self.model.embed(vec![text.to_string()], None)?;

        println!("⏱️ BGE embedding: {:.3}s", start.elapsed().as_secs_f64());

        Ok(embeddings.remove(0))
    }
}
