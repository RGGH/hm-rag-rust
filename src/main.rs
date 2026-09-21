// main.rs
mod embedder;
mod qdrant_store;
mod rag;

use clap::Parser;
use embedder::Embedder;
use qdrant_store::{Product, Store};

/// Query the H&M Qdrant collection and ask a local LLM about the results.
///
/// Rust port of the `qdrant-boosted-search` Python project
/// (https://github.com/RGGH/qdrant-boosted-search), retrieval logic only -
/// no GUI.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// The search / question text, e.g. "summer dress for a wedding"
    query: String,

    /// Weight applied to each product's `popularity` payload field
    #[arg(long, default_value_t = 0.05)]
    popularity_weight: f32,

    /// Weight applied to each product's `recency` payload field
    #[arg(long, default_value_t = 0.05)]
    recency_weight: f32,

    /// Number of products to retrieve
    #[arg(long, default_value_t = 10)]
    limit: u64,

    /// Qdrant gRPC URL (note: port 6334, not the 6333 REST port)
    #[arg(long, default_value = "http://localhost:6334")]
    qdrant_url: String,

    /// Ollama model to use for the final answer
    #[arg(long, default_value = "granite4.1:3b")]
    model: String,

    /// Skip the LLM step and just print the boosted search results
    #[arg(long, default_value_t = false)]
    no_llm: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let start = std::time::Instant::now();

    println!("🔎 Query: \"{}\"", args.query);

    // 1. Embed the query (BGE-small-en-v1.5, 384-dim - same model/space as
    //    the indexed products).
    let embedder = Embedder::new()?;
    let query_vector = embedder.embed_query(&args.query)?;

    // 2. Popularity/recency-boosted Qdrant search (formula query).
    let store = Store::connect(&args.qdrant_url)?;

    let points = store
        .boosted_search(
            query_vector,
            args.popularity_weight,
            args.recency_weight,
            args.limit,
        )
        .await?;

    // Use `from_boosted` (not the plain `From<&ScoredPoint>`) so each
    // Product also carries `raw_similarity`, backed out of the boosted
    // score using the same weights passed to `boosted_search`:
    //   raw_similarity = score - popularity_weight*popularity - recency_weight*recency
    let products: Vec<Product> = points
        .iter()
        .map(|p| Product::from_boosted(p, args.popularity_weight, args.recency_weight))
        .collect();
    print_results(&products);

    if args.no_llm {
        println!("\n⏱️ Time taken: {:.2?}", start.elapsed());
        return Ok(());
    }

    // 3. Feed the retrieved products to an LLM (via Rig + Ollama) as
    //    grounding context, then ask the original question.
    let context = rag::build_context(&products);
    let answer =
        rag::answer_with_context(&args.model, &args.query, &context).await?;

    println!("\n🤖 Answer:\n{answer}");
    println!("\n⏱️ Time taken: {:.2?}", start.elapsed());

    Ok(())
}

fn print_results(products: &[Product]) {
    println!(
        "\n{:<4}{:<14}{:<32}{:<10}{:<10}{:<12}{:<10}",
        "Rank", "ID", "Product", "Score", "Boosted", "Popularity", "Recency"
    );
    println!("{}", "-".repeat(92));

    for (i, p) in products.iter().enumerate() {
        let name: String = p.name.chars().take(30).collect();
        let id: String = p.id.chars().take(12).collect();
        let raw_sim = p
            .raw_similarity
            .map(|s| format!("{:.4}", s))
            .unwrap_or_else(|| "-".into());

        println!(
            "{:<4}{:<14}{:<32}{:<10}{:<10.4}{:<12.3}{:<10.3}",
            i + 1,
            id,
            name,
            raw_sim,
            p.score,
            p.popularity,
            p.recency
        );
    }
}