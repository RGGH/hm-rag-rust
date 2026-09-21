// rag.rs
use crate::qdrant_store::Product;
use anyhow::Result;
// `rig::prelude::*` brings in the `CompletionClient` / `AgentClientExt`
// traits needed for `.completion_model()` / `.agent()`, plus `Prompt` for
// `.prompt()`. `Nothing` is Rig's "no credentials needed" marker type,
// used here because a local Ollama instance needs no API key.
use rig::client::Nothing;
use rig::prelude::*;
use rig::providers::ollama;

/// Render the retrieved products into a compact context block for the LLM.
/// Keeping this as plain text (rather than JSON) keeps the prompt short and
/// easy for the model to read - similar in spirit to how `boosted_search.py`
/// prints a results table for humans.
pub fn build_context(products: &[Product]) -> String {
    let mut context = String::new();

    for (i, p) in products.iter().enumerate() {
        context.push_str(&format!(
            "{}. {} | type: {} | colour: {} | popularity: {:.2} | recency: {:.2}\n",
            i + 1,
            p.name,
            p.product_type,
            p.colour,
            p.popularity,
            p.recency,
        ));

        if !p.detail_desc.is_empty() {
            context.push_str(&format!("   description: {}\n", p.detail_desc));
        }
    }

    context
}

/// Ask a local Ollama model to answer a shopping question, grounded in the
/// products retrieved from Qdrant.
///
/// Structured like Rig's `rag_ollama` example
/// (https://github.com/0xPlaygrounds/rig/blob/main/examples/rag_ollama/src/main.rs),
/// except the retrieval step is our own Qdrant call (with the popularity /
/// recency boosting formula) rather than Rig's built-in in-memory vector
/// store, so we inject the retrieved context straight into the preamble
/// instead of using `.dynamic_context(...)`.
///
/// Uses `rig::providers::ollama::Client` (the stable, published API - see
/// https://docs.rs/rig-core/0.38.1/src/rig_core/providers/ollama.rs.html),
/// NOT `rig::providers::ollama::wire`, which only exists on Rig's
/// unreleased dev branch.
pub async fn answer_with_context(model: &str, question: &str, context: &str) -> Result<String> {
    // `Nothing` = no API key, since a local Ollama instance needs none.
    // Defaults to http://localhost:11434; use `ollama::Client::builder()`
    // instead if you need a custom host/port or an API key.
    let client = ollama::Client::new(Nothing)?;

    let preamble = format!(
        "You are a helpful H&M shopping assistant. \
         Answer the user's question using ONLY the product listings below. \
         If the listings don't contain a good answer, say so instead of \
         making one up.\n\n\
         PRODUCTS:\n{context}"
    );

    let agent = client.agent(model).preamble(&preamble).build();

    let response = agent.prompt(question).await?;

    // `response`'s exact type differs subtly across rig-core versions
    // (plain `String` vs. a wrapper struct), but it always implements
    // `Display` (per Rig's own doctest: `println!("{response}")`), so
    // formatting it is the version-agnostic way to get the answer text.
    Ok(format!("{response}"))
}