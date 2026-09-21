// qdrant_store.rs
use anyhow::Result;
use qdrant_client::qdrant::{
    point_id::PointIdOptions, value::Kind, Expression, FormulaBuilder, PrefetchQueryBuilder,
    QueryPointsBuilder, ScoredPoint,
};
use qdrant_client::Qdrant;
use std::collections::HashMap;

/// Same collection the Python project reads/writes:
/// `src/qdrant.py` / `src/boosted_search.py` -> COLLECTION_NAME = "hm_products"
pub const COLLECTION_NAME: &str = "hm_products";

/// The named dense vector the products were indexed under (`using="dense"`
/// in every Python query).
const DENSE_VECTOR: &str = "dense";

/// How many candidates to pull from the dense prefetch before applying the
/// popularity/recency formula. Matches the `limit=100` in `boosted_search.py`.
const PREFETCH_LIMIT: u64 = 100;

pub struct Store {
    client: Qdrant,
}

/// A lightly-typed view over a Qdrant `ScoredPoint`'s payload, matching the
/// fields the Python code reads in `app.py` / `search.py`.
#[derive(Debug, Clone, Default)]
pub struct Product {
    pub id: String,
    pub name: String,
    pub product_type: String,
    pub colour: String,
    pub detail_desc: String,
    /// The final `boosted_search` score: semantic + w1*popularity + w2*recency.
    pub score: f32,
    pub popularity: f64,
    pub recency: f64,
    /// The original (pre-boost) cosine similarity, backed out of `score`
    /// via `score - w1*popularity - w2*recency`. Only populated when the
    /// product came from `boosted_search` via `from_boosted` - `None` for
    /// products built from plain `semantic_search` results, where `score`
    /// already *is* the raw similarity.
    pub raw_similarity: Option<f32>,
}

/// One row of the semantic-vs-boosted comparison: a product's boosted
/// rank/score alongside where (if anywhere) it sat in the plain semantic
/// results for the same query.
#[derive(Debug, Clone)]
pub struct ComparisonRow {
    pub boosted_rank: usize,
    pub boosted_score: f32,
    pub product: Product,
    /// `None` means it didn't appear in the semantic results at all
    /// (i.e. fell outside `semantic_limit`).
    pub semantic_rank: Option<usize>,
    pub semantic_score: Option<f32>,
}

impl Store {
    /// Connect to Qdrant's gRPC endpoint (port 6334 by default).
    ///
    /// The Python project uses `QdrantClient("http://localhost:6333")`
    /// (the REST port) - the Rust client talks gRPC instead, so point this
    /// at 6334 unless you've reconfigured your Qdrant instance.
    pub fn connect(url: &str) -> Result<Self> {
        let client = Qdrant::from_url(url).build()?;
        Ok(Self { client })
    }

    /// Plain semantic search - equivalent to `semantic_search()` /
    /// `search_products()` in the Python code.
    pub async fn semantic_search(&self, query_vector: Vec<f32>, limit: u64) -> Result<Vec<ScoredPoint>> {
        let response = self
            .client
            .query(
                QueryPointsBuilder::new(COLLECTION_NAME)
                    .query(query_vector)
                    .using(DENSE_VECTOR)
                    .limit(limit)
                    .with_payload(true),
            )
            .await?;

        Ok(response.result)
    }

    /// Popularity/recency boosted search - a direct port of
    /// `boosted_search()` in `src/boosted_search.py`:
    ///
    /// ```python
    /// query=models.FormulaQuery(
    ///     formula=models.SumExpression(
    ///         sum=[
    ///             "$score",
    ///             models.MultExpression(mult=[popularity_weight, "popularity"]),
    ///             models.MultExpression(mult=[recency_weight, "recency"]),
    ///         ]
    ///     ),
    ///     defaults={"popularity": 0.0, "recency": 0.0},
    /// )
    /// ```
    pub async fn boosted_search(
        &self,
        query_vector: Vec<f32>,
        popularity_weight: f32,
        recency_weight: f32,
        limit: u64,
    ) -> Result<Vec<ScoredPoint>> {
        let formula = FormulaBuilder::new(Expression::sum_with([
            Expression::score(),
            Expression::mult_with([
                Expression::constant(popularity_weight),
                Expression::variable("popularity"),
            ]),
            Expression::mult_with([
                Expression::constant(recency_weight),
                Expression::variable("recency"),
            ]),
        ]))
        .add_default("popularity", 0.0)
        .add_default("recency", 0.0);

        let response = self
            .client
            .query(
                QueryPointsBuilder::new(COLLECTION_NAME)
                    .add_prefetch(
                        PrefetchQueryBuilder::default()
                            .query(query_vector)
                            .using(DENSE_VECTOR)
                            .limit(PREFETCH_LIMIT),
                    )
                    .query(formula)
                    .limit(limit)
                    .with_payload(true),
            )
            .await?;

        Ok(response.result)
    }

    /// Same as `boosted_search`, but returns `Product`s with
    /// `raw_similarity` already backed out - use this when you just want
    /// the original score in your printout without a second query.
    pub async fn boosted_search_with_products(
        &self,
        query_vector: Vec<f32>,
        popularity_weight: f32,
        recency_weight: f32,
        limit: u64,
    ) -> Result<Vec<Product>> {
        let points = self
            .boosted_search(query_vector, popularity_weight, recency_weight, limit)
            .await?;

        Ok(points
            .iter()
            .map(|p| Product::from_boosted(p, popularity_weight, recency_weight))
            .collect())
    }

    /// Runs both `semantic_search` and `boosted_search` on the same query
    /// vector and lines them up by product id, so you can see exactly how
    /// much the popularity/recency formula moved each result.
    ///
    /// `semantic_limit` is pulled independently of `limit` (the boosted
    /// page size) - set it to at least `PREFETCH_LIMIT` (100) so it covers
    /// the same candidate pool the formula itself reranked, otherwise a
    /// boosted top-10 item that was semantically rank ~50 will show up as
    /// "not in results" even though it was in-pool.
    pub async fn compare_search(
        &self,
        query_vector: Vec<f32>,
        popularity_weight: f32,
        recency_weight: f32,
        limit: u64,
        semantic_limit: u64,
    ) -> Result<Vec<ComparisonRow>> {
        let semantic = self
            .semantic_search(query_vector.clone(), semantic_limit)
            .await?;
        let boosted = self
            .boosted_search(query_vector, popularity_weight, recency_weight, limit)
            .await?;

        // id -> (1-based rank, raw semantic score)
        let semantic_index: HashMap<String, (usize, f32)> = semantic
            .iter()
            .enumerate()
            .filter_map(|(i, p)| point_id_key(p).map(|id| (id, (i + 1, p.score))))
            .collect();

        let rows = boosted
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let (semantic_rank, semantic_score) = point_id_key(p)
                    .and_then(|id| semantic_index.get(&id))
                    .map(|&(rank, score)| (Some(rank), Some(score)))
                    .unwrap_or((None, None));

                ComparisonRow {
                    boosted_rank: i + 1,
                    boosted_score: p.score,
                    product: Product::from_boosted(p, popularity_weight, recency_weight),
                    semantic_rank,
                    semantic_score,
                }
            })
            .collect();

        Ok(rows)
    }
}

/// Pretty-print `boosted_search_with_products` results in the same shape
/// as your original `RankProduct / Score / Popularity / Recency` table,
/// with the original (pre-boost) score and the final boosted score shown
/// as separate columns.
pub fn print_products(products: &[Product]) {
    println!(
        "\n{:<3} {:<12} {:<28} {:>9} {:>9} {:>11} {:>9}",
        "Rk", "ID", "Product", "Score", "Boosted", "Popularity", "Recency"
    );
    println!("{}", "-".repeat(90));

    for (i, p) in products.iter().enumerate() {
        let raw_sim = p
            .raw_similarity
            .map(|s| format!("{:.4}", s))
            .unwrap_or_else(|| "-".into());

        println!(
            "{:<3} {:<12} {:<28} {:>9} {:>9.4} {:>11.3} {:>9.3}",
            i + 1,
            truncate(&p.id, 12),
            truncate(&p.name, 28),
            raw_sim,
            p.score,
            p.popularity,
            p.recency
        );
    }
}

/// Pretty-print a `compare_search` result the same shape as your existing
/// `RankProduct / Score / Popularity / Recency` table, with two extra
/// columns showing where each boosted result sat pre-boost.
pub fn print_comparison(rows: &[ComparisonRow]) {
    println!(
        "\n{:<3} {:<28} {:>9} {:>9} {:>9} {:>10} {:>12}",
        "Rk", "Product", "Boosted", "SemScore", "Δ", "SemRank", "SemΔRank"
    );
    println!("{}", "-".repeat(88));

    for row in rows {
        let name = truncate(&row.product.name, 28);

        let (sem_score_s, delta_s) = match row.semantic_score {
            Some(s) => (format!("{:.4}", s), format!("{:+.4}", row.boosted_score - s)),
            None => ("-".into(), "-".into()),
        };

        let (sem_rank_s, rank_delta_s) = match row.semantic_rank {
            Some(r) => (
                r.to_string(),
                format!("{:+}", row.boosted_rank as i64 - r as i64),
            ),
            None => ("not ranked".into(), "-".into()),
        };

        println!(
            "{:<3} {:<28} {:>9.4} {:>9} {:>9} {:>10} {:>12}",
            row.boosted_rank, name, row.boosted_score, sem_score_s, delta_s, sem_rank_s, rank_delta_s
        );
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        s.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
    } else {
        s.to_string()
    }
}

/// Extracts a stable string key from a `ScoredPoint`'s id so results from
/// two separate queries can be matched up. Qdrant point ids are either
/// numeric or UUID - handle both.
fn point_id_key(point: &ScoredPoint) -> Option<String> {
    let options = point.id.as_ref()?.point_id_options.as_ref()?;
    Some(match options {
        PointIdOptions::Num(n) => n.to_string(),
        PointIdOptions::Uuid(u) => u.clone(),
    })
}

/// Turn a raw `ScoredPoint` into a `Product`, matching the payload fields
/// `app.py` reads (`prod_name`/`name`, `product_type`, `colour`,
/// `popularity`, `recency`, `detail_desc`).
///
/// Used for plain `semantic_search` results, where `point.score` already
/// *is* the raw similarity - `raw_similarity` is left `None` here since
/// `score` covers that case directly.
impl From<&ScoredPoint> for Product {
    fn from(point: &ScoredPoint) -> Self {
        let payload = &point.payload;

        Product {
            id: point_id_key(point).unwrap_or_else(|| "?".into()),
            name: payload_str(payload, "prod_name")
                .or_else(|| payload_str(payload, "name"))
                .unwrap_or_default(),
            product_type: payload_str(payload, "product_type")
                .or_else(|| payload_str(payload, "product_type_name"))
                .unwrap_or_default(),
            colour: payload_str(payload, "colour").unwrap_or_default(),
            detail_desc: payload_str(payload, "detail_desc").unwrap_or_default(),
            score: point.score,
            popularity: payload_f64(payload, "popularity").unwrap_or(0.0),
            recency: payload_f64(payload, "recency").unwrap_or(0.0),
            raw_similarity: None,
        }
    }
}

impl Product {
    /// Build a `Product` from a `boosted_search` result, backing out the
    /// original (pre-boost) similarity from the same formula Qdrant used
    /// to compute `score`:
    ///
    /// `score = raw_similarity + popularity_weight*popularity + recency_weight*recency`
    ///
    /// so:
    ///
    /// `raw_similarity = score - popularity_weight*popularity - recency_weight*recency`
    ///
    /// This needs no extra query - it's just inverting the sum formula
    /// with the same weights you passed into `boosted_search`.
    pub fn from_boosted(point: &ScoredPoint, popularity_weight: f32, recency_weight: f32) -> Self {
        let mut product = Product::from(point);
        let raw = product.score
            - popularity_weight * product.popularity as f32
            - recency_weight * product.recency as f32;
        product.raw_similarity = Some(raw);
        product
    }
}

fn payload_str(payload: &std::collections::HashMap<String, qdrant_client::qdrant::Value>, key: &str) -> Option<String> {
    match payload.get(key)?.kind.as_ref()? {
        Kind::StringValue(s) => Some(s.clone()),
        _ => None,
    }
}

fn payload_f64(payload: &std::collections::HashMap<String, qdrant_client::qdrant::Value>, key: &str) -> Option<f64> {
    match payload.get(key)?.kind.as_ref()? {
        Kind::DoubleValue(v) => Some(*v),
        Kind::IntegerValue(v) => Some(*v as f64),
        _ => None,
    }
}