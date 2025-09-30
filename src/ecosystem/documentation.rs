//! # Interactive Documentation System
//!
//! Comprehensive documentation, API references, and interactive help for ΨLang.
//! Includes searchable documentation, code references, and contextual help.

use crate::runtime::*;
use crate::stdlib::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Documentation library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Interactive Documentation System");
    Ok(())
}

/// Documentation System
pub struct DocumentationSystem {
    documents: HashMap<String, Document>,
    api_references: HashMap<String, ApiReference>,
    search_index: SearchIndex,
    examples: HashMap<String, String>,
    tutorials: HashMap<String, TutorialDoc>,
}

impl DocumentationSystem {
    /// Create a new documentation system
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            api_references: HashMap::new(),
            search_index: SearchIndex::new(),
            examples: HashMap::new(),
            tutorials: HashMap::new(),
        }
    }

    /// Add a document
    pub fn add_document(&mut self, document: Document) {
        self.documents.insert(document.title.clone(), document.clone());
        self.search_index.index_document(&document);
    }

    /// Add API reference
    pub fn add_api_reference(&mut self, reference: ApiReference) {
        self.api_references.insert(reference.name.clone(), reference.clone());
        self.search_index.index_api_reference(&reference);
    }

    /// Search documentation
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        self.search_index.search(query)
    }

    /// Get document by title
    pub fn get_document(&self, title: &str) -> Option<&Document> {
        self.documents.get(title)
    }

    /// Get API reference by name
    pub fn get_api_reference(&self, name: &str) -> Option<&ApiReference> {
        self.api_references.get(name)
    }

    /// Generate contextual help for code
    pub fn get_contextual_help(&self, code_snippet: &str, cursor_position: usize) -> ContextualHelp {
        let context = self.analyze_code_context(code_snippet, cursor_position);

        ContextualHelp {
            context_type: context.context_type,
            relevant_docs: self.find_relevant_docs(&context),
            examples: self.find_relevant_examples(&context),
            suggestions: self.generate_suggestions(&context),
        }
    }

    /// Analyze code context around cursor
    fn analyze_code_context(&self, code: &str, position: usize) -> CodeContext {
        // Simple context analysis - in practice would use proper parsing
        let before_cursor = &code[..position.min(code.len())];
        let after_cursor = &code[position..];

        // Determine context type based on keywords
        let context_type = if before_cursor.contains("neuron") {
            ContextType::NeuronDefinition
        } else if before_cursor.contains("synapse") {
            ContextType::SynapseDefinition
        } else if before_cursor.contains("topology") {
            ContextType::TopologyDefinition
        } else if before_cursor.contains("execute") {
            ContextType::Execution
        } else {
            ContextType::General
        };

        CodeContext {
            context_type,
            keywords: self.extract_keywords(before_cursor),
            position,
        }
    }

    /// Extract keywords from code
    fn extract_keywords(&self, code: &str) -> Vec<String> {
        code.split_whitespace()
            .filter(|word| {
                // Filter out common keywords and symbols
                !["the", "and", "or", "with", "for", "in", "on", "at", ":", "->", "=>"].contains(word)
            })
            .map(|s| s.to_string())
            .collect()
    }

    /// Find relevant documentation
    fn find_relevant_docs(&self, context: &CodeContext) -> Vec<Document> {
        let mut relevant = Vec::new();

        for keyword in &context.keywords {
            if let Some(results) = self.search_index.keyword_index.get(keyword) {
                for doc_title in results {
                    if let Some(doc) = self.documents.get(doc_title) {
                        relevant.push(doc.clone());
                    }
                }
            }
        }

        relevant
    }

    /// Find relevant examples
    fn find_relevant_examples(&self, context: &CodeContext) -> Vec<String> {
        let mut examples = Vec::new();

        for keyword in &context.keywords {
            if let Some(example_code) = self.examples.get(keyword) {
                examples.push(example_code.clone());
            }
        }

        examples
    }

    /// Generate code suggestions
    fn generate_suggestions(&self, context: &CodeContext) -> Vec<String> {
        match context.context_type {
            ContextType::NeuronDefinition => {
                vec![
                    "threshold: -50mV".to_string(),
                    "resting_potential: -70mV".to_string(),
                    "reset_potential: -80mV".to_string(),
                    "refractory_period: 2ms".to_string(),
                ]
            }
            ContextType::SynapseDefinition => {
                vec![
                    "⊸0.5:1ms⊸".to_string(),
                    "with STDP".to_string(),
                    "with Hebbian".to_string(),
                ]
            }
            ContextType::TopologyDefinition => {
                vec![
                    "∴ neuron_name { }".to_string(),
                    "neuron₁ ⊸ neuron₂".to_string(),
                    "assembly ⟪name⟫ { }".to_string(),
                ]
            }
            ContextType::Execution => {
                vec![
                    "execute network for 100ms".to_string(),
                    "simulate for 1000ms".to_string(),
                    "run until convergence".to_string(),
                ]
            }
            ContextType::General => {
                vec![
                    "topology ⟪name⟫ { }".to_string(),
                    "execute ⟪name⟫".to_string(),
                    "analyze patterns".to_string(),
                ]
            }
        }
    }
}

/// Document structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub title: String,
    pub content: String,
    pub category: DocCategory,
    pub tags: Vec<String>,
    pub difficulty: DifficultyLevel,
    pub last_updated: f64,
    pub author: String,
    pub version: String,
}

/// Document categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocCategory {
    Tutorial,
    Reference,
    Guide,
    API,
    Examples,
    FAQ,
    Troubleshooting,
}

/// API reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiReference {
    pub name: String,
    pub signature: String,
    pub description: String,
    pub parameters: Vec<Parameter>,
    pub return_type: String,
    pub examples: Vec<String>,
    pub related_topics: Vec<String>,
}

/// Parameter documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub required: bool,
    pub default_value: Option<String>,
}

/// Search index for fast document retrieval
#[derive(Debug, Clone)]
pub struct SearchIndex {
    keyword_index: HashMap<String, Vec<String>>, // keyword -> document titles
    title_index: HashMap<String, String>,       // title -> content
    tag_index: HashMap<String, Vec<String>>,    // tag -> document titles
}

impl SearchIndex {
    /// Create a new search index
    pub fn new() -> Self {
        Self {
            keyword_index: HashMap::new(),
            title_index: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }

    /// Index a document
    pub fn index_document(&mut self, document: &Document) {
        // Index by title
        self.title_index.insert(document.title.clone(), document.content.clone());

        // Index by keywords (simple word extraction)
        let keywords = self.extract_keywords(&document.content);
        for keyword in keywords {
            self.keyword_index.entry(keyword).or_insert_with(Vec::new).push(document.title.clone());
        }

        // Index by tags
        for tag in &document.tags {
            self.tag_index.entry(tag.clone()).or_insert_with(Vec::new).push(document.title.clone());
        }
    }

    /// Index API reference
    pub fn index_api_reference(&mut self, reference: &ApiReference) {
        // Index by name
        self.title_index.insert(reference.name.clone(), reference.description.clone());

        // Index by keywords in description
        let keywords = self.extract_keywords(&reference.description);
        for keyword in keywords {
            self.keyword_index.entry(keyword).or_insert_with(Vec::new).push(reference.name.clone());
        }
    }

    /// Search for documents
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        let mut results = Vec::new();
        let keywords = self.extract_keywords(query);

        for keyword in keywords {
            // Search keyword index
            if let Some(doc_titles) = self.keyword_index.get(&keyword) {
                for title in doc_titles {
                    if let Some(content) = self.title_index.get(title) {
                        results.push(SearchResult {
                            title: title.clone(),
                            content_preview: self.generate_preview(content),
                            relevance_score: self.calculate_relevance(&keyword, content),
                            result_type: SearchResultType::Document,
                        });
                    }
                }
            }

            // Search tag index
            if let Some(doc_titles) = self.tag_index.get(&keyword) {
                for title in doc_titles {
                    if let Some(content) = self.title_index.get(title) {
                        results.push(SearchResult {
                            title: title.clone(),
                            content_preview: self.generate_preview(content),
                            relevance_score: 0.8, // High relevance for tag matches
                            result_type: SearchResultType::Document,
                        });
                    }
                }
            }
        }

        // Remove duplicates and sort by relevance
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        results.dedup_by(|a, b| a.title == b.title);

        results
    }

    /// Extract keywords from text
    fn extract_keywords(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 3) // Filter short words
            .filter(|word| !self.is_stop_word(word))
            .map(|s| s.to_string())
            .collect()
    }

    /// Check if word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        let stop_words = ["the", "and", "or", "but", "with", "for", "from", "this", "that", "will"];
        stop_words.contains(&word)
    }

    /// Generate content preview
    fn generate_preview(&self, content: &str) -> String {
        if content.len() <= 200 {
            content.to_string()
        } else {
            format!("{}...", &content[..197])
        }
    }

    /// Calculate relevance score
    fn calculate_relevance(&self, keyword: &str, content: &str) -> f64 {
        let keyword_lower = keyword.to_lowercase();
        let content_lower = content.to_lowercase();

        let keyword_count = content_lower.matches(&keyword_lower).count();
        let total_words = content_lower.split_whitespace().count();

        if total_words == 0 {
            0.0
        } else {
            keyword_count as f64 / total_words as f64
        }
    }
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub content_preview: String,
    pub relevance_score: f64,
    pub result_type: SearchResultType,
}

/// Search result types
#[derive(Debug, Clone)]
pub enum SearchResultType {
    Document,
    APIReference,
    Example,
    Tutorial,
}

/// Code context analysis
#[derive(Debug, Clone)]
pub struct CodeContext {
    pub context_type: ContextType,
    pub keywords: Vec<String>,
    pub position: usize,
}

/// Context types
#[derive(Debug, Clone)]
pub enum ContextType {
    NeuronDefinition,
    SynapseDefinition,
    TopologyDefinition,
    Execution,
    General,
}

/// Contextual help
#[derive(Debug, Clone)]
pub struct ContextualHelp {
    pub context_type: ContextType,
    pub relevant_docs: Vec<Document>,
    pub examples: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Tutorial documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialDoc {
    pub title: String,
    pub description: String,
    pub sections: Vec<TutorialSection>,
    pub code_examples: Vec<String>,
    pub prerequisites: Vec<String>,
    pub learning_objectives: Vec<String>,
}

/// Tutorial section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialSection {
    pub title: String,
    pub content: String,
    pub code: String,
    pub interactive_elements: Vec<InteractiveElement>,
}

/// Interactive documentation element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveElement {
    pub element_type: InteractiveElementType,
    pub content: String,
    pub correct_answer: Option<String>,
    pub hints: Vec<String>,
}

/// Interactive element types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractiveElementType {
    MultipleChoice,
    CodeCompletion,
    DragAndDrop,
    Simulation,
}

/// Documentation Generator
pub struct DocumentationGenerator;

impl DocumentationGenerator {
    /// Generate API documentation from code
    pub fn generate_api_docs() -> Vec<ApiReference> {
        let mut docs = Vec::new();

        // Generate neuron API docs
        docs.push(ApiReference {
            name: "Neuron".to_string(),
            signature: "∴ neuron_name { parameters }".to_string(),
            description: "Creates a new neuron with specified parameters".to_string(),
            parameters: vec![
                Parameter {
                    name: "threshold".to_string(),
                    param_type: "Voltage".to_string(),
                    description: "Membrane potential threshold for spike generation".to_string(),
                    required: false,
                    default_value: Some("-50mV".to_string()),
                },
                Parameter {
                    name: "resting_potential".to_string(),
                    param_type: "Voltage".to_string(),
                    description: "Resting membrane potential".to_string(),
                    required: false,
                    default_value: Some("-70mV".to_string()),
                },
            ],
            return_type: "Neuron".to_string(),
            examples: vec![
                "∴ excitatory_neuron { threshold: -50mV, resting_potential: -70mV }".to_string(),
            ],
            related_topics: vec!["Synapse".to_string(), "Topology".to_string()],
        });

        // Generate synapse API docs
        docs.push(ApiReference {
            name: "Synapse".to_string(),
            signature: "neuron₁ ⊸weight:delay⊸ neuron₂".to_string(),
            description: "Creates a synapse between two neurons".to_string(),
            parameters: vec![
                Parameter {
                    name: "weight".to_string(),
                    param_type: "f64".to_string(),
                    description: "Synaptic weight (-1.0 to 1.0)".to_string(),
                    required: true,
                    default_value: None,
                },
                Parameter {
                    name: "delay".to_string(),
                    param_type: "Duration".to_string(),
                    description: "Synaptic transmission delay".to_string(),
                    required: false,
                    default_value: Some("1ms".to_string()),
                },
            ],
            return_type: "Synapse".to_string(),
            examples: vec![
                "input_neuron ⊸0.5:2ms⊸ output_neuron".to_string(),
                "excitatory ⊸0.8⊸ inhibitory with STDP".to_string(),
            ],
            related_topics: vec!["Neuron".to_string(), "Plasticity".to_string()],
        });

        docs
    }

    /// Generate tutorial documentation
    pub fn generate_tutorial_docs() -> Vec<TutorialDoc> {
        let mut tutorials = Vec::new();

        tutorials.push(TutorialDoc {
            title: "Getting Started with ΨLang".to_string(),
            description: "Learn the basics of neural network programming".to_string(),
            sections: vec![
                TutorialSection {
                    title: "Your First Neural Network".to_string(),
                    content: "Let's create your first neural network with ΨLang.".to_string(),
                    code: r#"topology ⟪first_network⟫ {
    ∴ neuron₁ { threshold: -50mV }
    ∴ neuron₂ { threshold: -55mV }
    neuron₁ ⊸0.5⊸ neuron₂
}

execute ⟪first_network⟫ for 100ms"#.to_string(),
                    interactive_elements: vec![
                        InteractiveElement {
                            element_type: InteractiveElementType::CodeCompletion,
                            content: "Complete the neuron definition".to_string(),
                            correct_answer: Some("∴ neuron₁ { threshold: -50mV }".to_string()),
                            hints: vec![
                                "Start with ∴ for neuron definition".to_string(),
                                "Include threshold parameter".to_string(),
                            ],
                        },
                    ],
                },
            ],
            code_examples: vec![
                "Basic neuron creation".to_string(),
                "Synapse connection".to_string(),
            ],
            prerequisites: vec!["Basic programming knowledge".to_string()],
            learning_objectives: vec![
                "Understand neuron and synapse creation".to_string(),
                "Learn basic network execution".to_string(),
            ],
        });

        tutorials
    }
}

/// Interactive Documentation Browser
pub struct DocumentationBrowser {
    current_document: Option<String>,
    navigation_history: Vec<String>,
    bookmarks: Vec<String>,
    search_results: Vec<SearchResult>,
    current_search_query: String,
}

impl DocumentationBrowser {
    /// Create a new documentation browser
    pub fn new() -> Self {
        Self {
            current_document: None,
            navigation_history: Vec::new(),
            bookmarks: Vec::new(),
            search_results: Vec::new(),
            current_search_query: String::new(),
        }
    }

    /// Navigate to document
    pub fn navigate_to(&mut self, document_title: String, doc_system: &DocumentationSystem) {
        if doc_system.get_document(&document_title).is_some() {
            if let Some(current) = &self.current_document {
                self.navigation_history.push(current.clone());
            }
            self.current_document = Some(document_title.clone());
        }
    }

    /// Go back in navigation history
    pub fn go_back(&mut self) -> Option<String> {
        self.navigation_history.pop()
    }

    /// Search documentation
    pub fn search(&mut self, query: String, doc_system: &DocumentationSystem) {
        self.search_results = doc_system.search(&query);
        self.current_search_query = query;
    }

    /// Add bookmark
    pub fn add_bookmark(&mut self, document_title: String) {
        if !self.bookmarks.contains(&document_title) {
            self.bookmarks.push(document_title);
        }
    }

    /// Get current document
    pub fn get_current_document(&self, doc_system: &DocumentationSystem) -> Option<&Document> {
        if let Some(title) = &self.current_document {
            doc_system.get_document(title)
        } else {
            None
        }
    }

    /// Get bookmarks
    pub fn get_bookmarks(&self) -> Vec<String> {
        self.bookmarks.clone()
    }

    /// Get search results
    pub fn get_search_results(&self) -> Vec<SearchResult> {
        self.search_results.clone()
    }
}

/// Documentation Export System
pub struct DocumentationExporter;

impl DocumentationExporter {
    /// Export documentation to HTML
    pub fn export_to_html(documents: &[Document], output_dir: &str) -> Result<(), String> {
        for document in documents {
            let html_content = Self::document_to_html(document);
            let filename = format!("{}/{}.html", output_dir, Self::sanitize_filename(&document.title));

            std::fs::write(&filename, html_content)
                .map_err(|e| format!("Failed to write HTML file: {}", e))?;
        }

        // Generate index page
        let index_html = Self::generate_index_html(documents);
        std::fs::write(&format!("{}/index.html", output_dir), index_html)
            .map_err(|e| format!("Failed to write index file: {}", e))?;

        Ok(())
    }

    /// Convert document to HTML
    fn document_to_html(document: &Document) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .document {{ max-width: 800px; margin: 0 auto; }}
        .title {{ color: #333; border-bottom: 2px solid #0066cc; }}
        .content {{ line-height: 1.6; }}
        .category {{ background: #f0f0f0; padding: 5px 10px; border-radius: 4px; display: inline-block; }}
        code {{ background: #f8f8f8; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="document">
        <h1 class="title">{}</h1>
        <div class="category">{:?}</div>
        <div class="content">
            <pre>{}</pre>
        </div>
    </div>
</body>
</html>"#,
            document.title,
            document.title,
            document.category,
            Self::escape_html(&document.content)
        )
    }

    /// Generate index HTML
    fn generate_index_html(documents: &[Document]) -> String {
        let mut doc_list = String::new();

        for document in documents {
            doc_list.push_str(&format!(
                r#"<li><a href="{}.html">{}</a> <span class="category">{:?}</span></li>"#,
                Self::sanitize_filename(&document.title),
                document.title,
                document.category
            ));
        }

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ΨLang Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .doc-list {{ display: grid; gap: 20px; }}
        .doc-item {{ background: #f9f9f9; padding: 20px; border-radius: 8px; }}
        .category {{ background: #0066cc; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ΨLang Documentation</h1>
            <p>Comprehensive documentation for the ΨLang neural computing platform</p>
        </div>
        <ul class="doc-list">
            {}
        </ul>
    </div>
</body>
</html>"#,
            doc_list
        )
    }

    /// Sanitize filename for HTML output
    fn sanitize_filename(title: &str) -> String {
        title.chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect::<String>()
            .to_lowercase()
    }

    /// Escape HTML characters
    fn escape_html(text: &str) -> String {
        text.replace("&", "&")
            .replace("<", "<")
            .replace(">", ">")
            .replace("\"", """)
            .replace("'", "'")
    }
}

/// Live Documentation Server
pub struct LiveDocumentationServer {
    docs: DocumentationSystem,
    browser: DocumentationBrowser,
    websocket_connections: Vec<WebSocketConnection>,
}

impl LiveDocumentationServer {
    /// Create a new live documentation server
    pub fn new() -> Self {
        Self {
            docs: DocumentationSystem::new(),
            browser: DocumentationBrowser::new(),
            websocket_connections: Vec::new(),
        }
    }

    /// Start the documentation server
    pub fn start_server(&mut self, port: u16) -> Result<(), String> {
        // Initialize with built-in documentation
        self.initialize_builtin_docs();

        println!("Live documentation server started on port {}", port);
        println!("Access documentation at http://localhost:{}/docs", port);

        Ok(())
    }

    /// Initialize built-in documentation
    fn initialize_builtin_docs(&mut self) {
        // Add basic documents
        let welcome_doc = Document {
            title: "Welcome to ΨLang".to_string(),
            content: "Welcome to ΨLang, the revolutionary neural computing platform...".to_string(),
            category: DocCategory::Guide,
            tags: vec!["welcome".to_string(), "introduction".to_string()],
            difficulty: DifficultyLevel::Beginner,
            last_updated: chrono::Utc::now().timestamp_millis() as f64,
            author: "ΨLang Team".to_string(),
            version: "1.0.0".to_string(),
        };

        self.docs.add_document(welcome_doc);

        // Add API references
        for api_ref in DocumentationGenerator::generate_api_docs() {
            self.docs.add_api_reference(api_ref);
        }
    }

    /// Handle WebSocket connection for real-time help
    pub fn handle_websocket_connection(&mut self, connection: WebSocketConnection) {
        self.websocket_connections.push(connection);
    }

    /// Broadcast update to all connections
    pub fn broadcast_update(&self, update: &str) {
        for connection in &self.websocket_connections {
            // In a real implementation, would send via WebSocket
            println!("Broadcasting to connection: {}", update);
        }
    }
}

/// WebSocket connection simulation
#[derive(Debug, Clone)]
pub struct WebSocketConnection {
    pub id: String,
    pub connected: bool,
}

/// Utility functions for documentation
pub mod utils {
    use super::*;

    /// Create a comprehensive documentation system
    pub fn create_documentation_system() -> DocumentationSystem {
        let mut system = DocumentationSystem::new();

        // Add built-in documents
        let documents = vec![
            Document {
                title: "ΨLang Language Reference".to_string(),
                content: "Complete reference for ΨLang syntax and semantics...".to_string(),
                category: DocCategory::Reference,
                tags: vec!["language".to_string(), "syntax".to_string()],
                difficulty: DifficultyLevel::Intermediate,
                last_updated: chrono::Utc::now().timestamp_millis() as f64,
                author: "ΨLang Team".to_string(),
                version: "1.0.0".to_string(),
            },
            Document {
                title: "Neural Network Concepts".to_string(),
                content: "Understanding spiking neural networks and neuromorphic computing...".to_string(),
                category: DocCategory::Tutorial,
                tags: vec!["neural-networks".to_string(), "concepts".to_string()],
                difficulty: DifficultyLevel::Beginner,
                last_updated: chrono::Utc::now().timestamp_millis() as f64,
                author: "ΨLang Team".to_string(),
                version: "1.0.0".to_string(),
            },
        ];

        for document in documents {
            system.add_document(document);
        }

        // Add API references
        for api_ref in DocumentationGenerator::generate_api_docs() {
            system.add_api_reference(api_ref);
        }

        system
    }

    /// Create an interactive documentation browser
    pub fn create_documentation_browser() -> DocumentationBrowser {
        DocumentationBrowser::new()
    }

    /// Generate quick start guide
    pub fn generate_quick_start_guide() -> String {
        format!(
            r#"# ΨLang Quick Start Guide

## 1. Creating Your First Neuron
```psilang
topology ⟪my_first_network⟫ {{
    ∴ neuron₁ {{
        threshold: -50mV,
        resting_potential: -70mV,
        reset_potential: -80mV
    }}
}}

execute ⟪my_first_network⟫ for 100ms
```

## 2. Connecting Neurons
```psilang
neuron₁ ⊸0.5:1ms⊸ neuron₂
```

## 3. Adding Learning
```psilang
neuron₁ ⊸0.5:1ms⊸ neuron₂ with STDP
```

## 4. Running Analysis
```psilang
analyze patterns in ⟪my_first_network⟫ for 1000ms
```

## Next Steps
- Explore the examples gallery
- Try the interactive tutorials
- Read the full documentation

Welcome to the future of neural computing! 🚀"#
        )
    }

    /// Generate cheat sheet
    pub fn generate_cheat_sheet() -> String {
        format!(
            r#"# ΨLang Cheat Sheet

## Neuron Definition
∴ neuron_name {{
    threshold: -50mV,
    resting_potential: -70mV,
    reset_potential: -80mV,
    refractory_period: 2ms
}}

## Synapse Connection
neuron₁ ⊸weight:delay⊸ neuron₂
neuron₁ ⊸0.5:1ms⊸ neuron₂ with STDP

## Topology Definition
topology ⟪network_name⟫ {{
    ∴ neuron₁ {{ parameters }}
    ∴ neuron₂ {{ parameters }}
    neuron₁ ⊸ neuron₂
}}

## Execution
execute ⟪network⟫ for 100ms
simulate ⟪network⟫ until convergence

## Pattern Analysis
analyze patterns in ⟪network⟫ for 1000ms
detect spikes in ⟪network⟫

## Learning
train ⟪network⟫ with dataset for 100 epochs
apply STDP to ⟪synapse⟫

## Cognitive Functions
working_memory ⟪wm⟫ {{ capacity: 50 }}
attention ⟪att⟫ {{ focus_radius: 10.0 }}

## Computer Vision
vision_pipeline ⟪vp⟫ {{
    preprocessing {{ grayscale, blur }}
    feature_extraction {{ edges, corners }}
    classification {{ cnn }}
}}

## Data Operations
import data from "file.csv"
export network to "network.json"
serialize ⟪network⟫ as binary

## Hardware Deployment
deploy ⟪network⟫ to loihi
execute on akida for 1000ms

## Quick Tips
- Use ⟪ ⟫ for named blocks
- Use ⊸ for synaptic connections
- Use ∴ for neuron definitions
- Use {{ }} for parameter blocks
- Use with for adding properties

Happy coding! 🎯"#
        )
    }
}