//! # ΨLang Lexer
//!
//! Lexical analyzer for the ΨLang programming language.
//! Tokenizes source code into a stream of tokens for parsing.

use logos::{Logos, SpannedIter};
use std::fmt;

/// Position in source code for error reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// Token types in ΨLang
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\n\f]+")] // Skip whitespace
#[logos(skip r"//[^\n]*\n?")] // Skip single-line comments
#[logos(skip r"/\*([^*]|\*[^/])*\*/")] // Skip multi-line comments
pub enum Token {
    // Neural operators
    #[token("⚡")]
    SpikeInjection,

    #[token("⟿")]
    Propagation,

    #[token("⊸")]
    SynapticConnection,

    #[token("∴")]
    NeuronDeclaration,

    #[token("←")]
    Assignment,

    #[token("→")]
    Causation,

    #[token("↑")]
    Potentiation,

    #[token("↓")]
    Depression,

    #[token("⏱")]
    TemporalMarker,

    #[token("∀")]
    UniversalQuantifier,

    #[token("∃")]
    ExistentialQuantifier,

    #[token("∈")]
    Membership,

    #[token("⊗")]
    TensorProduct,

    #[token("⊕")]
    AssemblyComposition,

    #[token("◉")]
    AttentionalFocus,

    #[token("≈")]
    ApproximateMatch,

    #[token("∿")]
    OscillatoryCoupling,

    #[token("⇝")]
    DelayedConnection,

    #[token("⊶")]
    ModulatorySynapse,

    // Keywords
    #[token("topology")]
    Topology,

    #[token("pattern")]
    Pattern,

    #[token("assembly")]
    Assembly,

    #[token("neuron")]
    Neuron,

    #[token("synapse")]
    Synapse,

    #[token("learning")]
    Learning,

    #[token("stdp")]
    STDP,

    #[token("hebbian")]
    Hebbian,

    #[token("flow")]
    Flow,

    #[token("evolve")]
    Evolve,

    #[token("monitor")]
    Monitor,

    #[token("with")]
    With,

    #[token("where")]
    Where,

    #[token("true")]
    True,

    #[token("false")]
    False,

    #[token("import")]
    Import,

    #[token("export")]
    Export,

    #[token("module")]
    Module,

    #[token("macro")]
    Macro,

    #[token("type")]
    Type,

    // Literals
    #[regex(r"-?[0-9]+(\.[0-9]+)?([eE]-?[0-9]+)?")]
    Float,

    #[regex(r"[A-Za-z_][A-Za-z0-9_]*")]
    Identifier,

    #[regex(r"'([^'\\]|\\.)*'")]
    String,

    // Units
    #[regex(r"[0-9]+(\.[0-9]+)?[mun]?[sS]")]
    Duration,

    #[regex(r"-?[0-9]+(\.[0-9]+)?[mun]?[Vv]")]
    Voltage,

    #[regex(r"[0-9]+(\.[0-9]+)?[HkMk]?Hz")]
    Frequency,

    #[regex(r"[0-9]+(\.[0-9]+)?[mun]?[Aa]")]
    Current,

    #[regex(r"[0-9]+(\.[0-9]+)?[mun]?[Ss]")]
    Conductance,

    // Punctuation
    #[token("{")]
    LeftBrace,

    #[token("}")]
    RightBrace,

    #[token("[")]
    LeftBracket,

    #[token("]")]
    RightBracket,

    #[token("(")]
    LeftParen,

    #[token(")")]
    RightParen,

    #[token(",")]
    Comma,

    #[token(":")]
    Colon,

    #[token(";")]
    Semicolon,

    #[token(".")]
    Dot,

    #[token("=")]
    Equals,

    #[token("::")]
    DoubleColon,

    #[token("⟪")]
    LeftDoubleAngle,

    #[token("⟫")]
    RightDoubleAngle,

    #[token("∧")]
    LogicalAnd,

    #[token("∨")]
    LogicalOr,

    #[token("¬")]
    LogicalNot,

    #[token("Δ")]
    Delta,

    #[token("λ")]
    Lambda,

    #[token("Π")]
    PiProduct,

    #[token("Σ")]
    SigmaSum,

    #[token("∂")]
    PartialDerivative,

    // Special patterns
    #[regex(r"⟨[^⟩]*⟩")]
    PatternDelimiter,

    #[regex(r"∘")]
    Composition,

    #[regex(r"⊙")]
    ElementWiseProduct,

    #[regex(r"⊘")]
    ElementWiseDivision,

    // End of file
    #[logos(skip r"[ \t\n\f]+")]
    EOF,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::SpikeInjection => write!(f, "⚡"),
            Token::Propagation => write!(f, "⟿"),
            Token::SynapticConnection => write!(f, "⊸"),
            Token::NeuronDeclaration => write!(f, "∴"),
            Token::Assignment => write!(f, "←"),
            Token::Causation => write!(f, "→"),
            Token::Potentiation => write!(f, "↑"),
            Token::Depression => write!(f, "↓"),
            Token::TemporalMarker => write!(f, "⏱"),
            Token::UniversalQuantifier => write!(f, "∀"),
            Token::ExistentialQuantifier => write!(f, "∃"),
            Token::Membership => write!(f, "∈"),
            Token::TensorProduct => write!(f, "⊗"),
            Token::AssemblyComposition => write!(f, "⊕"),
            Token::AttentionalFocus => write!(f, "◉"),
            Token::ApproximateMatch => write!(f, "≈"),
            Token::OscillatoryCoupling => write!(f, "∿"),
            Token::DelayedConnection => write!(f, "⇝"),
            Token::ModulatorySynapse => write!(f, "⊶"),
            Token::Topology => write!(f, "topology"),
            Token::Pattern => write!(f, "pattern"),
            Token::Assembly => write!(f, "assembly"),
            Token::Neuron => write!(f, "neuron"),
            Token::Synapse => write!(f, "synapse"),
            Token::Learning => write!(f, "learning"),
            Token::STDP => write!(f, "stdp"),
            Token::Hebbian => write!(f, "hebbian"),
            Token::Flow => write!(f, "flow"),
            Token::Evolve => write!(f, "evolve"),
            Token::Monitor => write!(f, "monitor"),
            Token::With => write!(f, "with"),
            Token::Where => write!(f, "where"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Import => write!(f, "import"),
            Token::Export => write!(f, "export"),
            Token::Module => write!(f, "module"),
            Token::Macro => write!(f, "macro"),
            Token::Type => write!(f, "type"),
            Token::Float => write!(f, "float_literal"),
            Token::Identifier => write!(f, "identifier"),
            Token::String => write!(f, "string_literal"),
            Token::Duration => write!(f, "duration_literal"),
            Token::Voltage => write!(f, "voltage_literal"),
            Token::Frequency => write!(f, "frequency_literal"),
            Token::Current => write!(f, "current_literal"),
            Token::Conductance => write!(f, "conductance_literal"),
            Token::LeftBrace => write!(f, "{{"),
            Token::RightBrace => write!(f, "}}"),
            Token::LeftBracket => write!(f, "["),
            Token::RightBracket => write!(f, "]"),
            Token::LeftParen => write!(f, "("),
            Token::RightParen => write!(f, ")"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Semicolon => write!(f, ";"),
            Token::Dot => write!(f, "."),
            Token::Equals => write!(f, "="),
            Token::DoubleColon => write!(f, "::"),
            Token::LeftDoubleAngle => write!(f, "⟪"),
            Token::RightDoubleAngle => write!(f, "⟫"),
            Token::LogicalAnd => write!(f, "∧"),
            Token::LogicalOr => write!(f, "∨"),
            Token::LogicalNot => write!(f, "¬"),
            Token::Delta => write!(f, "Δ"),
            Token::Lambda => write!(f, "λ"),
            Token::PiProduct => write!(f, "Π"),
            Token::SigmaSum => write!(f, "Σ"),
            Token::PartialDerivative => write!(f, "∂"),
            Token::PatternDelimiter => write!(f, "pattern_delimiter"),
            Token::Composition => write!(f, "∘"),
            Token::ElementWiseProduct => write!(f, "⊙"),
            Token::ElementWiseDivision => write!(f, "⊘"),
            Token::EOF => write!(f, "EOF"),
        }
    }
}

/// Token with position information
#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
    pub slice: String,
}

impl SpannedToken {
    pub fn new(token: Token, span: Span, slice: String) -> Self {
        Self { token, span, slice }
    }
}

/// Lexer error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum LexerError {
    #[error("Invalid token at position {span}: {message}")]
    InvalidToken { span: Span, message: String },

    #[error("Unexpected character '{character}' at position {position}")]
    UnexpectedCharacter { character: char, position: usize },

    #[error("Unterminated string literal starting at position {position}")]
    UnterminatedString { position: usize },

    #[error("Invalid number format at position {span}: {message}")]
    InvalidNumber { span: Span, message: String },
}

/// Result type for lexer operations
pub type LexerResult<T> = Result<T, LexerError>;

/// Main lexer struct
#[derive(Debug)]
pub struct Lexer<'source> {
    source: &'source str,
    chars: Vec<char>,
    position: usize,
}

impl<'source> Lexer<'source> {
    /// Create a new lexer from source code
    pub fn new(source: &'source str) -> Self {
        Self {
            source,
            chars: source.chars().collect(),
            position: 0,
        }
    }

    /// Tokenize the entire source into tokens
    pub fn tokenize(&self) -> LexerResult<Vec<SpannedToken>> {
        let mut tokens = Vec::new();
        let mut lexer = Token::lexer(self.source);

        while let Some(token) = lexer.next() {
            match token {
                Ok(token) => {
                    let span = Span::new(lexer.span().start, lexer.span().end);
                    let slice = lexer.slice().to_string();
                    tokens.push(SpannedToken::new(token, span, slice));
                }
                Err(_) => {
                    let span = Span::new(lexer.span().start, lexer.span().end);
                    return Err(LexerError::InvalidToken {
                        span,
                        message: "Unrecognized token".to_string(),
                    });
                }
            }
        }

        Ok(tokens)
    }

    /// Tokenize with detailed error reporting
    pub fn tokenize_with_errors(&self) -> (Vec<SpannedToken>, Vec<LexerError>) {
        let mut tokens = Vec::new();
        let mut errors = Vec::new();
        let mut lexer = Token::lexer(self.source);

        while let Some(result) = lexer.next() {
            match result {
                Ok(token) => {
                    let span = Span::new(lexer.span().start, lexer.span().end);
                    let slice = lexer.slice().to_string();
                    tokens.push(SpannedToken::new(token, span, slice));
                }
                Err(_) => {
                    let span = Span::new(lexer.span().start, lexer.span().end);
                    errors.push(LexerError::InvalidToken {
                        span,
                        message: "Unrecognized token".to_string(),
                    });
                }
            }
        }

        (tokens, errors)
    }
}

/// Convenience function to tokenize source code
pub fn tokenize(source: &str) -> LexerResult<Vec<SpannedToken>> {
    let lexer = Lexer::new(source);
    lexer.tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let source = "⚡ ∴ ⊸ ⟿";
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].token, Token::SpikeInjection);
        assert_eq!(tokens[1].token, Token::NeuronDeclaration);
        assert_eq!(tokens[2].token, Token::SynapticConnection);
        assert_eq!(tokens[3].token, Token::Propagation);
    }

    #[test]
    fn test_neuron_declaration() {
        let source = "∴ neuron₁ { threshold: -50mV, leak: 10mV/ms }";
        let tokens = tokenize(source).unwrap();

        assert!(tokens.iter().any(|t| t.token == Token::NeuronDeclaration));
        assert!(tokens.iter().any(|t| t.token == Token::Identifier));
        assert!(tokens.iter().any(|t| t.token == Token::Voltage));
    }

    #[test]
    fn test_synapse_connection() {
        let source = "neuron₁ ⊸0.5:2ms⊸ neuron₂";
        let tokens = tokenize(source).unwrap();

        assert!(tokens.iter().any(|t| t.token == Token::SynapticConnection));
        assert!(tokens.iter().any(|t| t.token == Token::Float));
        assert!(tokens.iter().any(|t| t.token == Token::Duration));
    }

    #[test]
    fn test_keywords() {
        let source = "topology pattern assembly learning stdp";
        let tokens = tokenize(source).unwrap();

        assert!(tokens.iter().any(|t| t.token == Token::Topology));
        assert!(tokens.iter().any(|t| t.token == Token::Pattern));
        assert!(tokens.iter().any(|t| t.token == Token::Assembly));
        assert!(tokens.iter().any(|t| t.token == Token::Learning));
        assert!(tokens.iter().any(|t| t.token == Token::STDP));
    }

    #[test]
    fn test_comments() {
        let source = r#"
// This is a comment
∴ neuron /* inline comment */ ⊸0.5⊸
/*
Multi-line comment
*/
⚡
"#;

        let tokens = tokenize(source).unwrap();
        // Should only have the actual tokens, comments should be filtered out
        assert!(tokens.iter().any(|t| t.token == Token::NeuronDeclaration));
        assert!(tokens.iter().any(|t| t.token == Token::SynapticConnection));
        assert!(tokens.iter().any(|t| t.token == Token::SpikeInjection));
    }
}