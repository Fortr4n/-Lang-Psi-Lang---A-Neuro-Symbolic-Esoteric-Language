//! # ΨLang Parser
//!
//! Recursive descent parser for the ΨLang programming language.
//! Converts token streams into abstract syntax trees.

use crate::ast::*;
use crate::lexer::{Token, SpannedToken, LexerError};
use std::iter::Peekable;
use std::slice::Iter;

/// Parser error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParserError {
    #[error("Unexpected token at position {span}: expected {expected}, found {found}")]
    UnexpectedToken {
        span: Span,
        expected: String,
        found: String,
    },

    #[error("Unexpected end of input at position {span}")]
    UnexpectedEOF { span: Span },

    #[error("Invalid syntax at position {span}: {message}")]
    InvalidSyntax { span: Span, message: String },

    #[error("Duplicate declaration at position {span}: {name}")]
    DuplicateDeclaration { span: Span, name: String },

    #[error("Undefined reference at position {span}: {name}")]
    UndefinedReference { span: Span, name: String },

    #[error("Type error at position {span}: {message}")]
    TypeError { span: Span, message: String },
}

/// Result type for parser operations
pub type ParserResult<T> = Result<T, ParserError>;

/// Main parser struct
#[derive(Debug)]
pub struct Parser<'source> {
    tokens: Peekable<Iter<'source, SpannedToken>>,
    current_token: Option<&'source SpannedToken>,
    position: usize,
    errors: Vec<ParserError>,
}

impl<'source> Parser<'source> {
    /// Create a new parser from tokens
    pub fn new(tokens: &'source [SpannedToken]) -> Self {
        let mut parser = Self {
            tokens: tokens.iter().peekable(),
            current_token: None,
            position: 0,
            errors: Vec::new(),
        };
        parser.advance();
        parser
    }

    /// Parse a complete program
    pub fn parse(&mut self) -> ParserResult<Program> {
        let start_pos = self.current_position();

        let header = if self.current_token_matches(&[Token::Topology]) {
            Some(self.parse_topology_header()?)
        } else {
            None
        };

        let mut imports = Vec::new();
        while self.current_token_matches(&[Token::Import]) {
            imports.push(self.parse_import_decl()?);
        }

        let mut declarations = Vec::new();
        while !self.is_at_end() {
            declarations.push(self.parse_declaration()?);
        }

        let end_pos = self.current_position();
        Ok(Program {
            span: Span::new(start_pos, end_pos, 1, 0),
            header,
            imports,
            declarations,
        })
    }

    /// Parse topology header
    fn parse_topology_header(&mut self) -> ParserResult<TopologyHeader> {
        let start_pos = self.current_position();

        self.expect_token(Token::Topology)?;
        let name = self.parse_identifier()?;

        let parameters = if self.current_token_matches(&[Token::With]) {
            self.advance();
            Some(self.parse_program_params()?)
        } else {
            None
        };

        let end_pos = self.current_position();
        Ok(TopologyHeader {
            span: Span::new(start_pos, end_pos, 1, 0),
            name,
            parameters,
        })
    }

    /// Parse program parameters
    fn parse_program_params(&mut self) -> ParserResult<ProgramParams> {
        let start_pos = self.current_position();

        self.expect_token(Token::LeftBrace)?;
        let mut precision = None;
        let mut learning_enabled = None;
        let mut evolution_enabled = None;
        let mut monitoring_enabled = None;

        while !self.current_token_matches(&[Token::RightBrace]) && !self.is_at_end() {
            let param_name = self.parse_identifier()?;

            self.expect_token(Token::Colon)?;

            match param_name.as_str() {
                "precision" => {
                    let prec_str = self.parse_identifier()?;
                    precision = Some(match prec_str.as_str() {
                        "single" => Precision::Single,
                        "double" => Precision::Double,
                        "extended" => Precision::Extended,
                        "quad" => Precision::Quad,
                        "half" => Precision::Half,
                        _ => return Err(ParserError::InvalidSyntax {
                            span: self.current_position_span(),
                            message: format!("Invalid precision: {}", prec_str),
                        }),
                    });
                }
                "learning" => {
                    let value = self.parse_boolean()?;
                    learning_enabled = Some(value);
                }
                "evolution" => {
                    let value = self.parse_boolean()?;
                    evolution_enabled = Some(value);
                }
                "monitoring" => {
                    let value = self.parse_boolean()?;
                    monitoring_enabled = Some(value);
                }
                _ => return Err(ParserError::InvalidSyntax {
                    span: self.current_position_span(),
                    message: format!("Unknown parameter: {}", param_name),
                }),
            }

            if !self.current_token_matches(&[Token::RightBrace]) {
                self.expect_token(Token::Comma)?;
            }
        }

        self.expect_token(Token::RightBrace)?;
        let end_pos = self.current_position();

        Ok(ProgramParams {
            span: Span::new(start_pos, end_pos, 1, 0),
            precision,
            learning_enabled,
            evolution_enabled,
            monitoring_enabled,
        })
    }

    /// Parse import declaration
    fn parse_import_decl(&mut self) -> ParserResult<ImportDecl> {
        let start_pos = self.current_position();

        self.expect_token(Token::Import)?;
        let module = self.parse_string()?;

        let alias = if self.current_token_matches(&[Token::Identifier]) {
            let token = self.current_token.as_ref().unwrap();
            if token.slice == "as" {
                self.advance();
                Some(self.parse_identifier()?)
            } else {
                None
            }
        } else {
            None
        };

        let end_pos = self.current_position();
        Ok(ImportDecl {
            span: Span::new(start_pos, end_pos, 1, 0),
            module,
            alias,
            imports: None,
        })
    }

    /// Parse any declaration
    fn parse_declaration(&mut self) -> ParserResult<Declaration> {
        match self.current_token {
            Some(token) => match &token.token {
                Token::NeuronDeclaration => {
                    Ok(Declaration::Neuron(self.parse_neuron_decl()?))
                }
                Token::SynapticConnection => {
                    Ok(Declaration::Synapse(self.parse_synapse_decl()?))
                }
                Token::Assembly => {
                    Ok(Declaration::Assembly(self.parse_assembly_decl()?))
                }
                Token::Pattern => {
                    Ok(Declaration::Pattern(self.parse_pattern_decl()?))
                }
                Token::Flow => {
                    Ok(Declaration::Flow(self.parse_flow_decl()?))
                }
                Token::Learning => {
                    Ok(Declaration::Learning(self.parse_learning_decl()?))
                }
                Token::Evolve | Token::Monitor => {
                    Ok(Declaration::Control(self.parse_control_decl()?))
                }
                Token::Type => {
                    Ok(Declaration::Type(self.parse_type_decl()?))
                }
                Token::Module => {
                    Ok(Declaration::Module(self.parse_module_decl()?))
                }
                Token::Macro => {
                    Ok(Declaration::Macro(self.parse_macro_decl()?))
                }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "declaration".to_string(),
                    found: token.token.to_string(),
                }),
            },
            None => Err(ParserError::UnexpectedEOF {
                span: Span::new(self.position, self.position, 1, 0),
            }),
        }
    }

    /// Parse neuron declaration
    fn parse_neuron_decl(&mut self) -> ParserResult<NeuronDecl> {
        let start_pos = self.current_position();

        self.expect_token(Token::NeuronDeclaration)?;
        let name = self.parse_identifier()?;

        let neuron_type = if self.current_token_matches(&[Token::Colon]) {
            self.advance();
            Some(self.parse_neuron_type()?)
        } else {
            None
        };

        let parameters = self.parse_neuron_params()?;

        let end_pos = self.current_position();
        Ok(NeuronDecl {
            span: Span::new(start_pos, end_pos, 1, 0),
            name,
            neuron_type,
            parameters,
        })
    }

    /// Parse neuron type
    fn parse_neuron_type(&mut self) -> ParserResult<NeuronType> {
        let token = self.current_token.as_ref().unwrap();

        match &token.token {
            Token::Identifier => {
                let type_name = token.slice.clone();
                self.advance();
                match type_name.as_str() {
                    "lif" => Ok(NeuronType::LIF),
                    "izhikevich" => Ok(NeuronType::Izhikevich),
                    "hodgkin_huxley" => Ok(NeuronType::HodgkinHuxley),
                    "adaptive_exponential" => Ok(NeuronType::AdaptiveExponential),
                    "quantum" => Ok(NeuronType::Quantum),
                    "stochastic" => Ok(NeuronType::Stochastic),
                    _ => Ok(NeuronType::Custom(type_name)),
                }
            }
            _ => Err(ParserError::UnexpectedToken {
                span: token.span,
                expected: "neuron type".to_string(),
                found: token.token.to_string(),
            }),
        }
    }

    /// Parse neuron parameters
    fn parse_neuron_params(&mut self) -> ParserResult<NeuronParams> {
        let start_pos = self.current_position();

        self.expect_token(Token::LeftBrace)?;

        let mut threshold = None;
        let mut leak_rate = None;
        let mut refractory_period = None;
        let mut position = None;
        let mut precision = None;

        while !self.current_token_matches(&[Token::RightBrace]) && !self.is_at_end() {
            let param_name = self.parse_identifier()?;
            self.expect_token(Token::Colon)?;

            match param_name.as_str() {
                "threshold" => {
                    threshold = Some(self.parse_voltage()?);
                }
                "leak" => {
                    leak_rate = Some(self.parse_voltage_per_time()?);
                }
                "refractory" => {
                    refractory_period = Some(self.parse_duration()?);
                }
                "position" => {
                    position = Some(self.parse_position_3d()?);
                }
                "precision" => {
                    let prec_str = self.parse_identifier()?;
                    precision = Some(match prec_str.as_str() {
                        "single" => Precision::Single,
                        "double" => Precision::Double,
                        "extended" => Precision::Extended,
                        "quad" => Precision::Quad,
                        "half" => Precision::Half,
                        _ => return Err(ParserError::InvalidSyntax {
                            span: self.current_position_span(),
                            message: format!("Invalid precision: {}", prec_str),
                        }),
                    });
                }
                _ => return Err(ParserError::InvalidSyntax {
                    span: self.current_position_span(),
                    message: format!("Unknown neuron parameter: {}", param_name),
                }),
            }

            if !self.current_token_matches(&[Token::RightBrace]) {
                self.expect_token(Token::Comma)?;
            }
        }

        self.expect_token(Token::RightBrace)?;
        let end_pos = self.current_position();

        Ok(NeuronParams {
            span: Span::new(start_pos, end_pos, 1, 0),
            threshold,
            leak_rate,
            refractory_period,
            position,
            precision,
        })
    }

    /// Parse synapse declaration
    fn parse_synapse_decl(&mut self) -> ParserResult<SynapseDecl> {
        let start_pos = self.current_position();

        let presynaptic = self.parse_expression()?;
        self.expect_token(Token::SynapticConnection)?;

        let weight = if self.current_token_matches(&[Token::Float]) {
            Some(self.parse_weight()?)
        } else {
            None
        };

        let delay = if self.current_token_matches(&[Token::Colon]) {
            self.advance();
            Some(self.parse_duration()?)
        } else {
            None
        };

        self.expect_token(Token::SynapticConnection)?;
        let postsynaptic = self.parse_expression()?;

        let parameters = if self.current_token_matches(&[Token::With]) {
            self.advance();
            Some(self.parse_synapse_params()?)
        } else {
            None
        };

        let end_pos = self.current_position();
        Ok(SynapseDecl {
            span: Span::new(start_pos, end_pos, 1, 0),
            presynaptic,
            weight,
            delay,
            postsynaptic,
            parameters,
        })
    }

    /// Parse synapse parameters
    fn parse_synapse_params(&mut self) -> ParserResult<SynapseParams> {
        let start_pos = self.current_position();

        self.expect_token(Token::LeftBrace)?;

        let mut plasticity = None;
        let mut modulatory = None;
        let mut delay = None;

        while !self.current_token_matches(&[Token::RightBrace]) && !self.is_at_end() {
            let param_name = self.parse_identifier()?;
            self.expect_token(Token::Colon)?;

            match param_name.as_str() {
                "plasticity" => {
                    plasticity = Some(self.parse_plasticity_rule()?);
                }
                "modulatory" => {
                    let mod_str = self.parse_identifier()?;
                    modulatory = Some(match mod_str.as_str() {
                        "excitation" => ModulationType::Excitation,
                        "inhibition" => ModulationType::Inhibition,
                        _ => return Err(ParserError::InvalidSyntax {
                            span: self.current_position_span(),
                            message: format!("Invalid modulation type: {}", mod_str),
                        }),
                    });
                }
                "delay" => {
                    delay = Some(self.parse_duration()?);
                }
                _ => return Err(ParserError::InvalidSyntax {
                    span: self.current_position_span(),
                    message: format!("Unknown synapse parameter: {}", param_name),
                }),
            }

            if !self.current_token_matches(&[Token::RightBrace]) {
                self.expect_token(Token::Comma)?;
            }
        }

        self.expect_token(Token::RightBrace)?;
        let end_pos = self.current_position();

        Ok(SynapseParams {
            span: Span::new(start_pos, end_pos, 1, 0),
            plasticity,
            modulatory,
            delay,
        })
    }

    /// Parse assembly declaration
    fn parse_assembly_decl(&mut self) -> ParserResult<AssemblyDecl> {
        let start_pos = self.current_position();

        self.expect_token(Token::Assembly)?;
        let name = self.parse_identifier()?;
        self.expect_token(Token::LeftBrace)?;

        let body = self.parse_assembly_body()?;

        self.expect_token(Token::RightBrace)?;
        let end_pos = self.current_position();

        Ok(AssemblyDecl {
            span: Span::new(start_pos, end_pos, 1, 0),
            name,
            body,
        })
    }

    /// Parse assembly body
    fn parse_assembly_body(&mut self) -> ParserResult<AssemblyBody> {
        let start_pos = self.current_position();

        let mut neurons = Vec::new();
        let mut connections = Vec::new();
        let mut plasticity = Vec::new();

        while !self.current_token_matches(&[Token::RightBrace]) && !self.is_at_end() {
            let param_name = self.parse_identifier()?;
            self.expect_token(Token::Colon)?;

            match param_name.as_str() {
                "neurons" => {
                    neurons = self.parse_expression_list()?;
                }
                "connections" => {
                    connections = self.parse_connection_specs()?;
                }
                "plasticity" => {
                    plasticity = self.parse_plasticity_rules()?;
                }
                _ => return Err(ParserError::InvalidSyntax {
                    span: self.current_position_span(),
                    message: format!("Unknown assembly parameter: {}", param_name),
                }),
            }

            if !self.current_token_matches(&[Token::RightBrace]) {
                self.expect_token(Token::Comma)?;
            }
        }

        let end_pos = self.current_position();
        Ok(AssemblyBody {
            span: Span::new(start_pos, end_pos, 1, 0),
            neurons,
            connections,
            plasticity,
        })
    }

    /// Parse pattern declaration
    fn parse_pattern_decl(&mut self) -> ParserResult<PatternDecl> {
        let start_pos = self.current_position();

        self.expect_token(Token::Pattern)?;
        let name = self.parse_identifier()?;
        self.expect_token(Token::LeftBrace)?;

        let body = self.parse_pattern_body()?;

        self.expect_token(Token::RightBrace)?;
        let end_pos = self.current_position();

        Ok(PatternDecl {
            span: Span::new(start_pos, end_pos, 1, 0),
            name,
            body,
        })
    }

    /// Parse pattern body
    fn parse_pattern_body(&mut self) -> ParserResult<PatternBody> {
        // This is a simplified version - in practice, we'd need to handle
        // spike sequences, temporal constraints, and pattern composition
        if self.current_token_matches(&[Token::SpikeInjection]) {
            let spike_events = self.parse_spike_events()?;
            Ok(PatternBody::SpikeSequence(spike_events))
        } else {
            // For now, return empty spike sequence
            Ok(PatternBody::SpikeSequence(Vec::new()))
        }
    }

    /// Parse flow declaration
    fn parse_flow_decl(&mut self) -> ParserResult<FlowDecl> {
        let start_pos = self.current_position();

        self.expect_token(Token::Flow)?;
        let name = if self.current_token_matches(&[Token::Identifier]) {
            Some(self.parse_identifier()?)
        } else {
            None
        };

        self.expect_token(Token::LeftBrace)?;
        let mut rules = Vec::new();

        while !self.current_token_matches(&[Token::RightBrace]) && !self.is_at_end() {
            rules.push(self.parse_flow_rule()?);
            if self.current_token_matches(&[Token::Semicolon]) {
                self.advance();
            }
        }

        self.expect_token(Token::RightBrace)?;
        let end_pos = self.current_position();

        Ok(FlowDecl {
            span: Span::new(start_pos, end_pos, 1, 0),
            name,
            rules,
        })
    }

    /// Parse flow rule
    fn parse_flow_rule(&mut self) -> ParserResult<FlowRule> {
        let start_pos = self.current_position();

        let source = self.parse_expression()?;
        self.expect_token(Token::Propagation)?;
        let conditions = self.parse_conditions()?;
        self.expect_token(Token::Causation)?;
        let target = self.parse_expression()?;

        let end_pos = self.current_position();
        Ok(FlowRule {
            span: Span::new(start_pos, end_pos, 1, 0),
            source,
            conditions,
            target,
        })
    }

    /// Parse learning declaration
    fn parse_learning_decl(&mut self) -> ParserResult<LearningDecl> {
        let start_pos = self.current_position();

        self.expect_token(Token::Learning)?;
        self.expect_token(Token::Colon)?;
        let rule = self.parse_learning_rule()?;

        let end_pos = self.current_position();
        Ok(LearningDecl {
            span: Span::new(start_pos, end_pos, 1, 0),
            rule,
        })
    }

    /// Parse control declaration
    fn parse_control_decl(&mut self) -> ParserResult<ControlDecl> {
        let start_pos = self.current_position();

        let control_type = if self.current_token_matches(&[Token::Evolve]) {
            self.advance();
            ControlType::Evolve(self.parse_evolution_strategy()?)
        } else if self.current_token_matches(&[Token::Monitor]) {
            self.advance();
            ControlType::Monitor(self.parse_monitoring_spec()?)
        } else {
            return Err(ParserError::UnexpectedToken {
                span: self.current_position_span(),
                expected: "evolve or monitor".to_string(),
                found: self.current_token.unwrap().token.to_string(),
            });
        };

        let end_pos = self.current_position();
        Ok(ControlDecl {
            span: Span::new(start_pos, end_pos, 1, 0),
            control_type,
        })
    }

    /// Parse type declaration
    fn parse_type_decl(&mut self) -> ParserResult<TypeDecl> {
        let start_pos = self.current_position();

        self.expect_token(Token::Type)?;
        let name = self.parse_identifier()?;
        self.expect_token(Token::Equals)?;
        let type_expr = self.parse_type_expression()?;

        let end_pos = self.current_position();
        Ok(TypeDecl {
            span: Span::new(start_pos, end_pos, 1, 0),
            name,
            type_expr,
        })
    }

    /// Parse module declaration
    fn parse_module_decl(&mut self) -> ParserResult<ModuleDecl> {
        let start_pos = self.current_position();

        self.expect_token(Token::Module)?;
        let name = self.parse_identifier()?;
        self.expect_token(Token::LeftBrace)?;

        // For simplicity, we'll parse declarations directly
        let mut declarations = Vec::new();
        while !self.current_token_matches(&[Token::RightBrace]) && !self.is_at_end() {
            declarations.push(self.parse_declaration()?);
        }

        self.expect_token(Token::RightBrace)?;
        let end_pos = self.current_position();

        Ok(ModuleDecl {
            span: Span::new(start_pos, end_pos, 1, 0),
            name,
            exports: Vec::new(), // Simplified for now
            imports: Vec::new(), // Simplified for now
            declarations,
        })
    }

    /// Parse macro declaration
    fn parse_macro_decl(&mut self) -> ParserResult<MacroDecl> {
        let start_pos = self.current_position();

        self.expect_token(Token::Macro)?;
        let name = self.parse_identifier()?;
        self.expect_token(Token::LeftParen)?;

        let mut parameters = Vec::new();
        if !self.current_token_matches(&[Token::RightParen]) {
            parameters.push(self.parse_identifier()?);
            while self.current_token_matches(&[Token::Comma]) {
                self.advance();
                parameters.push(self.parse_identifier()?);
            }
        }

        self.expect_token(Token::RightParen)?;
        self.expect_token(Token::Equals)?;
        let body = self.parse_expression()?;

        let end_pos = self.current_position();
        Ok(MacroDecl {
            span: Span::new(start_pos, end_pos, 1, 0),
            name,
            parameters,
            body,
        })
    }

    /// Parse expression
    fn parse_expression(&mut self) -> ParserResult<Expression> {
        self.parse_assignment_expression()
    }

    /// Parse assignment expression
    fn parse_assignment_expression(&mut self) -> ParserResult<Expression> {
        let expr = self.parse_conditional_expression()?;

        if self.current_token_matches(&[Token::Assignment]) {
            self.advance();
            let value = self.parse_assignment_expression()?;
            Ok(Expression::BinaryOp(BinaryOp {
                span: Span::new(expr.span().start, value.span().end, 1, 0),
                left: Box::new(expr),
                operator: BinaryOperator::Assignment,
                right: Box::new(value),
            }))
        } else {
            Ok(expr)
        }
    }

    /// Parse conditional expression
    fn parse_conditional_expression(&mut self) -> ParserResult<Expression> {
        let expr = self.parse_flow_expression()?;

        if self.current_token_matches(&[Token::LogicalAnd, Token::LogicalOr]) {
            let operator = match self.current_token.unwrap().token {
                Token::LogicalAnd => BinaryOperator::LogicalAnd,
                Token::LogicalOr => BinaryOperator::LogicalOr,
                _ => unreachable!(),
            };
            self.advance();

            let right = self.parse_conditional_expression()?;
            Ok(Expression::BinaryOp(BinaryOp {
                span: Span::new(expr.span().start, right.span().end, 1, 0),
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            }))
        } else {
            Ok(expr)
        }
    }

    /// Parse flow expression
    fn parse_flow_expression(&mut self) -> ParserResult<Expression> {
        let expr = self.parse_temporal_expression()?;

        if self.current_token_matches(&[Token::Propagation]) {
            self.advance();
            let conditions = self.parse_conditions()?;
            self.expect_token(Token::Causation)?;
            let target = self.parse_temporal_expression()?;

            Ok(Expression::BinaryOp(BinaryOp {
                span: Span::new(expr.span().start, target.span().end, 1, 0),
                left: Box::new(expr),
                operator: BinaryOperator::Flow,
                right: Box::new(target),
            }))
        } else {
            Ok(expr)
        }
    }

    /// Parse temporal expression
    fn parse_temporal_expression(&mut self) -> ParserResult<Expression> {
        let expr = self.parse_spike_expression()?;

        if self.current_token_matches(&[Token::TemporalMarker]) {
            self.advance();
            let duration = self.parse_duration()?;
            self.expect_token(Token::Causation)?;
            let action = self.parse_spike_expression()?;

            Ok(Expression::BinaryOp(BinaryOp {
                span: Span::new(expr.span().start, action.span().end, 1, 0),
                left: Box::new(expr),
                operator: BinaryOperator::Causation,
                right: Box::new(action),
            }))
        } else {
            Ok(expr)
        }
    }

    /// Parse spike expression
    fn parse_spike_expression(&mut self) -> ParserResult<Expression> {
        let expr = self.parse_product_expression()?;

        if self.current_token_matches(&[Token::SpikeInjection]) {
            self.advance();
            let amplitude = if self.current_token_matches(&[Token::Voltage, Token::Current]) {
                Some(self.parse_voltage()?)
            } else {
                None
            };

            let timestamp = if self.current_token_matches(&[Token::Duration]) {
                Some(self.parse_duration()?)
            } else {
                None
            };

            self.expect_token(Token::Causation)?;
            let target = self.parse_product_expression()?;

            Ok(Expression::UnaryOp(UnaryOp {
                span: Span::new(expr.span().start, target.span().end, 1, 0),
                operator: UnaryOperator::SpikeInjection,
                operand: Box::new(target),
            }))
        } else {
            Ok(expr)
        }
    }

    /// Parse product expression
    fn parse_product_expression(&mut self) -> ParserResult<Expression> {
        let expr = self.parse_assembly_expression()?;

        while self.current_token_matches(&[Token::TensorProduct, Token::AssemblyComposition]) {
            let operator = match self.current_token.unwrap().token {
                Token::TensorProduct => BinaryOperator::TensorProduct,
                Token::AssemblyComposition => BinaryOperator::AssemblyComposition,
                _ => unreachable!(),
            };
            self.advance();

            let right = self.parse_assembly_expression()?;
            expr = Expression::BinaryOp(BinaryOp {
                span: Span::new(expr.span().start, right.span().end, 1, 0),
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            });
        }

        Ok(expr)
    }

    /// Parse assembly expression
    fn parse_assembly_expression(&mut self) -> ParserResult<Expression> {
        let expr = self.parse_unary_expression()?;

        if self.current_token_matches(&[Token::Membership]) {
            self.advance();
            let assembly = self.parse_unary_expression()?;
            Ok(Expression::BinaryOp(BinaryOp {
                span: Span::new(expr.span().start, assembly.span().end, 1, 0),
                left: Box::new(expr),
                operator: BinaryOperator::LogicalAnd, // Simplified
                right: Box::new(assembly),
            }))
        } else {
            Ok(expr)
        }
    }

    /// Parse unary expression
    fn parse_unary_expression(&mut self) -> ParserResult<Expression> {
        if self.current_token_matches(&[
            Token::SpikeInjection, Token::Potentiation, Token::Depression,
            Token::AttentionalFocus, Token::LogicalNot
        ]) {
            let operator = match self.current_token.unwrap().token {
                Token::SpikeInjection => UnaryOperator::SpikeInjection,
                Token::Potentiation => UnaryOperator::Potentiation,
                Token::Depression => UnaryOperator::Depression,
                Token::AttentionalFocus => UnaryOperator::AttentionalFocus,
                Token::LogicalNot => UnaryOperator::LogicalNot,
                _ => unreachable!(),
            };
            self.advance();

            let operand = self.parse_unary_expression()?;
            Ok(Expression::UnaryOp(UnaryOp {
                span: Span::new(operand.span().start, operand.span().end, 1, 0),
                operator,
                operand: Box::new(operand),
            }))
        } else {
            self.parse_primary_expression()
        }
    }

    /// Parse primary expression
    fn parse_primary_expression(&mut self) -> ParserResult<Expression> {
        match self.current_token {
            Some(token) => match &token.token {
                Token::Identifier => {
                    let name = token.slice.clone();
                    self.advance();

                    if self.current_token_matches(&[Token::LeftBrace]) {
                        // Pattern or assembly declaration
                        if name == "pattern" {
                            Ok(Expression::Pattern(self.parse_pattern_expr()?))
                        } else if name == "assembly" {
                            Ok(Expression::Assembly(self.parse_assembly_expr()?))
                        } else {
                            Ok(Expression::Variable(name))
                        }
                    } else if self.current_token_matches(&[Token::LeftParen]) {
                        // Function call
                        Ok(Expression::FunctionCall(self.parse_function_call(name)?))
                    } else {
                        Ok(Expression::Variable(name))
                    }
                }
                Token::Float => {
                    let value = self.parse_float()?;
                    Ok(Expression::Literal(Literal {
                        span: token.span,
                        value: LiteralValue::Float(value),
                    }))
                }
                Token::String => {
                    let value = self.parse_string()?;
                    Ok(Expression::Literal(Literal {
                        span: token.span,
                        value: LiteralValue::String(value),
                    }))
                }
                Token::True => {
                    self.advance();
                    Ok(Expression::Literal(Literal {
                        span: token.span,
                        value: LiteralValue::Boolean(true),
                    }))
                }
                Token::False => {
                    self.advance();
                    Ok(Expression::Literal(Literal {
                        span: token.span,
                        value: LiteralValue::Boolean(false),
                    }))
                }
                Token::LeftBracket => {
                    Ok(Expression::List(self.parse_expression_list()?))
                }
                Token::LeftBrace => {
                    Ok(Expression::Map(self.parse_expression_map()?))
                }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "expression".to_string(),
                    found: token.token.to_string(),
                }),
            },
            None => Err(ParserError::UnexpectedEOF {
                span: Span::new(self.position, self.position, 1, 0),
            }),
        }
    }

    /// Parse pattern expression
    fn parse_pattern_expr(&mut self) -> ParserResult<PatternExpr> {
        let start_pos = self.current_position();

        let name = self.parse_identifier()?;
        self.expect_token(Token::LeftBrace)?;
        let body = self.parse_pattern_body()?;
        self.expect_token(Token::RightBrace)?;

        let end_pos = self.current_position();
        Ok(PatternExpr {
            span: Span::new(start_pos, end_pos, 1, 0),
            name,
            body: Some(body),
        })
    }

    /// Parse assembly expression
    fn parse_assembly_expr(&mut self) -> ParserResult<AssemblyExpr> {
        let start_pos = self.current_position();

        let name = self.parse_identifier()?;
        self.expect_token(Token::LeftBrace)?;
        let body = self.parse_assembly_body()?;
        self.expect_token(Token::RightBrace)?;

        let end_pos = self.current_position();
        Ok(AssemblyExpr {
            span: Span::new(start_pos, end_pos, 1, 0),
            name,
            body: Some(body),
        })
    }

    /// Parse function call
    fn parse_function_call(&mut self, name: String) -> ParserResult<FunctionCall> {
        let start_pos = self.current_position();

        self.expect_token(Token::LeftParen)?;
        let mut arguments = Vec::new();

        if !self.current_token_matches(&[Token::RightParen]) {
            arguments.push(self.parse_expression()?);
            while self.current_token_matches(&[Token::Comma]) {
                self.advance();
                arguments.push(self.parse_expression()?);
            }
        }

        self.expect_token(Token::RightParen)?;
        let end_pos = self.current_position();

        Ok(FunctionCall {
            span: Span::new(start_pos, end_pos, 1, 0),
            name,
            arguments,
        })
    }

    /// Parse expression list
    fn parse_expression_list(&mut self) -> ParserResult<Vec<Expression>> {
        let mut expressions = Vec::new();

        if !self.current_token_matches(&[Token::RightBracket]) {
            expressions.push(self.parse_expression()?);
            while self.current_token_matches(&[Token::Comma]) {
                self.advance();
                expressions.push(self.parse_expression()?);
            }
        }

        Ok(expressions)
    }

    /// Parse expression map
    fn parse_expression_map(&mut self) -> ParserResult<HashMap<String, Expression>> {
        let mut map = HashMap::new();

        if !self.current_token_matches(&[Token::RightBrace]) {
            let key = self.parse_string()?;
            self.expect_token(Token::Colon)?;
            let value = self.parse_expression()?;
            map.insert(key, value);

            while self.current_token_matches(&[Token::Comma]) {
                self.advance();
                let key = self.parse_string()?;
                self.expect_token(Token::Colon)?;
                let value = self.parse_expression()?;
                map.insert(key, value);
            }
        }

        Ok(map)
    }

    /// Parse conditions
    fn parse_conditions(&mut self) -> ParserResult<Vec<Condition>> {
        let mut conditions = Vec::new();

        if !self.current_token_matches(&[Token::Causation]) {
            conditions.push(self.parse_condition()?);
            while self.current_token_matches(&[Token::LogicalAnd, Token::LogicalOr]) {
                // Simplified for now
                break;
            }
        }

        Ok(conditions)
    }

    /// Parse condition
    fn parse_condition(&mut self) -> ParserResult<Condition> {
        // Simplified condition parsing
        if self.current_token_matches(&[Token::Identifier]) {
            let token = self.current_token.as_ref().unwrap();
            match token.slice.as_str() {
                "Δt" => {
                    self.advance();
                    self.expect_token(Token::LeftParen)?;
                    let neuron1 = self.parse_expression()?;
                    self.expect_token(Token::Comma)?;
                    let neuron2 = self.parse_expression()?;
                    self.expect_token(Token::RightParen)?;

                    Ok(Condition::Temporal(TemporalCondition {
                        span: token.span,
                        neuron1,
                        neuron2,
                        operator: ComparisonOp::Less,
                        duration: Duration { value: 1.0, unit: TimeUnit::Milliseconds },
                    }))
                }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "condition".to_string(),
                    found: token.token.to_string(),
                }),
            }
        } else {
            Err(ParserError::UnexpectedToken {
                span: self.current_position_span(),
                expected: "condition".to_string(),
                found: self.current_token.unwrap().token.to_string(),
            })
        }
    }

    /// Parse learning rule
    fn parse_learning_rule(&mut self) -> ParserResult<LearningRule> {
        if self.current_token_matches(&[Token::STDP]) {
            self.advance();
            Ok(LearningRule::STDP(self.parse_stdp_params()?))
        } else if self.current_token_matches(&[Token::Hebbian]) {
            self.advance();
            Ok(LearningRule::Hebbian(self.parse_hebbian_params()?))
        } else {
            Err(ParserError::UnexpectedToken {
                span: self.current_position_span(),
                expected: "learning rule".to_string(),
                found: self.current_token.unwrap().token.to_string(),
            })
        }
    }

    /// Parse STDP parameters
    fn parse_stdp_params(&mut self) -> ParserResult<STDPParams> {
        let start_pos = self.current_position();

        self.expect_token(Token::With)?;
        self.expect_token(Token::LeftBrace)?;

        let mut a_plus = None;
        let mut a_minus = None;
        let mut tau_plus = None;
        let mut tau_minus = None;

        while !self.current_token_matches(&[Token::RightBrace]) && !self.is_at_end() {
            let param_name = self.parse_identifier()?;
            self.expect_token(Token::Colon)?;

            match param_name.as_str() {
                "A_plus" => a_plus = Some(self.parse_float()?),
                "A_minus" => a_minus = Some(self.parse_float()?),
                "tau_plus" => tau_plus = Some(self.parse_duration()?),
                "tau_minus" => tau_minus = Some(self.parse_duration()?),
                _ => return Err(ParserError::InvalidSyntax {
                    span: self.current_position_span(),
                    message: format!("Unknown STDP parameter: {}", param_name),
                }),
            }

            if !self.current_token_matches(&[Token::RightBrace]) {
                self.expect_token(Token::Comma)?;
            }
        }

        self.expect_token(Token::RightBrace)?;
        let end_pos = self.current_position();

        Ok(STDPParams {
            span: Span::new(start_pos, end_pos, 1, 0),
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
        })
    }

    /// Parse Hebbian parameters
    fn parse_hebbian_params(&mut self) -> ParserResult<HebbianParams> {
        let start_pos = self.current_position();

        self.expect_token(Token::With)?;
        self.expect_token(Token::LeftBrace)?;

        let mut learning_rate = None;
        let mut threshold = None;
        let mut soft_bound = None;

        while !self.current_token_matches(&[Token::RightBrace]) && !self.is_at_end() {
            let param_name = self.parse_identifier()?;
            self.expect_token(Token::Colon)?;

            match param_name.as_str() {
                "learning_rate" => learning_rate = Some(self.parse_float()?),
                "threshold" => threshold = Some(self.parse_float()?),
                "soft_bound" => soft_bound = Some(self.parse_float()?),
                _ => return Err(ParserError::InvalidSyntax {
                    span: self.current_position_span(),
                    message: format!("Unknown Hebbian parameter: {}", param_name),
                }),
            }

            if !self.current_token_matches(&[Token::RightBrace]) {
                self.expect_token(Token::Comma)?;
            }
        }

        self.expect_token(Token::RightBrace)?;
        let end_pos = self.current_position();

        Ok(HebbianParams {
            span: Span::new(start_pos, end_pos, 1, 0),
            learning_rate,
            threshold,
            soft_bound,
        })
    }

    /// Parse evolution strategy
    fn parse_evolution_strategy(&mut self) -> ParserResult<EvolutionStrategy> {
        let start_pos = self.current_position();

        self.expect_token(Token::With)?;
        self.expect_token(Token::LeftBrace)?;

        // Simplified - just parse the strategy type
        let strategy_type = if self.current_token_matches(&[Token::Identifier]) {
            let token = self.current_token.as_ref().unwrap();
            match token.slice.as_str() {
                "genetic" => EvolutionType::Genetic(GeneticParams {
                    population_size: None,
                    mutation_rate: None,
                    crossover_rate: None,
                }),
                "gradient" => EvolutionType::Gradient(GradientParams {
                    learning_rate: None,
                    momentum: None,
                    decay: None,
                }),
                "random" => EvolutionType::Random(RandomParams {
                    exploration: None,
                    temperature: None,
                }),
                _ => return Err(ParserError::InvalidSyntax {
                    span: token.span,
                    message: format!("Unknown evolution strategy: {}", token.slice),
                }),
            }
        } else {
            EvolutionType::Random(RandomParams {
                exploration: None,
                temperature: None,
            })
        };

        self.expect_token(Token::RightBrace)?;
        let end_pos = self.current_position();

        Ok(EvolutionStrategy {
            span: Span::new(start_pos, end_pos, 1, 0),
            strategy_type,
        })
    }

    /// Parse monitoring specification
    fn parse_monitoring_spec(&mut self) -> ParserResult<MonitoringSpec> {
        let start_pos = self.current_position();

        self.expect_token(Token::LeftBrace)?;
        let mut metrics = Vec::new();

        while !self.current_token_matches(&[Token::RightBrace]) && !self.is_at_end() {
            let metric_name = self.parse_identifier()?;
            self.expect_token(Token::Colon)?;

            let metric_type = if self.current_token_matches(&[Token::Identifier]) {
                let token = self.current_token.as_ref().unwrap();
                match token.slice.as_str() {
                    "histogram" => MetricType::Histogram,
                    "gauge" => MetricType::Gauge,
                    "counter" => MetricType::Counter,
                    _ => MetricType::Gauge,
                }
            } else {
                MetricType::Gauge
            };

            metrics.push(MetricSpec {
                name: metric_name,
                metric_type,
            });

            if !self.current_token_matches(&[Token::RightBrace]) {
                self.expect_token(Token::Comma)?;
            }
        }

        self.expect_token(Token::RightBrace)?;
        let end_pos = self.current_position();

        Ok(MonitoringSpec {
            span: Span::new(start_pos, end_pos, 1, 0),
            metrics,
        })
    }

    /// Parse type expression
    fn parse_type_expression(&mut self) -> ParserResult<TypeExpression> {
        // Enhanced type expression parsing with temporal and topological types
        if self.current_token_matches(&[Token::Identifier]) {
            let token = self.current_token.as_ref().unwrap();
            let type_name = token.slice.clone();
            self.advance();

            match type_name.as_str() {
                "SpikeTrain" => self.parse_spike_train_type(token.span),
                "TimingWindow" => self.parse_timing_window_type(token.span),
                "BurstPattern" => self.parse_burst_pattern_type(token.span),
                "Rhythm" => self.parse_rhythm_type(token.span),
                "FeedForwardNetwork" => self.parse_feedforward_network_type(token.span),
                "RecurrentNetwork" => self.parse_recurrent_network_type(token.span),
                "ModularNetwork" => self.parse_modular_network_type(token.span),
                "SmallWorldNetwork" => self.parse_small_world_network_type(token.span),
                "ScaleFreeNetwork" => self.parse_scale_free_network_type(token.span),
                "LIF" => self.parse_lif_type(token.span),
                "Izhikevich" => self.parse_izhikevich_type(token.span),
                "HodgkinHuxley" => self.parse_hodgkin_huxley_type(token.span),
                "AdaptiveExponential" => self.parse_adaptive_exponential_type(token.span),
                "Chemical" => self.parse_chemical_synapse_type(token.span),
                "Electrical" => self.parse_electrical_synapse_type(token.span),
                "Plastic" => self.parse_plastic_synapse_type(token.span),
                "Modulatory" => self.parse_modulatory_synapse_type(token.span),
                _ => Ok(TypeExpression::Base(BaseType {
                    span: token.span,
                    type_name,
                })),
            }
        } else if self.current_token_matches(&[Token::LeftParen]) {
            // Handle dependent types like (x: Type) -> Result
            self.parse_dependent_type_expression()
        } else {
            Ok(TypeExpression::Variable("unknown".to_string()))
        }
    }

    /// Parse dependent type expression
    fn parse_dependent_type_expression(&mut self) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftParen)?;
        let param_name = self.parse_identifier()?;
        self.expect_token(Token::Colon)?;
        let param_type = self.parse_type_expression()?;
        self.expect_token(Token::RightParen)?;
        self.expect_token(Token::Arrow)?;
        let return_type = self.parse_type_expression()?;

        Ok(TypeExpression::Dependent(
            param_name,
            Box::new(param_type),
            Box::new(return_type),
        ))
    }

    /// Parse temporal type expressions

    fn parse_spike_train_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let duration = self.parse_duration()?;
        let frequency = if self.current_token_matches(&[Token::Comma]) {
            self.advance();
            Some(self.parse_frequency()?)
        } else {
            None
        };
        let regularity = if self.current_token_matches(&[Token::Comma]) {
            self.advance();
            Some(self.parse_regularity_constraint()?)
        } else {
            None
        };
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::Temporal(Box::new(TemporalType::SpikeTrain {
            duration,
            frequency,
            regularity,
        })))
    }

    fn parse_timing_window_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let min_delay = self.parse_duration()?;
        self.expect_token(Token::Comma)?;
        let max_delay = self.parse_duration()?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::Temporal(Box::new(TemporalType::TimingWindow {
            min_delay,
            max_delay,
        })))
    }

    fn parse_burst_pattern_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let spike_count = self.parse_integer()? as usize;
        self.expect_token(Token::Comma)?;
        let inter_spike_interval = self.parse_duration()?;
        let tolerance = if self.current_token_matches(&[Token::Comma]) {
            self.advance();
            Some(self.parse_duration()?)
        } else {
            None
        };
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::Temporal(Box::new(TemporalType::BurstPattern {
            spike_count,
            inter_spike_interval,
            tolerance,
        })))
    }

    fn parse_rhythm_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let period = self.parse_duration()?;
        let jitter_tolerance = if self.current_token_matches(&[Token::Comma]) {
            self.advance();
            Some(self.parse_duration()?)
        } else {
            None
        };
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::Temporal(Box::new(TemporalType::Rhythm {
            period,
            jitter_tolerance,
        })))
    }

    /// Parse topological type expressions

    fn parse_feedforward_network_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let density = self.parse_float()?;
        self.expect_token(Token::Comma)?;
        self.expect_token(Token::LeftBracket)?;
        let mut layers = Vec::new();
        layers.push(self.parse_integer()? as usize);
        while self.current_token_matches(&[Token::Comma]) {
            self.advance();
            layers.push(self.parse_integer()? as usize);
        }
        self.expect_token(Token::RightBracket)?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::Topological(Box::new(TopologicalType::FeedForwardNetwork {
            density,
            layers,
        })))
    }

    fn parse_recurrent_network_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let reservoir_size = self.parse_integer()? as usize;
        self.expect_token(Token::Comma)?;
        let connectivity = self.parse_connectivity_pattern()?;
        let spectral_radius = if self.current_token_matches(&[Token::Comma]) {
            self.advance();
            Some(self.parse_float()?)
        } else {
            None
        };
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::Topological(Box::new(TopologicalType::RecurrentNetwork {
            reservoir_size,
            connectivity,
            spectral_radius,
        })))
    }

    fn parse_modular_network_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        self.expect_token(Token::LeftBracket)?;
        let mut modules = Vec::new();
        modules.push(self.parse_module_spec()?);
        while self.current_token_matches(&[Token::Comma]) {
            self.advance();
            modules.push(self.parse_module_spec()?);
        }
        self.expect_token(Token::RightBracket)?;
        self.expect_token(Token::Comma)?;
        self.expect_token(Token::LeftBracket)?;
        let mut inter_module_connections = Vec::new();
        if !self.current_token_matches(&[Token::RightBracket]) {
            inter_module_connections.push(self.parse_inter_module_connection()?);
            while self.current_token_matches(&[Token::Comma]) {
                self.advance();
                inter_module_connections.push(self.parse_inter_module_connection()?);
            }
        }
        self.expect_token(Token::RightBracket)?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::Topological(Box::new(TopologicalType::ModularNetwork {
            modules,
            inter_module_connections,
        })))
    }

    fn parse_small_world_network_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let clustering_coefficient = self.parse_float()?;
        self.expect_token(Token::Comma)?;
        let average_path_length = self.parse_float()?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::Topological(Box::new(TopologicalType::SmallWorldNetwork {
            clustering_coefficient,
            average_path_length,
        })))
    }

    fn parse_scale_free_network_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let power_law_exponent = self.parse_float()?;
        self.expect_token(Token::Comma)?;
        let min_degree = self.parse_integer()? as usize;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::Topological(Box::new(TopologicalType::ScaleFreeNetwork {
            power_law_exponent,
            min_degree,
        })))
    }

    /// Parse neural type expressions

    fn parse_lif_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let time_constant = self.parse_duration()?;
        self.expect_token(Token::Comma)?;
        let rest_potential = self.parse_voltage()?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::MembraneDynamics(MembraneType::LIF {
            time_constant,
            rest_potential,
        }))
    }

    fn parse_izhikevich_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let a = self.parse_float()?;
        self.expect_token(Token::Comma)?;
        let b = self.parse_float()?;
        self.expect_token(Token::Comma)?;
        let c = self.parse_voltage()?;
        self.expect_token(Token::Comma)?;
        let d = self.parse_float()?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::MembraneDynamics(MembraneType::Izhikevich { a, b, c, d }))
    }

    fn parse_hodgkin_huxley_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let sodium_conductance = self.parse_conductance()?;
        self.expect_token(Token::Comma)?;
        let potassium_conductance = self.parse_conductance()?;
        self.expect_token(Token::Comma)?;
        let leak_conductance = self.parse_conductance()?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::MembraneDynamics(MembraneType::HodgkinHuxley {
            sodium_conductance,
            potassium_conductance,
            leak_conductance,
        }))
    }

    fn parse_adaptive_exponential_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let adaptation_time_constant = self.parse_duration()?;
        self.expect_token(Token::Comma)?;
        let adaptation_increment = self.parse_conductance()?;
        self.expect_token(Token::Comma)?;
        let spike_triggered_increment = self.parse_current()?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::MembraneDynamics(MembraneType::AdaptiveExponential {
            adaptation_time_constant,
            adaptation_increment,
            spike_triggered_increment,
        }))
    }

    /// Parse synaptic type expressions

    fn parse_chemical_synapse_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let receptor_type = self.parse_receptor_type()?;
        self.expect_token(Token::Comma)?;
        let time_constant = self.parse_duration()?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::SynapticWeight(SynapticType::Chemical {
            receptor_type,
            time_constant,
        }))
    }

    fn parse_electrical_synapse_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let gap_junction_conductance = self.parse_conductance()?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::SynapticWeight(SynapticType::Electrical {
            gap_junction_conductance,
        }))
    }

    fn parse_plastic_synapse_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let learning_rule = self.parse_learning_rule()?;
        self.expect_token(Token::Comma)?;
        let potentiation_amplitude = self.parse_float()?;
        self.expect_token(Token::Comma)?;
        let depression_amplitude = self.parse_float()?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::SynapticWeight(SynapticType::Plastic {
            learning_rule,
            potentiation_amplitude,
            depression_amplitude,
        }))
    }

    fn parse_modulatory_synapse_type(&mut self, span: Span) -> ParserResult<TypeExpression> {
        self.expect_token(Token::LeftBracket)?;
        let modulator_type = self.parse_string()?;
        self.expect_token(Token::Comma)?;
        let gain_factor = self.parse_float()?;
        self.expect_token(Token::RightBracket)?;

        Ok(TypeExpression::SynapticWeight(SynapticType::Modulatory {
            modulator_type,
            gain_factor,
        }))
    }

    /// Helper parsing methods for type system

    fn parse_frequency(&mut self) -> ParserResult<Frequency> {
        let value = self.parse_float()?;
        let unit = if self.current_token_matches(&[Token::Identifier]) {
            let unit_str = self.parse_identifier()?;
            match unit_str.as_str() {
                "Hz" => FrequencyUnit::Hertz,
                "kHz" => FrequencyUnit::Kilohertz,
                _ => FrequencyUnit::Hertz,
            }
        } else {
            FrequencyUnit::Hertz
        };

        Ok(Frequency { value, unit })
    }

    fn parse_regularity_constraint(&mut self) -> ParserResult<RegularityConstraint> {
        if self.current_token_matches(&[Token::Identifier]) {
            let token = self.current_token.as_ref().unwrap();
            match token.slice.as_str() {
                "regular" => {
                    self.advance();
                    self.expect_token(Token::LeftBrace)?;
                    self.expect_token(Token::RightBrace)?; // Simplified
                    Ok(RegularityConstraint::Regular {
                        jitter: Duration { span: token.span, value: 0.0, unit: TimeUnit::Milliseconds },
                    })
                }
                "irregular" => {
                    self.advance();
                    self.expect_token(Token::LeftBrace)?;
                    let coefficient_of_variation = self.parse_float()?;
                    self.expect_token(Token::RightBrace)?;
                    Ok(RegularityConstraint::Irregular { coefficient_of_variation })
                }
                "poisson" => {
                    self.advance();
                    self.expect_token(Token::LeftBrace)?;
                    let rate = self.parse_frequency()?;
                    self.expect_token(Token::RightBrace)?;
                    Ok(RegularityConstraint::Poisson { rate })
                }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "regularity constraint".to_string(),
                    found: token.token.to_string(),
                }),
            }
        } else {
            Err(ParserError::UnexpectedToken {
                span: self.current_position_span(),
                expected: "regularity constraint".to_string(),
                found: self.current_token.unwrap().token.to_string(),
            })
        }
    }

    fn parse_connectivity_pattern(&mut self) -> ParserResult<ConnectivityPattern> {
        if self.current_token_matches(&[Token::Identifier]) {
            let token = self.current_token.as_ref().unwrap();
            match token.slice.as_str() {
                "dense" => {
                    self.advance();
                    Ok(ConnectivityPattern::Dense)
                }
                "sparse" => {
                    self.advance();
                    self.expect_token(Token::LeftBrace)?;
                    let density = self.parse_float()?;
                    self.expect_token(Token::RightBrace)?;
                    Ok(ConnectivityPattern::Sparse { density })
                }
                "local" => {
                    self.advance();
                    self.expect_token(Token::LeftBrace)?;
                    let radius = self.parse_float()?;
                    self.expect_token(Token::RightBrace)?;
                    Ok(ConnectivityPattern::Local { radius })
                }
                _ => Ok(ConnectivityPattern::Dense),
            }
        } else {
            Ok(ConnectivityPattern::Dense)
        }
    }

    fn parse_module_spec(&mut self) -> ParserResult<ModuleSpec> {
        self.expect_token(Token::LeftBrace)?;
        let name = self.parse_string()?;
        self.expect_token(Token::Comma)?;
        let size = self.parse_integer()? as usize;
        self.expect_token(Token::Comma)?;
        let internal_connectivity = self.parse_connectivity_pattern()?;
        self.expect_token(Token::RightBrace)?;

        Ok(ModuleSpec {
            span: Span::new(0, 0, 0, 0), // Would need proper span tracking
            name,
            size,
            internal_connectivity,
        })
    }

    fn parse_inter_module_connection(&mut self) -> ParserResult<InterModuleConnection> {
        self.expect_token(Token::LeftBrace)?;
        let from_module = self.parse_string()?;
        self.expect_token(Token::Comma)?;
        let to_module = self.parse_string()?;
        self.expect_token(Token::Comma)?;
        let connection_type = self.parse_connectivity_pattern()?;
        let weight_range = if self.current_token_matches(&[Token::Comma]) {
            self.advance();
            self.expect_token(Token::LeftParen)?;
            let min = self.parse_float()?;
            self.expect_token(Token::Comma)?;
            let max = self.parse_float()?;
            self.expect_token(Token::RightParen)?;
            Some((min, max))
        } else {
            None
        };
        self.expect_token(Token::RightBrace)?;

        Ok(InterModuleConnection {
            span: Span::new(0, 0, 0, 0), // Would need proper span tracking
            from_module,
            to_module,
            connection_type,
            weight_range,
        })
    }

    fn parse_receptor_type(&mut self) -> ParserResult<ReceptorType> {
        if self.current_token_matches(&[Token::Identifier]) {
            let token = self.current_token.as_ref().unwrap();
            match token.slice.as_str() {
                "AMPA" => { self.advance(); Ok(ReceptorType::AMPA) }
                "NMDA" => { self.advance(); Ok(ReceptorType::NMDA) }
                "GABA_A" => { self.advance(); Ok(ReceptorType::GABA_A) }
                "GABA_B" => { self.advance(); Ok(ReceptorType::GABA_B) }
                "Dopamine" => { self.advance(); Ok(ReceptorType::Dopamine) }
                "Serotonin" => { self.advance(); Ok(ReceptorType::Serotonin) }
                "Acetylcholine" => { self.advance(); Ok(ReceptorType::Acetylcholine) }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "receptor type".to_string(),
                    found: token.token.to_string(),
                }),
            }
        } else {
            Err(ParserError::UnexpectedToken {
                span: self.current_position_span(),
                expected: "receptor type".to_string(),
                found: self.current_token.unwrap().token.to_string(),
            })
        }
    }

    fn parse_conductance(&mut self) -> ParserResult<Conductance> {
        let value = self.parse_float()?;
        let unit = if self.current_token_matches(&[Token::Identifier]) {
            let unit_str = self.parse_identifier()?;
            match unit_str.as_str() {
                "S" => ConductanceUnit::Siemens,
                "mS" => ConductanceUnit::Millisiemens,
                "uS" => ConductanceUnit::Microsiemens,
                "nS" => ConductanceUnit::Nanosiemens,
                "pS" => ConductanceUnit::Picosiemens,
                _ => ConductanceUnit::Siemens,
            }
        } else {
            ConductanceUnit::Siemens
        };

        Ok(Conductance { value, unit })
    }

    fn parse_current(&mut self) -> ParserResult<Current> {
        let value = self.parse_float()?;
        let unit = if self.current_token_matches(&[Token::Identifier]) {
            let unit_str = self.parse_identifier()?;
            match unit_str.as_str() {
                "A" => CurrentUnit::Amperes,
                "mA" => CurrentUnit::Milliamperes,
                "uA" => CurrentUnit::Microamperes,
                "nA" => CurrentUnit::Nanoamperes,
                "pA" => CurrentUnit::Picoamperes,
                _ => CurrentUnit::Amperes,
            }
        } else {
            CurrentUnit::Amperes
        };

        Ok(Current { value, unit })
    }

    fn parse_integer(&mut self) -> ParserResult<i64> {
        match self.current_token {
            Some(token) => match &token.token {
                Token::Integer => {
                    let value = token.slice.parse().unwrap_or(0);
                    self.advance();
                    Ok(value)
                }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "integer".to_string(),
                    found: token.token.to_string(),
                }),
            },
            None => Err(ParserError::UnexpectedEOF {
                span: Span::new(self.position, self.position, 1, 0),
            }),
        }
    }

    /// Parse plasticity rule
    fn parse_plasticity_rule(&mut self) -> ParserResult<PlasticityRule> {
        let rule = self.parse_learning_rule()?;
        Ok(PlasticityRule {
            span: Span::new(0, 0, 1, 0), // Simplified
            rule_type: rule,
        })
    }

    /// Parse connection specifications
    fn parse_connection_specs(&mut self) -> ParserResult<Vec<ConnectionSpec>> {
        let mut specs = Vec::new();

        // Simplified for now
        Ok(specs)
    }

    /// Parse plasticity rules
    fn parse_plasticity_rules(&mut self) -> ParserResult<Vec<PlasticityRule>> {
        let mut rules = Vec::new();

        // Simplified for now
        Ok(rules)
    }

    /// Parse spike events
    fn parse_spike_events(&mut self) -> ParserResult<Vec<SpikeEvent>> {
        let mut events = Vec::new();

        // Simplified for now
        Ok(events)
    }

    /// Helper parsing methods

    fn parse_identifier(&mut self) -> ParserResult<String> {
        match self.current_token {
            Some(token) => match &token.token {
                Token::Identifier => {
                    let name = token.slice.clone();
                    self.advance();
                    Ok(name)
                }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "identifier".to_string(),
                    found: token.token.to_string(),
                }),
            },
            None => Err(ParserError::UnexpectedEOF {
                span: Span::new(self.position, self.position, 1, 0),
            }),
        }
    }

    fn parse_string(&mut self) -> ParserResult<String> {
        match self.current_token {
            Some(token) => match &token.token {
                Token::String => {
                    let value = token.slice.clone();
                    self.advance();
                    Ok(value)
                }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "string".to_string(),
                    found: token.token.to_string(),
                }),
            },
            None => Err(ParserError::UnexpectedEOF {
                span: Span::new(self.position, self.position, 1, 0),
            }),
        }
    }

    fn parse_float(&mut self) -> ParserResult<f64> {
        match self.current_token {
            Some(token) => match &token.token {
                Token::Float => {
                    let value = token.slice.parse().unwrap_or(0.0);
                    self.advance();
                    Ok(value)
                }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "float".to_string(),
                    found: token.token.to_string(),
                }),
            },
            None => Err(ParserError::UnexpectedEOF {
                span: Span::new(self.position, self.position, 1, 0),
            }),
        }
    }

    fn parse_boolean(&mut self) -> ParserResult<bool> {
        match self.current_token {
            Some(token) => match &token.token {
                Token::True => {
                    self.advance();
                    Ok(true)
                }
                Token::False => {
                    self.advance();
                    Ok(false)
                }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "boolean".to_string(),
                    found: token.token.to_string(),
                }),
            },
            None => Err(ParserError::UnexpectedEOF {
                span: Span::new(self.position, self.position, 1, 0),
            }),
        }
    }

    fn parse_weight(&mut self) -> ParserResult<Weight> {
        let value = self.parse_float()?;
        Ok(Weight { value })
    }

    fn parse_duration(&mut self) -> ParserResult<Duration> {
        match self.current_token {
            Some(token) => match &token.token {
                Token::Duration => {
                    let value_str = token.slice.clone();
                    self.advance();

                    // Parse duration value and unit
                    let value = value_str.chars()
                        .take_while(|c| c.is_ascii_digit() || *c == '.')
                        .collect::<String>()
                        .parse()
                        .unwrap_or(1.0);

                    let unit = if value_str.ends_with("ms") {
                        TimeUnit::Milliseconds
                    } else if value_str.ends_with("s") {
                        TimeUnit::Seconds
                    } else if value_str.ends_with("us") || value_str.ends_with("μs") {
                        TimeUnit::Microseconds
                    } else if value_str.ends_with("ns") {
                        TimeUnit::Nanoseconds
                    } else {
                        TimeUnit::Milliseconds
                    };

                    Ok(Duration { value, unit })
                }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "duration".to_string(),
                    found: token.token.to_string(),
                }),
            },
            None => Err(ParserError::UnexpectedEOF {
                span: Span::new(self.position, self.position, 1, 0),
            }),
        }
    }

    fn parse_voltage(&mut self) -> ParserResult<Voltage> {
        match self.current_token {
            Some(token) => match &token.token {
                Token::Voltage => {
                    let value_str = token.slice.clone();
                    self.advance();

                    let value = value_str.chars()
                        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                        .collect::<String>()
                        .parse()
                        .unwrap_or(0.0);

                    let unit = if value_str.ends_with("mV") {
                        VoltageUnit::Millivolts
                    } else if value_str.ends_with("V") {
                        VoltageUnit::Volts
                    } else if value_str.ends_with("uV") || value_str.ends_with("μV") {
                        VoltageUnit::Microvolts
                    } else {
                        VoltageUnit::Millivolts
                    };

                    Ok(Voltage { value, unit })
                }
                _ => Err(ParserError::UnexpectedToken {
                    span: token.span,
                    expected: "voltage".to_string(),
                    found: token.token.to_string(),
                }),
            },
            None => Err(ParserError::UnexpectedEOF {
                span: Span::new(self.position, self.position, 1, 0),
            }),
        }
    }

    fn parse_voltage_per_time(&mut self) -> ParserResult<VoltagePerTime> {
        let voltage = self.parse_voltage()?;
        self.expect_token(Token::Causation)?;
        let time = self.parse_duration()?;

        Ok(VoltagePerTime { voltage, time })
    }

    fn parse_position_3d(&mut self) -> ParserResult<Position3D> {
        self.expect_token(Token::LeftParen)?;
        let x = self.parse_float()?;
        self.expect_token(Token::Comma)?;
        let y = self.parse_float()?;
        self.expect_token(Token::Comma)?;
        let z = self.parse_float()?;
        self.expect_token(Token::RightParen)?;

        Ok(Position3D { x, y, z })
    }

    /// Advance to next token
    fn advance(&mut self) {
        if let Some(_) = self.tokens.next() {
            self.position += 1;
            self.current_token = self.tokens.peek().copied();
        } else {
            self.current_token = None;
        }
    }

    /// Check if current token matches any of the expected tokens
    fn current_token_matches(&mut self, expected: &[Token]) -> bool {
        if let Some(current) = self.current_token {
            expected.iter().any(|token| std::mem::discriminant(&current.token) == std::mem::discriminant(token))
        } else {
            false
        }
    }

    /// Expect a specific token, return error if not found
    fn expect_token(&mut self, expected: Token) -> ParserResult<()> {
        if let Some(current) = self.current_token {
            if std::mem::discriminant(&current.token) == std::mem::discriminant(&expected) {
                self.advance();
                Ok(())
            } else {
                Err(ParserError::UnexpectedToken {
                    span: current.span,
                    expected: expected.to_string(),
                    found: current.token.to_string(),
                })
            }
        } else {
            Err(ParserError::UnexpectedEOF {
                span: Span::new(self.position, self.position, 1, 0),
            })
        }
    }

    /// Check if we've reached the end of input
    fn is_at_end(&self) -> bool {
        self.current_token.is_none()
    }

    /// Get current position for span creation
    fn current_position(&self) -> usize {
        self.position
    }

    /// Get current position as span
    fn current_position_span(&self) -> Span {
        Span::new(self.position, self.position, 1, 0)
    }
}

/// Convenience function to parse tokens
pub fn parse(tokens: &[SpannedToken]) -> ParserResult<Program> {
    let mut parser = Parser::new(tokens);
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;

    #[test]
    fn test_parse_neuron_declaration() {
        let source = "∴ test_neuron { threshold: -50mV, leak: 10mV/ms }";
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();

        assert!(program.declarations.len() >= 1);
    }

    #[test]
    fn test_parse_synapse_connection() {
        let source = "neuron₁ ⊸0.5:2ms⊸ neuron₂";
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();

        assert!(program.declarations.len() >= 1);
    }

    #[test]
    fn test_parse_topology_header() {
        let source = "topology ⟪test⟫ { precision: double }";
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();

        assert!(program.header.is_some());
        assert_eq!(program.header.unwrap().name, "test");
    }
}