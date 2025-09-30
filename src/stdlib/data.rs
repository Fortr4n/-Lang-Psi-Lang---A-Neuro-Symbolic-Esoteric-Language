//! # Data Import/Export and Serialization Library
//!
//! Tools for importing, exporting, and serializing neural networks and data.
//! Supports various formats and external system integration.

use crate::runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};

/// Data library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Data Import/Export Library");
    Ok(())
}

/// Neural Network Serializer
pub struct NetworkSerializer;

impl NetworkSerializer {
    /// Serialize network to JSON
    pub fn to_json(network: &RuntimeNetwork) -> Result<String, String> {
        serde_json::to_string_pretty(network)
            .map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Deserialize network from JSON
    pub fn from_json(json_data: &str) -> Result<RuntimeNetwork, String> {
        serde_json::from_str(json_data)
            .map_err(|e| format!("JSON deserialization failed: {}", e))
    }

    /// Serialize network to binary format
    pub fn to_binary(network: &RuntimeNetwork) -> Result<Vec<u8>, String> {
        bincode::serialize(network)
            .map_err(|e| format!("Binary serialization failed: {}", e))
    }

    /// Deserialize network from binary format
    pub fn from_binary(data: &[u8]) -> Result<RuntimeNetwork, String> {
        bincode::deserialize(data)
            .map_err(|e| format!("Binary deserialization failed: {}", e))
    }

    /// Save network to file
    pub fn save_to_file(network: &RuntimeNetwork, filepath: &str, format: SerializationFormat) -> Result<(), String> {
        let data = match format {
            SerializationFormat::Json => Self::to_json(network)?,
            SerializationFormat::Binary => {
                let binary_data = Self::to_binary(network)?;
                return fs::write(filepath, binary_data)
                    .map_err(|e| format!("Failed to write file: {}", e));
            }
            SerializationFormat::MessagePack => {
                let msgpack_data = rmp_serde::to_vec(network)
                    .map_err(|e| format!("MessagePack serialization failed: {}", e))?;
                return fs::write(filepath, msgpack_data)
                    .map_err(|e| format!("Failed to write file: {}", e));
            }
        };

        fs::write(filepath, data)
            .map_err(|e| format!("Failed to write file: {}", e))
    }

    /// Load network from file
    pub fn load_from_file(filepath: &str, format: SerializationFormat) -> Result<RuntimeNetwork, String> {
        let data = fs::read(filepath)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        match format {
            SerializationFormat::Json => Self::from_json(&String::from_utf8_lossy(&data)),
            SerializationFormat::Binary => Self::from_binary(&data),
            SerializationFormat::MessagePack => {
                rmp_serde::from_slice(&data)
                    .map_err(|e| format!("MessagePack deserialization failed: {}", e))
            }
        }
    }
}

/// Serialization formats
#[derive(Debug, Clone)]
pub enum SerializationFormat {
    Json,
    Binary,
    MessagePack,
}

/// Data Import/Export System
pub struct DataManager {
    importers: HashMap<String, Box<dyn DataImporter>>,
    exporters: HashMap<String, Box<dyn DataExporter>>,
    converters: HashMap<String, Box<dyn DataConverter>>,
}

impl DataManager {
    /// Create a new data manager
    pub fn new() -> Self {
        Self {
            importers: HashMap::new(),
            exporters: HashMap::new(),
            converters: HashMap::new(),
        }
    }

    /// Register data importer
    pub fn register_importer(&mut self, format: String, importer: Box<dyn DataImporter>) {
        self.importers.insert(format, importer);
    }

    /// Register data exporter
    pub fn register_exporter(&mut self, format: String, exporter: Box<dyn DataExporter>) {
        self.exporters.insert(format, exporter);
    }

    /// Register data converter
    pub fn register_converter(&mut self, from_format: String, to_format: String, converter: Box<dyn DataConverter>) {
        let key = format!("{}:{}", from_format, to_format);
        self.converters.insert(key, converter);
    }

    /// Import data from file
    pub fn import_data(&self, filepath: &str, format: &str) -> Result<ImportedData, String> {
        if let Some(importer) = self.importers.get(format) {
            importer.import(filepath)
        } else {
            Err(format!("No importer registered for format: {}", format))
        }
    }

    /// Export data to file
    pub fn export_data(&self, data: &ExportedData, filepath: &str, format: &str) -> Result<(), String> {
        if let Some(exporter) = self.exporters.get(format) {
            exporter.export(data, filepath)
        } else {
            Err(format!("No exporter registered for format: {}", format))
        }
    }

    /// Convert data between formats
    pub fn convert_data(&self, data: &ImportedData, from_format: &str, to_format: &str) -> Result<ImportedData, String> {
        let key = format!("{}:{}", from_format, to_format);

        if let Some(converter) = self.converters.get(&key) {
            converter.convert(data)
        } else {
            Err(format!("No converter registered for {} to {}", from_format, to_format))
        }
    }
}

/// Data importer trait
pub trait DataImporter {
    fn import(&self, filepath: &str) -> Result<ImportedData, String>;
    fn get_format_name(&self) -> String;
}

/// Data exporter trait
pub trait DataExporter {
    fn export(&self, data: &ExportedData, filepath: &str) -> Result<(), String>;
    fn get_format_name(&self) -> String;
}

/// Data converter trait
pub trait DataConverter {
    fn convert(&self, data: &ImportedData) -> Result<ImportedData, String>;
    fn get_conversion_name(&self) -> String;
}

/// Imported data structure
#[derive(Debug, Clone)]
pub struct ImportedData {
    pub data_type: DataType,
    pub metadata: HashMap<String, String>,
    pub content: DataContent,
}

/// Exported data structure
#[derive(Debug, Clone)]
pub struct ExportedData {
    pub data_type: DataType,
    pub metadata: HashMap<String, String>,
    pub content: DataContent,
}

/// Data types
#[derive(Debug, Clone)]
pub enum DataType {
    NeuralNetwork,
    TrainingData,
    SpikeData,
    ImageData,
    TextData,
    TimeSeries,
    Custom(String),
}

/// Data content
#[derive(Debug, Clone)]
pub enum DataContent {
    NeuralNetwork(RuntimeNetwork),
    TrainingDataset(Vec<TrainingExample>),
    SpikeEvents(Vec<RuntimeSpikeEvent>),
    ImageData(Vec<u8>),
    TextData(String),
    TimeSeries(Vec<(f64, f64)>),
    RawData(Vec<u8>),
}

/// Training example for data import/export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub inputs: Vec<f64>,
    pub targets: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

/// CSV Data Importer
pub struct CsvImporter;

impl DataImporter for CsvImporter {
    fn import(&self, filepath: &str) -> Result<ImportedData, String> {
        let content = fs::read_to_string(filepath)
            .map_err(|e| format!("Failed to read CSV file: {}", e))?;

        let mut lines = content.lines();
        let header = lines.next().ok_or("Empty CSV file")?;

        let column_names: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();

        let mut examples = Vec::new();
        let mut metadata = HashMap::new();

        for (line_idx, line) in lines.enumerate() {
            let values: Vec<f64> = line.split(',')
                .map(|s| s.trim().parse().unwrap_or(0.0))
                .collect();

            if values.len() >= 2 {
                let input_size = values.len() - 1;
                let inputs = values[..input_size].to_vec();
                let targets = values[input_size..].to_vec();

                examples.push(TrainingExample {
                    inputs,
                    targets,
                    metadata: HashMap::new(),
                });
            }
        }

        metadata.insert("rows".to_string(), examples.len().to_string());
        metadata.insert("columns".to_string(), column_names.len().to_string());
        metadata.insert("input_size".to_string(), (column_names.len() - 1).to_string());

        Ok(ImportedData {
            data_type: DataType::TrainingData,
            metadata,
            content: DataContent::TrainingDataset(examples),
        })
    }

    fn get_format_name(&self) -> String {
        "CSV".to_string()
    }
}

/// CSV Data Exporter
pub struct CsvExporter;

impl DataExporter for CsvExporter {
    fn export(&self, data: &ExportedData, filepath: &str) -> Result<(), String> {
        let mut csv_content = String::new();

        match &data.content {
            DataContent::TrainingDataset(examples) => {
                if examples.is_empty() {
                    return Err("No training examples to export".to_string());
                }

                // Create header
                csv_content.push_str("input_0");
                for i in 1..examples[0].inputs.len() {
                    csv_content.push_str(&format!(",input_{}", i));
                }
                for i in 0..examples[0].targets.len() {
                    csv_content.push_str(&format!(",target_{}", i));
                }
                csv_content.push('\n');

                // Add data rows
                for example in examples {
                    for (i, &input) in example.inputs.iter().enumerate() {
                        if i > 0 { csv_content.push(','); }
                        csv_content.push_str(&format!("{:.6}", input));
                    }
                    for (i, &target) in example.targets.iter().enumerate() {
                        csv_content.push(',');
                        csv_content.push_str(&format!("{:.6}", target));
                    }
                    csv_content.push('\n');
                }
            }
            _ => return Err("CSV export only supports training datasets".to_string()),
        }

        fs::write(filepath, csv_content)
            .map_err(|e| format!("Failed to write CSV file: {}", e))
    }

    fn get_format_name(&self) -> String {
        "CSV".to_string()
    }
}

/// JSON Data Importer
pub struct JsonImporter;

impl DataImporter for JsonImporter {
    fn import(&self, filepath: &str) -> Result<ImportedData, String> {
        let content = fs::read_to_string(filepath)
            .map_err(|e| format!("Failed to read JSON file: {}", e))?;

        // Try to parse as neural network first
        if let Ok(network) = serde_json::from_str::<RuntimeNetwork>(&content) {
            let mut metadata = HashMap::new();
            metadata.insert("type".to_string(), "neural_network".to_string());

            return Ok(ImportedData {
                data_type: DataType::NeuralNetwork,
                metadata,
                content: DataContent::NeuralNetwork(network),
            });
        }

        // Try to parse as training data
        if let Ok(examples) = serde_json::from_str::<Vec<TrainingExample>>(&content) {
            let mut metadata = HashMap::new();
            metadata.insert("type".to_string(), "training_data".to_string());
            metadata.insert("examples".to_string(), examples.len().to_string());

            return Ok(ImportedData {
                data_type: DataType::TrainingData,
                metadata,
                content: DataContent::TrainingDataset(examples),
            });
        }

        Err("Could not parse JSON as known data type".to_string())
    }

    fn get_format_name(&self) -> String {
        "JSON".to_string()
    }
}

/// JSON Data Exporter
pub struct JsonExporter;

impl DataExporter for JsonExporter {
    fn export(&self, data: &ExportedData, filepath: &str) -> Result<(), String> {
        let json_content = match &data.content {
            DataContent::NeuralNetwork(network) => {
                NetworkSerializer::to_json(network)?
            }
            DataContent::TrainingDataset(examples) => {
                serde_json::to_string_pretty(examples)
                    .map_err(|e| format!("Failed to serialize training data: {}", e))?
            }
            _ => return Err("JSON export not supported for this data type".to_string()),
        };

        fs::write(filepath, json_content)
            .map_err(|e| format!("Failed to write JSON file: {}", e))
    }

    fn get_format_name(&self) -> String {
        "JSON".to_string()
    }
}

/// Spike Data Format (SDF) Handler
pub struct SpikeDataHandler;

impl SpikeDataHandler {
    /// Export spike events to SDF format
    pub fn export_spike_data(spike_events: &[RuntimeSpikeEvent], filepath: &str) -> Result<(), String> {
        let mut sdf_content = String::new();

        // SDF header
        sdf_content.push_str("# Spike Data Format (SDF)\n");
        sdf_content.push_str(&format!("# Generated: {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")));
        sdf_content.push_str("# Columns: neuron_id, timestamp, amplitude\n");
        sdf_content.push_str("#\n");

        // Spike data
        for event in spike_events {
            sdf_content.push_str(&format!(
                "{}\t{:.6}\t{:.6}\n",
                event.neuron_id, event.timestamp, event.amplitude
            ));
        }

        fs::write(filepath, sdf_content)
            .map_err(|e| format!("Failed to write SDF file: {}", e))
    }

    /// Import spike events from SDF format
    pub fn import_spike_data(filepath: &str) -> Result<Vec<RuntimeSpikeEvent>, String> {
        let content = fs::read_to_string(filepath)
            .map_err(|e| format!("Failed to read SDF file: {}", e))?;

        let mut spike_events = Vec::new();
        let mut line_number = 0;

        for line in content.lines() {
            line_number += 1;

            // Skip comments and empty lines
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }

            // Parse spike data
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let neuron_id = parts[0].parse::<u32>().unwrap_or(0);
                let timestamp = parts[1].parse::<f64>().unwrap_or(0.0);
                let amplitude = parts[2].parse::<f64>().unwrap_or(0.0);

                spike_events.push(RuntimeSpikeEvent {
                    event_id: spike_events.len() as u64,
                    neuron_id: NeuronId(neuron_id),
                    timestamp,
                    amplitude,
                    event_type: SpikeEventType::Spike,
                });
            }
        }

        Ok(spike_events)
    }
}

/// Image Data Handler
pub struct ImageDataHandler;

impl ImageDataHandler {
    /// Load image data (placeholder implementation)
    pub fn load_image(filepath: &str) -> Result<ImageData, String> {
        // In a real implementation, this would use an image processing library
        let data = fs::read(filepath)
            .map_err(|e| format!("Failed to read image file: {}", e))?;

        Ok(ImageData {
            width: 224, // Placeholder
            height: 224, // Placeholder
            channels: 3,
            data,
            format: "RGB".to_string(),
        })
    }

    /// Save image data (placeholder implementation)
    pub fn save_image(image: &ImageData, filepath: &str) -> Result<(), String> {
        fs::write(filepath, &image.data)
            .map_err(|e| format!("Failed to write image file: {}", e))
    }
}

/// Image data structure
#[derive(Debug, Clone)]
pub struct ImageData {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<u8>,
    pub format: String,
}

/// Neuromorphic Data Format (NDF) Handler
pub struct NeuromorphicDataHandler;

impl NeuromorphicDataHandler {
    /// Export network in neuromorphic format
    pub fn export_neuromorphic_data(network: &RuntimeNetwork, filepath: &str) -> Result<(), String> {
        let mut ndf_content = String::new();

        // NDF header
        ndf_content.push_str("NDF 1.0\n");
        ndf_content.push_str(&format!("network_name: {}\n", network.metadata.name));
        ndf_content.push_str(&format!("neurons: {}\n", network.neurons.len()));
        ndf_content.push_str(&format!("synapses: {}\n", network.synapses.len()));
        ndf_content.push_str("\n");

        // Neuron data
        ndf_content.push_str("[neurons]\n");
        for (id, neuron) in &network.neurons {
            ndf_content.push_str(&format!(
                "{}: {} {:.3} {:.3} {:.3}\n",
                id.0, neuron.neuron_type, neuron.membrane_potential,
                neuron.parameters.threshold, neuron.parameters.resting_potential
            ));
        }
        ndf_content.push_str("\n");

        // Synapse data
        ndf_content.push_str("[synapses]\n");
        for (id, synapse) in &network.synapses {
            ndf_content.push_str(&format!(
                "{}: {} -> {} {:.3} {:.3}\n",
                id.0, synapse.presynaptic_id.0, synapse.postsynaptic_id.0,
                synapse.weight, synapse.delay.value
            ));
        }

        fs::write(filepath, ndf_content)
            .map_err(|e| format!("Failed to write NDF file: {}", e))
    }

    /// Import network from neuromorphic format
    pub fn import_neuromorphic_data(filepath: &str) -> Result<RuntimeNetwork, String> {
        let content = fs::read_to_string(filepath)
            .map_err(|e| format!("Failed to read NDF file: {}", e))?;

        // Parse NDF format (simplified implementation)
        let mut neurons = HashMap::new();
        let mut synapses = HashMap::new();

        let mut in_neurons_section = false;
        let mut in_synapses_section = false;

        for line in content.lines() {
            if line.starts_with("[neurons]") {
                in_neurons_section = true;
                in_synapses_section = false;
                continue;
            } else if line.starts_with("[synapses]") {
                in_neurons_section = false;
                in_synapses_section = true;
                continue;
            }

            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }

            if in_neurons_section {
                // Parse neuron data
                if let Some((id_str, data)) = line.split_once(':') {
                    if let Ok(id) = id_str.trim().parse::<u32>() {
                        // Simplified neuron creation
                        let neuron = RuntimeNeuron {
                            id: NeuronId(id),
                            name: format!("neuron_{}", id),
                            neuron_type: NeuronType::LIF,
                            parameters: NeuronParameters {
                                threshold: -50.0,
                                resting_potential: -70.0,
                                reset_potential: -80.0,
                                refractory_period: 2.0,
                                leak_rate: 0.1,
                            },
                            position: None,
                            membrane_potential: -70.0,
                            last_spike_time: None,
                            refractory_until: None,
                            incoming_spikes: Vec::new(),
                            activity_history: std::collections::VecDeque::new(),
                            incoming_synapse_ids: Vec::new(),
                            outgoing_synapse_ids: Vec::new(),
                        };
                        neurons.insert(NeuronId(id), neuron);
                    }
                }
            } else if in_synapses_section {
                // Parse synapse data
                if let Some((id_str, data)) = line.split_once(':') {
                    if let Ok(id) = id_str.trim().parse::<u32>() {
                        // Simplified synapse creation
                        let synapse = RuntimeSynapse {
                            id: SynapseId(id),
                            presynaptic_id: NeuronId(0), // Would parse from data
                            postsynaptic_id: NeuronId(1), // Would parse from data
                            weight: 0.5,
                            delay: Duration { value: 1.0 },
                            plasticity_rule: None,
                            last_presynaptic_spike: None,
                            last_postsynaptic_spike: None,
                            stdp_accumulator: 0.0,
                            modulatory: None,
                        };
                        synapses.insert(SynapseId(id), synapse);
                    }
                }
            }
        }

        // Create basic runtime network
        Ok(RuntimeNetwork {
            neurons,
            synapses,
            assemblies: HashMap::new(),
            patterns: HashMap::new(),
            event_queue: EventQueue::new(10000).unwrap(),
            neuron_pool: MemoryPool::new(1000).unwrap(),
            synapse_pool: MemoryPool::new(2000).unwrap(),
            metadata: NetworkMetadata {
                name: "imported_network".to_string(),
                precision: Precision::Double,
                learning_enabled: true,
                evolution_enabled: false,
                monitoring_enabled: true,
                created_at: chrono::Utc::now().to_rfc3339(),
                version: "1.0.0".to_string(),
            },
            statistics: NetworkStatistics {
                neuron_count: 0,
                synapse_count: 0,
                assembly_count: 0,
                pattern_count: 0,
                total_weight: 0.0,
                average_connectivity: 0.0,
            },
            type_context: TypeInferenceContext::new(),
            runtime_type_validator: RuntimeTypeValidator::new(),
            temporal_constraints: Vec::new(),
            topological_constraints: Vec::new(),
        })
    }
}

/// Data Streaming System
pub struct DataStreamer {
    streams: HashMap<String, DataStream>,
}

impl DataStreamer {
    /// Create a new data streamer
    pub fn new() -> Self {
        Self {
            streams: HashMap::new(),
        }
    }

    /// Create a new data stream
    pub fn create_stream(&mut self, name: String, stream_type: StreamType) -> Result<(), String> {
        let stream = DataStream::new(stream_type);
        self.streams.insert(name, stream);
        Ok(())
    }

    /// Stream data to a stream
    pub fn stream_data(&mut self, stream_name: &str, data: StreamData) -> Result<(), String> {
        if let Some(stream) = self.streams.get_mut(stream_name) {
            stream.add_data(data);
            Ok(())
        } else {
            Err(format!("Stream '{}' not found", stream_name))
        }
    }

    /// Read data from stream
    pub fn read_stream(&self, stream_name: &str, count: usize) -> Result<Vec<StreamData>, String> {
        if let Some(stream) = self.streams.get(stream_name) {
            Ok(stream.read_data(count))
        } else {
            Err(format!("Stream '{}' not found", stream_name))
        }
    }
}

/// Data stream
#[derive(Debug, Clone)]
pub struct DataStream {
    stream_type: StreamType,
    buffer: Vec<StreamData>,
    max_buffer_size: usize,
}

impl DataStream {
    /// Create a new data stream
    pub fn new(stream_type: StreamType) -> Self {
        Self {
            stream_type,
            buffer: Vec::new(),
            max_buffer_size: 1000,
        }
    }

    /// Add data to stream
    pub fn add_data(&mut self, data: StreamData) {
        self.buffer.push(data);

        // Maintain buffer size
        if self.buffer.len() > self.max_buffer_size {
            self.buffer.remove(0);
        }
    }

    /// Read data from stream
    pub fn read_data(&self, count: usize) -> Vec<StreamData> {
        let start = self.buffer.len().saturating_sub(count);
        self.buffer[start..].to_vec()
    }
}

/// Stream types
#[derive(Debug, Clone)]
pub enum StreamType {
    SpikeEvents,
    NeuronActivity,
    NetworkMetrics,
    TrainingData,
}

/// Stream data
#[derive(Debug, Clone)]
pub enum StreamData {
    SpikeEvent(RuntimeSpikeEvent),
    NeuronActivity(NeuronId, f64), // (neuron_id, membrane_potential)
    NetworkMetric(String, f64),    // (metric_name, value)
    TrainingExample(TrainingExample),
}

/// Database Integration
pub mod database {
    use super::*;

    /// Database connection
    pub struct DatabaseConnection {
        connection_string: String,
        connection_type: DatabaseType,
    }

    /// Database types
    #[derive(Debug, Clone)]
    pub enum DatabaseType {
        SQLite,
        PostgreSQL,
        MySQL,
        MongoDB,
    }

    impl DatabaseConnection {
        /// Create a new database connection
        pub fn new(connection_string: String, db_type: DatabaseType) -> Self {
            Self {
                connection_string,
                connection_type: db_type,
            }
        }

        /// Store network in database
        pub fn store_network(&self, network: &RuntimeNetwork, collection: &str) -> Result<(), String> {
            match self.connection_type {
                DatabaseType::SQLite => self.store_sqlite(network, collection),
                DatabaseType::PostgreSQL => self.store_postgresql(network, collection),
                DatabaseType::MongoDB => self.store_mongodb(network, collection),
                _ => Err("Database type not supported".to_string()),
            }
        }

        /// Load network from database
        pub fn load_network(&self, network_id: &str, collection: &str) -> Result<RuntimeNetwork, String> {
            match self.connection_type {
                DatabaseType::SQLite => self.load_sqlite(network_id, collection),
                DatabaseType::PostgreSQL => self.load_postgresql(network_id, collection),
                DatabaseType::MongoDB => self.load_mongodb(network_id, collection),
                _ => Err("Database type not supported".to_string()),
            }
        }

        fn store_sqlite(&self, _network: &RuntimeNetwork, _collection: &str) -> Result<(), String> {
            // SQLite implementation would go here
            Ok(())
        }

        fn load_sqlite(&self, _network_id: &str, _collection: &str) -> Result<RuntimeNetwork, String> {
            // SQLite implementation would go here
            Err("SQLite not implemented".to_string())
        }

        fn store_postgresql(&self, _network: &RuntimeNetwork, _collection: &str) -> Result<(), String> {
            // PostgreSQL implementation would go here
            Ok(())
        }

        fn load_postgresql(&self, _network_id: &str, _collection: &str) -> Result<RuntimeNetwork, String> {
            // PostgreSQL implementation would go here
            Err("PostgreSQL not implemented".to_string())
        }

        fn store_mongodb(&self, _network: &RuntimeNetwork, _collection: &str) -> Result<(), String> {
            // MongoDB implementation would go here
            Ok(())
        }

        fn load_mongodb(&self, _network_id: &str, _collection: &str) -> Result<RuntimeNetwork, String> {
            // MongoDB implementation would go here
            Err("MongoDB not implemented".to_string())
        }
    }
}

/// Utility functions for data management
pub mod utils {
    use super::*;

    /// Create a standard data manager with common importers/exporters
    pub fn create_standard_data_manager() -> DataManager {
        let mut manager = DataManager::new();

        // Register CSV importer/exporter
        manager.register_importer("csv".to_string(), Box::new(CsvImporter));
        manager.register_exporter("csv".to_string(), Box::new(CsvExporter));

        // Register JSON importer/exporter
        manager.register_importer("json".to_string(), Box::new(JsonImporter));
        manager.register_exporter("json".to_string(), Box::new(JsonExporter));

        manager
    }

    /// Convert training dataset to runtime network for testing
    pub fn dataset_to_test_network(dataset: &[TrainingExample]) -> RuntimeNetwork {
        let mut builder = NetworkBuilder::new();

        // Create input neurons
        let mut input_neurons = Vec::new();
        if let Some(example) = dataset.first() {
            for i in 0..example.inputs.len() {
                let neuron = NeuronFactory::create_lif_neuron(
                    NeuronId(i as u32),
                    format!("input_{}", i),
                    -50.0,
                    -70.0,
                    -80.0,
                    2.0,
                );
                input_neurons.push(builder.add_neuron(neuron));
            }

            // Create output neurons
            let mut output_neurons = Vec::new();
            for i in 0..example.targets.len() {
                let neuron = NeuronFactory::create_lif_neuron(
                    NeuronId((i + example.inputs.len()) as u32),
                    format!("output_{}", i),
                    -50.0,
                    -70.0,
                    -80.0,
                    2.0,
                );
                output_neurons.push(builder.add_neuron(neuron));
            }

            // Connect input to output neurons
            for &input_id in &input_neurons {
                for &output_id in &output_neurons {
                    let synapse = SynapseFactory::create_excitatory_synapse(
                        SynapseId(builder.next_synapse_id),
                        input_id,
                        output_id,
                        0.5,
                        1.0,
                    );
                    builder.add_synapse(synapse);
                }
            }
        }

        builder.build()
    }

    /// Export network statistics
    pub fn export_network_statistics(network: &RuntimeNetwork, filepath: &str) -> Result<(), String> {
        let stats = format!(
            "Network Statistics:\n\
             Neurons: {}\n\
             Synapses: {}\n\
             Assemblies: {}\n\
             Patterns: {}\n\
             Total Weight: {:.2}\n\
             Average Connectivity: {:.2}\n\
             Memory Utilization: {:.2}%\n",
            network.neurons.len(),
            network.synapses.len(),
            network.assemblies.len(),
            network.patterns.len(),
            network.synapses.values().map(|s| s.weight.abs()).sum::<f64>(),
            network.neurons.len() as f64 / network.synapses.len().max(1) as f64,
            network.neuron_pool.utilization() * 100.0
        );

        fs::write(filepath, stats)
            .map_err(|e| format!("Failed to write statistics file: {}", e))
    }
}