use eyre::Result;
use futures::{SinkExt, StreamExt};
use maplit::hashmap;
use rustls::{Certificate, PrivateKey, ServerConfig};
use sha2::{Digest, Sha256};
use std::{
    convert::Infallible, env, fs::File, io::BufReader, net::SocketAddr, path::PathBuf, sync::Arc,
};
use string_template::Template;
use tokio::{
    fs,
    net::{TcpListener, TcpStream},
    task::yield_now,
};
use tokio_rustls::{server::TlsStream, TlsAcceptor};
use tracing::info;
use tracing_subscriber::EnvFilter;
use tungstenite::Message;

const TLS_CERT_VAR: &str = "TLS_CERT";
const TLS_KEY_VAR: &str = "TLS_KEY";
const ADDR_VAR: &str = "ADDR";
const MODEL_PATH_VAR: &str = "MODEL_PATH";
const MODEL_ARCH_VAR: &str = "MODEL_ARCH";
const TEMPLATE_FILE_VAR: &str = "TEMPLATE_FILE";
const ACCEL_OFFLOAD_LAYERS_VAR: &str = "ACCEL_OFFLOAD_LAYERS";
const INFERENCE_BATCH_SIZE_VAR: &str = "INFERENCE_BATCH_SIZE";
const INFERENCE_THREADS_VAR: &str = "INFERENCE_THREADS";

// running multi-threaded breaks ggml-sys with multiple client?
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    color_eyre::install()?;
    dotenv::dotenv()?;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .compact()
        .init();

    let cert_file = File::open(env::var(TLS_CERT_VAR)?)?;
    let mut cert_reader = BufReader::new(cert_file);
    let certs = rustls_pemfile::certs(&mut cert_reader)?
        .into_iter()
        .map(Certificate)
        .collect::<Vec<_>>();
    let key_file = File::open(env::var(TLS_KEY_VAR)?)?;
    let mut key_reader = BufReader::new(key_file);
    let key = rustls_pemfile::pkcs8_private_keys(&mut key_reader)?
        .into_iter()
        .map(PrivateKey)
        .next()
        .expect("PEM-encoded private key");

    let model: Arc<dyn llm::Model> = Arc::from(llm::load_dynamic(
        env::var(MODEL_ARCH_VAR)?.parse().ok(),
        &env::var(MODEL_PATH_VAR)?.parse::<PathBuf>()?,
        llm::TokenizerSource::Embedded,
        llm::ModelParameters {
            use_gpu: accelerated!(),
            gpu_layers: if accelerated!() {
                env::var(ACCEL_OFFLOAD_LAYERS_VAR)?.parse().ok()
            } else {
                None
            },
            ..Default::default()
        },
        |progress| match progress {
            llm::LoadProgress::HyperparametersLoaded => info!("loaded hyperparams!"),
            llm::LoadProgress::ContextSize { bytes } => info!("context size: {bytes}B"),
            llm::LoadProgress::TensorLoaded {
                current_tensor,
                tensor_count,
            } => info!("loaded {current_tensor}/{tensor_count} tensors"),
            llm::LoadProgress::LoraApplied { name, source } => {
                info!("applied LoRA {name} from {source:?}")
            }
            llm::LoadProgress::Loaded {
                file_size,
                tensor_count,
            } => info!("loaded model ({file_size}B, {tensor_count} tensors)"),
        },
    )?);

    let acceptor = TlsAcceptor::from(Arc::new(
        ServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(certs, key)?,
    ));

    let socket = TcpListener::bind(env::var(ADDR_VAR)?.parse::<SocketAddr>()?).await?;

    while let Ok((stream, addr)) = socket.accept().await {
        let Ok(stream) = acceptor.accept(stream).await else {
            continue;
        };
        info!("{addr} connected with TLS!");
        tokio::spawn(handler(stream, addr, Arc::clone(&model)));
    }

    Ok(())
}

#[macro_export]
macro_rules! accelerated {
    () => {
        cfg!(any(
            feature = "clblast",
            feature = "cublas",
            feature = "metal"
        ))
    };
}

async fn handler(
    stream: TlsStream<TcpStream>,
    addr: SocketAddr,
    model: Arc<dyn llm::Model>,
) -> Result<()> {
    let stream = tokio_tungstenite::accept_async(stream).await?;
    info!("{addr} completed WebSocket handshake!");
    let (mut writer, mut reader) = stream.split();

    let mut session = model.start_session(llm::InferenceSessionConfig {
        n_batch: env::var(INFERENCE_BATCH_SIZE_VAR)?.parse()?,
        n_threads: env::var(INFERENCE_THREADS_VAR)?.parse()?,
        ..Default::default()
    });

    let template = Template::new(&fs::read_to_string(env::var(TEMPLATE_FILE_VAR)?).await?);

    while let Some(Ok(Message::Text(prompt))) = reader.next().await {
        let mut hasher = Sha256::new();
        hasher.update(&prompt);
        info!("{addr} submitted prompt with hash {:x}", hasher.finalize());

        let template_params = hashmap! {
            "prompt" => prompt.as_str(),
        };
        let prompt = template.render(&template_params);

        for word in prompt.split_whitespace() {
            session.feed_prompt(&*model, word, &mut llm::OutputRequest::default(), |_| {
                Ok::<_, Infallible>(llm::InferenceFeedback::Continue)
            })?;
            yield_now().await;
        }

        loop {
            let Ok(tok) = session.infer_next_token(
                &*model,
                &llm::InferenceParameters::default(),
                &mut llm::OutputRequest::default(),
                &mut rand::thread_rng(),
            ) else {
                break;
            };

            assert!(!tok.is_empty()); // otherwise it breaks everything
            writer.send(Message::Text(String::from_utf8(tok)?)).await?;
            writer.flush().await?;
            yield_now().await;
        }

        writer.send(Message::Text(String::new())).await?;
        writer.flush().await?;
        yield_now().await;
    }

    info!("{addr} disconnected!");
    Ok(())
}
