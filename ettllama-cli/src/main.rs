use eyre::Result;
use futures::{SinkExt, StreamExt};
use http::Uri;
use inquire::Text;
use owo_colors::OwoColorize;
use rustls::{ClientConfig, OwnedTrustAnchor, RootCertStore, ServerName};
use std::{
    env,
    io::{self, Write},
    sync::Arc,
};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;
use tracing_subscriber::EnvFilter;
use trust_dns_resolver::{
    config::{ResolverConfig, ResolverOpts},
    TokioAsyncResolver,
};
use tungstenite::Message;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .compact()
        .init();

    let mut cert_store = RootCertStore::empty();
    cert_store.add_trust_anchors(webpki_roots::TLS_SERVER_ROOTS.iter().map(|anchor| {
        OwnedTrustAnchor::from_subject_spki_name_constraints(
            anchor.subject,
            anchor.spki,
            anchor.name_constraints,
        )
    }));

    let connector = TlsConnector::from(Arc::new(
        ClientConfig::builder()
            .with_safe_defaults()
            .with_root_certificates(cert_store)
            .with_no_client_auth(),
    ));

    let uri = env::args().nth(1).expect("server URI").parse::<Uri>()?;
    let port = uri.port_u16().unwrap_or(3000);
    let resolver = TokioAsyncResolver::tokio(ResolverConfig::quad9_tls(), ResolverOpts::default());
    let server_ip = resolver
        .lookup_ip(uri.host().unwrap())
        .await?
        .into_iter()
        .next()
        .expect("server IP address");

    let stream = connector
        .connect(
            ServerName::try_from(uri.host().unwrap())?,
            TcpStream::connect((server_ip, port)).await?,
        )
        .await?;
    let (ws_stream, _) = tokio_tungstenite::client_async(uri, stream).await?;
    let (mut writer, mut reader) = ws_stream.split();

    loop {
        let Ok(prompt) = Text::new("user:")
            .with_placeholder("your prompt here...")
            .prompt()
        else {
            break;
        };

        writer.send(Message::Text(prompt)).await?;
        writer.flush().await?;

        print!("{} bot: ", "!".green());

        while let Some(Ok(Message::Text(tok))) = reader.next().await {
            if tok.is_empty() {
                break;
            }

            print!("{}", tok.green());
            io::stdout().lock().flush()?;
        }

        println!();
        io::stdout().lock().flush()?;
    }
}
