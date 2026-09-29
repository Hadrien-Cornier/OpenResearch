use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::process::Command;

#[test]
fn full_text_preserves_requested_versions_and_reports_stale_text() {
    let root = std::env::temp_dir().join(format!("orx-paper-cli-{}", uuid::Uuid::new_v4()));
    for (id, text, success, warning) in [
        ("1706.03762v7", "arXiv:1706.03762v7 [cs.CL]", true, ""),
        ("1706.03762v1", "arXiv:1706.03762v1 [cs.CL]", true, ""),
        (
            "1706.03762v7",
            "arXiv:1706.03762v1 [cs.CL]",
            false,
            "version mismatch",
        ),
        (
            "1706.03762v7",
            "No title-page footer",
            true,
            "could not verify",
        ),
        ("1706.03762", "arXiv:1706.03762v1 [cs.CL]", true, ""),
        (
            "hep-th/9711200v3",
            "arXiv:hep-th/9711200v2 [hep-th]",
            false,
            "version mismatch",
        ),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let mut paths = Vec::new();
            for stream in listener.incoming().take(2) {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(&stream);
                let mut request = String::new();
                reader.read_line(&mut request).unwrap();
                let path = request.split_whitespace().nth(1).unwrap().to_string();
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    if header == "\r\n" {
                        break;
                    }
                }
                let body = if path.starts_with("/abs/") {
                    text
                } else {
                    "{\"papers\":[]}"
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .unwrap();
                paths.push(path);
            }
            paths
        });
        let output = Command::new(env!("CARGO_BIN_EXE_orx"))
            .env("ALPHAXIV_WEB_URL", &base)
            .env("ALPHAXIV_API_URL", &base)
            .env("XDG_CONFIG_HOME", root.join("config"))
            .env("ORX_DATA_DIR", root.join("data"))
            .env("ORX_NO_UPDATE_CHECK", "1")
            .args(["--no-telemetry", "paper", id, "--full"])
            .output()
            .unwrap();
        let paths = server.join().unwrap();
        assert!(paths.contains(&format!("/abs/{id}.md")), "{paths:?}");
        assert_eq!(output.status.success(), success);
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        if success {
            assert!(stdout.contains(&format!("alphaXiv: https://www.alphaxiv.org/abs/{id}\n")));
            assert!(stdout.contains(text));
        } else {
            assert!(stdout.is_empty(), "{stdout}");
            assert!(stderr.contains(&format!("https://arxiv.org/pdf/{id}")));
        }
        if warning.is_empty() {
            assert!(!stderr.contains("version"), "{stderr}");
        } else {
            assert!(stderr.contains(warning), "{stderr}");
        }
    }
    let _ = std::fs::remove_dir_all(root);
}
