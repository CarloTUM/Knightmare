# Security Policy

## Supported versions

Only the latest minor release receives security fixes.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | yes       |
| < 0.1   | no        |

## Reporting a vulnerability

If you believe you have found a security vulnerability in Knightmare,
please **do not** open a public issue. Instead, email the maintainer at
`deutschmanncarlo@gmail.com` with:

- A description of the vulnerability and its impact.
- Steps to reproduce.
- Any suggested mitigation if you have one.

You can expect an acknowledgement within seven days. Once the issue is
confirmed and a fix is ready, we coordinate the disclosure timeline with
you.

## Scope

This project ships ML model code and a UCI engine. Threats most likely
to be in scope:

- Code execution through deserialisation of untrusted checkpoints
  (`torch.load`).
- Path traversal in the ingestion / replay-shard helpers.
- Malformed PGN input causing hangs or excessive memory use.

Out of scope:

- Self-induced GPU crashes or out-of-memory while training large
  configurations.
- Issues stemming from third-party dependencies; please report those
  upstream.
