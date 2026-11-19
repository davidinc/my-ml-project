# Modern Phishing Tooling & Defensive Architecture

A reference guide for senior engineers covering Adversary-in-the-Middle (AiTM) phishing frameworks, emerging identity attack vectors, and cryptographic defense-in-depth strategies.

> **Disclaimer:** Offensive tools listed in this repository are documented strictly for authorized red teaming, adversary simulation, and defensive engineering.

## 1. Modern Phishing & AiTM Tooling

| Tool / Framework | Primary Capability | Mechanism |
| :--- | :--- | :--- |
| **Evilginx3** | AiTM Reverse Proxy | Intercepts traffic between the victim and IdP (Entra ID, Okta) to capture live session cookies and bypass standard MFA. |
| **Modlishka** | Reverse Proxy with Automated TLS | Dynamically rewrites target domain responses on the fly without static HTML templates. |
| **GoPhish** | Phishing Simulation | Open-source framework for authorized internal testing and API-driven campaign tracking. |
| **EvilnoVNC** | Browser-in-the-Browser (BitB) / VNC | Streams a containerized headless browser session to the victim to capture full browser state and tokens. |

## 2. Emerging Attack Vectors

* **Device Code Phishing:** Abuses the OAuth 2.0 Device Authorization Grant flow (`microsoft.com/devicelogin`) to obtain valid refresh tokens without stealing passwords.
* **Consent Phishing:** Tricks users into granting permissions (`Mail.Read`, `Files.Read.All`) to rogue third-party OAuth apps; access persists even after a password reset.
* **Quishing (QR Code Phishing):** Embeds malicious URLs in QR codes inside emails/PDFs to evade Secure Email Gateways (SEGs) and move victims to unmanaged mobile devices.
* **LLM-Driven Spear Phishing:** Automates OSINT scraping and generates polymorphic, hyper-contextualized lures.
* **HTML Smuggling & SVG Payloads:** Assembles malware locally inside the victim's browser via JavaScript blobs to bypass perimeter inspection.

## 3. Defensive Architecture

* **Phishing-Resistant MFA (FIDO2 / WebAuthn):** Enforce passkeys or hardware keys to cryptographically bind authentication to the origin domain, breaking AiTM proxies.
* **Token Protection:** Bind refresh tokens to the device's TPM via Conditional Access to neutralize stolen session cookies.
* **OAuth App Governance:** Disable user consent for unverified third-party apps and enforce admin approval workflows.
* **Email Authentication:** Deploy SPF, DKIM, and DMARC with `p=reject` and monitor aggregate reports.
* **Block Device Code Flow:** Restrict OAuth Device Code authentication via Conditional Access unless explicitly required.
