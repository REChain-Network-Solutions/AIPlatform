# Titan Zero BOS

Titan Zero BOS is a business operating system built for device-first, privacy-first operations. It treats every device as a federated node that can work offline, reconcile when connected, and prefer on-device intelligence before using any remote AI. The platform ships with a Laravel core, Blade/Livewire UI, and a PWA-oriented frontend toolchain (Vite, Tailwind) that we are aligning to this model.

## What this repository is
- **Platform identity:** Titan Zero BOS (formerly documented as MagicAI / AIPlatform in legacy materials).
- **Repository slug:** `Titan-BOS` (kept for continuity); platform identity is Titan Zero BOS.
- **Purpose:** Provide a mobile-first, PWA-first runtime with federated device nodes, local storage/queues, and voice-first control surfaces.
- **Current implementation:** Laravel 10 backend, Blade/Livewire views, Vite/Tailwind build, extensions under `packages/` and `app/Providers/ExtensionServiceProvider`.
- **Target model:** Offline-capable PWA with service worker, IndexedDB/local stores, local signal queue, node bootstrap + trust handshake, and reconciliation-based sync.

## Core principles
- **Device-first & federated:** Devices are primary execution nodes that sync via reconciliation, not overwrite. Server coordination is supportive, not default.
- **Privacy-first:** Local data and local inference are preferred; remote calls are gated and auditable.
- **Mobile-first & PWA-first:** Service worker + IndexedDB/local caches keep the experience responsive and offline-capable.
- **Voice-first control:** Voice is a primary command surface across mobile and desktop surfaces.
- **AI fallback order:**  
  1) On-device / native models  
  2) Local/Ollama hosts  
  3) Server/cloud AI as last resort

## Architecture at a glance
- **Runtime:** Laravel backend, Blade/Livewire UI, Vite-built assets, Tailwind styling.
- **Local-first data:** Target use of IndexedDB/local stores for PWA data, with local signal queues before server submission.
- **Federated nodes:** Each device performs a bootstrap + trust validation handshake before joining the mesh; sync is reconciliation-focused.
- **Governance & audit:** Actions are logged for replayability; tenant boundaries must remain explicit in shared environments.
- **Extensions:** Optional packages live under `packages/` and discovery code under `app/Providers/ExtensionServiceProvider`; foundational runtime stays in core.
- **Voice & accessibility:** Voice commands and low-friction mobile controls are first-class interaction goals.

### Current state vs. target (honest view)
| Area | Current | Target / Plan |
| --- | --- | --- |
| Identity | Legacy MagicAI/AIPlatform labels remain in code & packages | Canonical name Titan Zero BOS; keep legacy labels noted until safe migrations |
| Frontend | Laravel Blade/Livewire + Vite | PWA-first with service worker + IndexedDB caches and local signal queues |
| AI execution | Remote AI providers configured; local preference not enforced | Enforce local/on-device then Ollama/local hosts, cloud last |
| Sync | Server-first flows | Reconciliation-based sync with device-led queues |
| Voice | Voice integrations present in bridges | Voice treated as default command surface in UX copy and flows |

## Getting started (developer)
```bash
git clone https://github.com/Masterleeaus/Titan-BOS.git Titan-BOS
cd Titan-BOS
cp .env.example .env    # set APP_NAME, DB credentials, API keys
composer install
npm install
php artisan migrate --seed   # import seed data as needed
npm run dev                  # or npm run build for production assets
php artisan serve            # start Laravel server
```

> Repository slug is `Titan-BOS` for compatibility; platform identity is **Titan Zero BOS**.
> Legacy docs referenced `REChain-Network-Solutions/AIPlatform`; the active repository lives at `Masterleeaus/Titan-BOS`.

### PWA/mobile-first notes
- Ensure HTTPS + valid host when testing service-worker capable builds.
- Prefer local storage/queues for commands; treat server sync as reconciliation.
- When implementing new screens, keep mobile-first layouts and voice entry points visible.

### AI execution policy
- Attempt **on-device** or **native** inference first.
- If unavailable, route to **local/Ollama** hosts.
- Use **server/cloud** AI only when the above cannot satisfy the request, and surface that fallthrough to the user.

## Migration notes (legacy names)
- Composer packages and DB names still reference `magicai`/`AIPlatform`; these remain for stability and require a separate migration plan.
- UI text, docs, and display names should use **Titan Zero BOS** going forward. If legacy terms appear for technical reasons, annotate them as legacy.

## Contributing
- Follow privacy-first and device-first patterns for new features.
- Prefer local/offline-friendly storage and reconciliation patterns over server-first writes.
- Document whether a feature is **implemented**, **in transition**, or **target** to avoid overstating capabilities.

## Security & privacy
- Keep secrets in `.env`; do not commit keys.
- Enforce HTTPS, rate limits, and tenant isolation.
- Audit trails and replayability are required for governance surfaces.

## License
BSD 3-Clause (see `LICENSE` where applicable).
