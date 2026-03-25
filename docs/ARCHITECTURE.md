# Titan Zero BOS — Architecture Overview

## High-level model
Titan Zero BOS treats every device as a federated node. The server coordinates and reconciles, but execution aims to stay local first.

- **PWA runtime (`resources/`, `public/`)**: service worker-ready build, IndexedDB/local caches (planned), voice-first UI surfaces.
- **Backend (`app/`, `routes/`, `config/`)**: Laravel core for orchestration, APIs, tenancy, and policy enforcement.
- **Signal queue (planned)**: local queue per device with reconciliation to server queues; server resolves conflicts rather than overwriting.
- **AI adapters (`packages/`, `bridges/`)**: connectors to on-device/native, local/Ollama, and cloud AI in that preference order.
- **Extensions (`packages/`, `app/Providers/ExtensionServiceProvider`)**: optional modules; core runtime stays lean.

## Data & interaction flows
1. User (voice/touch) -> PWA -> **local queue/storage first**.
2. Device performs **trust handshake** with coordinating services when online.
3. Reconciliation sync merges local signals; conflicts are resolved, not overwritten.
4. AI calls try **on-device**, then **local/Ollama**, then **cloud** as last resort.
5. Audit events are logged for replayability and governance.

## Deployment topology (current vs. target)
- **Current:** Laravel app with Blade/Livewire UI behind a web server; workers handle background jobs; frontend built with Vite/Tailwind.
- **Target:** Service-worker-enabled PWA assets served at the edge; device-local stores keep working offline; server nodes act as coordinators and reconciliation authorities; optional GPU/AI nodes join as local-first inference hosts.

## Security & tenancy
- Tenant isolation is required in any shared deployment; respect per-tenant DB/schema boundaries.
- Trust bootstrap between devices and coordinators must validate identity and permissions before sync.
- Prefer end-to-end encryption for device <-> server channels; local data remains local unless explicitly shared.

## Observability
- Collect metrics/traces/logs, but ensure privacy-first sampling and redaction policies.
- Offline nodes buffer telemetry and flush after reconciliation.

## Extensibility
- Keep platform primitives in core; extensions remain modular under `packages/` or dedicated providers.
- When adding adapters, document whether they execute locally, on nearby nodes, or in the cloud, and how they respect the AI fallback order.
