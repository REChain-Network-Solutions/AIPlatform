# Titan Zero BOS Architecture Overview

This document realigns the architecture narrative to the **Titan Zero BOS** model. The platform is device-first, federated, privacy-first, mobile/PWA-first, voice-first, and AI-orchestrated with a strict local-first execution preference.

## Layered view (target model)

```mermaid
graph TD
    A[Client / Device Node (PWA + Voice)] --> B[Local Runtime (Service Worker + IndexedDB)]
    B --> C[Local Signal Queue]
    C --> D[Coordinator / Server Reconciliation]
    D --> E[Shared Services (Auth, Policy, Audit)]
    B --> F[On-Device / Native AI]
    B --> G[Local / Ollama AI]
    D --> H[Cloud AI (fallback)]
    D --> I[Federation Bridges (Web3, Identity, Payments)]
```

## Key concepts
- **Device nodes first:** Each device is a federated node with its own storage and queues. Server sync is reconciliation, not overwrite.
- **PWA runtime:** Service worker + IndexedDB/local caches (planned) backstop offline behavior; mobile-first layouts dominate.
- **Voice-first control:** Voice is a primary interaction channel across mobile and desktop.
- **AI fallback order:** On-device/native → Local/Ollama → Cloud (last resort, auditable).
- **Governance & audit:** Actions are logged for replay and tenant-safe boundaries.
- **Federation bridges:** Web3/identity/payment rails attach as bridges, not core coupling.

## Component breakdown (current vs target)
- **Frontend:** Blade/Livewire + Vite (current). Target: PWA-first build with service worker, local stores, and mobile-friendly voice entry points.
- **Backend:** Laravel APIs, tenancy, and orchestration (current). Target: coordinator role for trust bootstrap, reconciliation, and policy enforcement.
- **Storage:** SQL + object storage (current). Target: IndexedDB/local caches on device; server stores remain authoritative after reconciliation.
- **Queues:** Redis/worker queues (current). Target: device-local signal queue that drains to server coordinators.
- **AI adapters:** Cloud-first integrations exist. Target: enforce on-device/native first, then local/Ollama, then cloud.
- **Bridges:** Web3/4/5 materials remain as bridges. Treat them as optional extensions rather than required runtime.

## Deployment expectations
- **Offline-first:** Devices must continue operating without connectivity; sync later.
- **Reconciliation:** Conflict resolution beats overwriting; design data models with merge semantics.
- **Security:** Trust bootstrap per node, tenant isolation, end-to-end encryption for sync.
- **Edge-friendly:** PWA assets should be cacheable at the edge; coordinators can run centrally or regionally.

## Status labels for docs
- **Implemented:** Present in code today.
- **In transition:** Partially implemented; work underway.
- **Target:** Architectural direction not yet built.

Use these labels in feature docs to avoid overstating current capabilities.
