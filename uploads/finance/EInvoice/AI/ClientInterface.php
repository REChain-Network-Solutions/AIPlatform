<?php

namespace Modules\EInvoice\AI;

interface ClientInterface
{
    /**
     * Minimal text completion helper to verify AI wiring.
     * Should throw a descriptive exception if misconfigured.
     */
    public function complete(string $prompt, array $opts = []): string;

    /**
     * Health check returns structured status.
     */
    public function health(): array;
}
