<?php

declare(strict_types=1);

namespace App\Domains\Signal\ValueObjects;

class SignalEnvelope
{
    public function __construct(
        public readonly string $id,
        public readonly int $version,
        public readonly array $payload,
        public readonly string $originNodeId,
        public readonly ?string $signature = null,
        public readonly ?string $digest = null,
    ) {
    }
}
