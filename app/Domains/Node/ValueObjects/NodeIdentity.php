<?php

declare(strict_types=1);

namespace App\Domains\Node\ValueObjects;

class NodeIdentity
{
    public function __construct(
        public readonly string $nodeId,
        public readonly string $publicKey,
    ) {
    }
}
