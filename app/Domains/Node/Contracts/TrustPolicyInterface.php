<?php

declare(strict_types=1);

namespace App\Domains\Node\Contracts;

use App\Domains\Node\ValueObjects\NodeIdentity;
use App\Domains\Signal\ValueObjects\SignalEnvelope;

interface TrustPolicyInterface
{
    public function verify(NodeIdentity $identity, SignalEnvelope $envelope): bool;
}
