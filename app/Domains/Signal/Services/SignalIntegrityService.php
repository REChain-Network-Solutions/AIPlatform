<?php

declare(strict_types=1);

namespace App\Domains\Signal\Services;

use App\Domains\Node\Contracts\TrustPolicyInterface;
use App\Domains\Node\Services\NodeIdentityService;
use App\Domains\Signal\ValueObjects\SignalEnvelope;

class SignalIntegrityService
{
    public function __construct(
        private readonly NodeIdentityService $nodeIdentity,
        private readonly ?TrustPolicyInterface $trustPolicy = null,
    ) {
    }

    public function sign(array $payload, int $version = 1): SignalEnvelope
    {
        $identity = $this->nodeIdentity->getIdentity();
        $envelopeId = uniqid('signal_', true);
        $digest = hash('sha256', json_encode($payload) . $version . $identity->nodeId);

        return new SignalEnvelope(
            id: $envelopeId,
            version: $version,
            payload: $payload,
            originNodeId: $identity->nodeId,
            signature: hash_hmac('sha256', $digest, config('app.key')),
            digest: $digest
        );
    }

    public function verify(SignalEnvelope $envelope): bool
    {
        if ($this->trustPolicy) {
            return $this->trustPolicy->verify($this->nodeIdentity->getIdentity(), $envelope);
        }

        $expected = hash_hmac('sha256', (string) $envelope->digest, config('app.key'));

        return hash_equals($expected, (string) $envelope->signature);
    }
}
