<?php

declare(strict_types=1);

namespace App\Domains\Node\Services;

use App\Domains\Node\ValueObjects\NodeIdentity;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Str;

class NodeIdentityService
{
    public function getIdentity(): NodeIdentity
    {
        $nodeId = Cache::rememberForever('node_identity:id', fn () => $this->generateNodeId());
        $publicKey = Cache::rememberForever('node_identity:pub', fn () => $this->derivePublicKey($nodeId));

        return new NodeIdentity($nodeId, $publicKey);
    }

    private function generateNodeId(): string
    {
        return Str::uuid()->toString();
    }

    private function derivePublicKey(string $nodeId): string
    {
        // Placeholder derivation until a real key management layer is added.
        return hash('sha256', $nodeId . config('app.key'));
    }
}
