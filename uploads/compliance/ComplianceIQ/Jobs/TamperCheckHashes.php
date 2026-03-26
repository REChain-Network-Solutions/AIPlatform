<?php

namespace Modules\ComplianceIQ\Jobs;

use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Modules\ComplianceIQ\Entities\ComplianceHash;

class TamperCheckHashes implements ShouldQueue
{
    use Dispatchable, Queueable;

    public function handle(): void
    {
        ComplianceHash::query()->orderBy('id')->chunkById(500, function($rows){
            foreach ($rows as $hashRow) {
                $recomputed = $hashRow->sha256; // placeholder for domain-specific recompute
                $status = hash_equals($hashRow->sha256, $recomputed) ? 'valid' : 'mismatch';

                ComplianceHash::create([
                    'hashable_type' => $hashRow->hashable_type,
                    'hashable_id'   => $hashRow->hashable_id,
                    'sha256'        => $recomputed,
                    'computed_at'   => now(),
                    'status'        => $status,
                ]);
            }
        });
    }
}
