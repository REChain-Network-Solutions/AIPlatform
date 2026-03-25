<?php

namespace App\Services\TitanSignal;

use Illuminate\Support\Collection;
use Illuminate\Support\Facades\DB;

class LocalQueueService
{
    /**
    * Queue a local signal for later reconciliation.
    */
    public function enqueueSignal(array $signal, ?string $deviceId = null): int
    {
        return DB::table('tz_local_signals')->insertGetId([
            'signal_type' => $signal['type'] ?? null,
            'payload' => $signal,
            'status' => 'pending',
            'device_id' => $deviceId,
            'queued_at' => now(),
            'created_at' => now(),
            'updated_at' => now(),
        ]);
    }

    /**
    * Pull a batch of pending signals for dispatch.
    */
    public function pullPending(int $limit = 50): Collection
    {
        return DB::table('tz_local_signals')
            ->where('status', 'pending')
            ->orderBy('queued_at')
            ->limit($limit)
            ->get();
    }

    /**
    * Mark a signal as dispatched and record optional meta.
    */
    public function markDispatched(int $id, string $status = 'dispatched', ?array $meta = null): void
    {
        DB::table('tz_local_signals')
            ->where('id', $id)
            ->update([
                'status' => $status,
                'dispatched_at' => now(),
                'updated_at' => now(),
            ]);

        if ($meta !== null) {
            $this->logRuntimeMeta('signal_dispatch', [
                'signal_id' => $id,
                'status' => $status,
                'meta' => $meta,
            ]);
        }
    }

    /**
    * Record runtime metadata for auditing/reconciliation.
    */
    public function logRuntimeMeta(string $metaKey, array $value): void
    {
        DB::table('tz_runtime_meta')->insert([
            'category' => 'titan_signal',
            'meta_key' => $metaKey,
            'meta_value' => $value,
            'created_at' => now(),
            'updated_at' => now(),
        ]);
    }
}
