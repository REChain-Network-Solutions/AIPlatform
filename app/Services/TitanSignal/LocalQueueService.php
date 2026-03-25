<?php

namespace App\Services\TitanSignal;

use App\Services\TitanRuntime\TelemetryService;
use Illuminate\Support\Arr;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\DB;

class LocalQueueService
{
    public function __construct(private ?TelemetryService $telemetry = null)
    {
        $this->telemetry = $telemetry ?? new TelemetryService();
    }

    /**
     * Queue a local signal for later reconciliation.
     * Capture -> store locally -> queue for sync. No cloud calls here.
     * Expected signal keys: signal_type/type, payload/payload_json/data, action, origin_node, team_id.
     */
    public function enqueue(array $signal): int
    {
        $payload = $signal['payload'] ?? $signal['payload_json'] ?? $signal['data'] ?? null;
        if ($payload === null) {
            // Keep payload limited to business fields if provided structure is missing.
            $payload = Arr::except($signal, ['type', 'signal_type', 'action', 'origin_node', 'team_id']);
        }

        $id = DB::table('tz_local_signals')->insertGetId([
            'signal_type' => $signal['type'] ?? $signal['signal_type'] ?? null,
            'payload_json' => json_encode($payload),
            'origin_node' => $signal['origin_node'] ?? null,
            'team_id' => $signal['team_id'] ?? null,
            'status' => 'pending',
            'queued_at' => now(),
            'created_at' => now(),
        ]);

        DB::table('tz_sync_queue')->insert([
            'object_type' => $signal['type'] ?? $signal['signal_type'] ?? 'unknown_signal',
            // Maintain string object_id for sync queue compatibility while referencing the local signal id.
            'object_id' => (string) $id,
            'action' => $signal['action'] ?? 'capture',
            'status' => 'pending',
            'retry_count' => 0,
            'payload_json' => json_encode($payload),
            'created_at' => now(),
        ]);

        $this->telemetry->logExecution(
            subsystem: 'titan_signal',
            event: 'enqueue',
            executionLayer: 'device',
            success: true,
            metadata: [
                'signal_id' => $id,
                'signal_type' => $signal['type'] ?? null,
            ]
        );

        return $id;
    }

    /**
     * Pull a batch of pending signals for dispatch.
     */
    public function dequeue(int $limit = 50): Collection
    {
        return DB::table('tz_local_signals')
            ->where('status', 'pending')
            ->orderBy('queued_at')
            ->limit($limit)
            ->get();
    }

    /**
     * Retry failed sync queue entries by resetting them to pending.
     */
    public function retryFailed(int $limit = 50): int
    {
        $ids = DB::table('tz_sync_queue')
            ->where('status', 'failed')
            ->limit($limit)
            ->pluck('id');

        if ($ids->isEmpty()) {
            return 0;
        }

        DB::table('tz_sync_queue')
            ->whereIn('id', $ids)
            ->update([
                'status' => 'pending',
                'retry_count' => DB::raw('retry_count + 1'),
                'last_attempt_at' => null,
                'updated_at' => now(),
            ]);

        $this->telemetry->logFailure(
            subsystem: 'titan_signal',
            event: 'retry_scheduled',
            metadata: ['queue_ids' => $ids->all()]
        );

        return $ids->count();
    }

    /**
     * Mark a batch of pending sync items as in-flight and return them.
     */
    public function flushBatch(int $limit = 25): Collection
    {
        $items = DB::table('tz_sync_queue')
            ->where('status', 'pending')
            ->orderBy('created_at')
            ->limit($limit)
            ->get();

        if ($items->isNotEmpty()) {
            DB::table('tz_sync_queue')
                ->whereIn('id', $items->pluck('id'))
                ->update([
                    'status' => 'processing',
                    'last_attempt_at' => now(),
                    'updated_at' => now(),
                ]);

            $this->telemetry->logExecution(
                subsystem: 'titan_signal',
                event: 'flush_batch',
                executionLayer: 'device',
                success: true,
                metadata: ['count' => $items->count()]
            );
        }

        return $items;
    }

    /**
     * Mark a signal as processed and record optional meta.
     */
    public function markProcessed(int $id, string $status = 'processed', ?array $meta = null): void
    {
        DB::table('tz_local_signals')
            ->where('id', $id)
            ->update([
                'status' => $status,
                'processed_at' => now(),
                'updated_at' => now(),
            ]);

        DB::table('tz_sync_queue')
            ->where('object_id', (string) $id)
            ->update([
                'status' => $status,
                'processed_at' => now(),
                'updated_at' => now(),
            ]);

        $this->telemetry->logExecution(
            subsystem: 'titan_signal',
            event: 'mark_processed',
            executionLayer: 'device',
            success: true,
            metadata: [
                'signal_id' => $id,
                'status' => $status,
                'meta' => $meta,
            ]
        );
    }
}
