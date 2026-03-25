<?php

namespace App\Services\TitanRuntime;

use Illuminate\Support\Facades\DB;

class TelemetryService
{
    public function logExecution(
        string $subsystem,
        string $event,
        string $executionLayer,
        bool $success,
        array $metadata = [],
        ?int $latencyMs = null
    ): void {
        $this->writeMeta($subsystem, $event, $executionLayer, $success, $latencyMs, $metadata);
    }

    public function logFallback(string $subsystem, string $provider, bool $success, array $metadata = [], ?int $latencyMs = null): void
    {
        $this->writeMeta($subsystem, 'fallback', $provider, $success, $latencyMs, $metadata);
    }

    public function logFailure(string $subsystem, string $event, array $metadata = [], string $executionLayer = 'device'): void
    {
        $this->writeMeta($subsystem, $event, $executionLayer, false, metadata: $metadata);
    }

    public function logLatency(string $subsystem, string $executionLayer, int $latencyMs, array $metadata = []): void
    {
        $this->writeMeta($subsystem, 'latency', $executionLayer, true, $latencyMs, $metadata);
    }

    public function logProviderSelection(string $subsystem, string $executionLayer, bool $success, array $metadata = []): void
    {
        $this->writeMeta($subsystem, 'provider_selection', $executionLayer, $success, metadata: $metadata);
    }

    protected function writeMeta(
        string $subsystem,
        string $event,
        ?string $executionLayer,
        bool $success,
        ?int $latencyMs = null,
        array $metadata = []
    ): void {
        DB::table('tz_runtime_meta')->insert([
            'subsystem' => $subsystem,
            'event' => $event,
            'execution_layer' => $executionLayer,
            'latency_ms' => $latencyMs,
            'success_flag' => $success,
            'metadata_json' => json_encode($metadata),
            'created_at' => now(),
            'updated_at' => now(),
        ]);
    }
}
