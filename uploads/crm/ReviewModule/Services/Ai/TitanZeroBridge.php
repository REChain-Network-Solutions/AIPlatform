<?php

namespace Modules\ReviewModule\Services\Ai;

use Illuminate\Support\Facades\Log;

/**
 * TitanZeroBridge (Pass 16)
 *
 * Bridges module signals and agent calls into TitanZero's ZeroGateway service.
 * Safe-by-default: never throws, never blocks core flows if AI is unavailable.
 */
class TitanZeroBridge
{
    protected function gateway()
    {
        if (class_exists(\Modules\TitanZero\Services\ZeroGateway::class)) {
            return app(\Modules\TitanZero\Services\ZeroGateway::class);
        }
        return null;
    }

    public function ingestSignal(array $signal, $tenantId = null): array
    {
        try {
            $gw = $this->gateway();
            if (!$gw) return ['ok' => false, 'reason' => 'TitanZero not installed/enabled'];

            $resp = $gw->ingestSignal($signal, is_numeric($tenantId) ? (int)$tenantId : null);
            return ['ok' => true, 'response' => $resp];
        } catch (\Throwable $e) {
            Log::warning('[TitanAI] ingestSignal failed', ['module' => 'ReviewModule', 'err' => $e->getMessage()]);
            return ['ok' => false, 'reason' => $e->getMessage()];
        }
    }

    /**
     * Run a specialist agent by slug (e.g. quote_agent, dispatch_agent, training_agent).
     */
    public function runAgent(string $agentSlug, array $context, $tenantId = null): ?array
    {
        try {
            $gw = $this->gateway();
            if (!$gw) return null;

            return $gw->runAgent([
                'agent_slug' => $agentSlug,
                'input' => $context,
            ], is_numeric($tenantId) ? (int)$tenantId : null);
        } catch (\Throwable $e) {
            Log::warning('[TitanAI] runAgent failed', ['module' => 'ReviewModule', 'agent' => $agentSlug, 'err' => $e->getMessage()]);
            return null;
        }
    }

    /**
     * Ask TitanZero for suggested actions (proposals).
     *
     * Strategy:
     * - Use dispatch_agent for operational proposals.
     * - Try to parse JSON from the agent's message; if not JSON, wrap as a single draft proposal.
     */
    public function proposeActions(array $context, $tenantId = null): array
    {
        $resp = $this->runAgent('dispatch_agent', $context, $tenantId);
        if (!is_array($resp)) return [];

        $msg = data_get($resp, 'result.message');
        if (!is_string($msg) || trim($msg) === '') return [];

        $trim = trim($msg);
        // attempt JSON parse
        if ((str_starts_with($trim, '{') && str_ends_with($trim, '}')) || (str_starts_with($trim, '[') && str_ends_with($trim, ']'))) {
            $decoded = json_decode($trim, true);
            if (is_array($decoded)) {
                // If agent already returns proposals-like structure, return it.
                return $decoded['proposals'] ?? $decoded;
            }
        }

        // Fallback: wrap message into a draft proposal for Communication
        return [[
            'target_module' => 'Communication',
            'action_type' => 'draft_message',
            'payload' => [
                'channel' => 'internal',
                'text' => $msg,
                'context' => $context,
            ],
            'confidence' => 0.6,
            'risk_level' => 'green',
            'explanation' => 'Draft generated from dispatch_agent output (fallback wrapper).',
            'evidence_refs' => ['titan_zero_audit_id' => data_get($resp, 'audit_id')],
        ]];
    }

    public function proposeDraftMessage(array $context, $tenantId = null): ?array
    {
        $context['intent'] = $context['intent'] ?? 'draft_message';
        $proposals = $this->proposeActions($context, $tenantId);
        return $proposals[0] ?? null;
    }
}
