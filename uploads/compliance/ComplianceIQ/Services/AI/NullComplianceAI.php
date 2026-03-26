<?php
namespace Modules\ComplianceIQ\Services\AI;
use Modules\ComplianceIQ\Entities\ComplianceReport;
class NullComplianceAI implements ComplianceAIInterface {
    public function summarize(ComplianceReport $report): array {
        return [
            'overview' => 'AI (null) summary placeholder.',
            'risks' => [],
            'recommendations' => ['Install real AI provider']
        ];
    }
    public function explainIssue(string $issue, array $context = []): array {
        return [
            'explanation' => 'AI (null) explanation placeholder for: '.$issue,
            'next_steps' => ['Attach provider to get real analysis'],
            'confidence' => 0.2
        ];
    }
}
