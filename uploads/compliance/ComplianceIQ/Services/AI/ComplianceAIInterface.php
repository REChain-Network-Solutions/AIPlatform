<?php
namespace Modules\ComplianceIQ\Services\AI;
use Modules\ComplianceIQ\Entities\ComplianceReport;
interface ComplianceAIInterface {
    public function summarize(ComplianceReport $report): array;
    public function explainIssue(string $issue, array $context = []): array;
}
