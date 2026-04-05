<?php
namespace App\Services\CustomModules\DTO;

class ModuleAnalysisResult
{
    public function __construct(
        public string $status,
        public string $file,
        public int $size,
        public string $modifiedAt,
        public array $checks,
        public array $warnings,
        public array $blockingIssues,
        public array $summaryCards,
        public array $fixPlan,
        public array $meta = [],
    ) {}

    public function toArray(): array
    {
        return [
            'status' => $this->status,
            'file' => $this->file,
            'size' => $this->size,
            'modified_at' => $this->modifiedAt,
            'checks' => $this->checks,
            'warnings' => $this->warnings,
            'blocking_issues' => $this->blockingIssues,
            'summary_cards' => $this->summaryCards,
            'fix_plan' => $this->fixPlan,
            'meta' => $this->meta,
        ];
    }
}
