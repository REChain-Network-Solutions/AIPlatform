<?php
namespace App\Services\CustomModules\DTO;

class SafeFixPlan
{
    public function __construct(
        public array $selectedRepairs,
        public array $plannedActions,
        public array $notes = [],
    ) {}

    public function toArray(): array
    {
        return [
            'selected_repairs' => $this->selectedRepairs,
            'planned_actions' => $this->plannedActions,
            'notes' => $this->notes,
        ];
    }
}
