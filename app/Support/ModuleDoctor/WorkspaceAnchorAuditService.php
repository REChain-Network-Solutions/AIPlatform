<?php
namespace App\Support\ModuleDoctor;
class WorkspaceAnchorAuditService {
    public function inspect(array $module): array {
        $files = $module['files'] ?? [];
        $workAnchored = (new \Illuminate\Support\Collection($files))->contains(fn ($file) => str_contains($file, 'sections/work/') || str_contains($file, 'setting-sidebar'));
        return ['work_anchored' => $workAnchored, 'ready' => $workAnchored];
    }
}
