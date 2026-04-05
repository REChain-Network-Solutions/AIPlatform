<?php

namespace App\Support\ModuleDoctor;

class InstallSurfaceAuditService
{
    public function inspect(array $analysis): array
    {
        return [
            'status' => empty($analysis['blocking_issues'] ?? []) ? 'pass' : 'warn',
            'detail' => empty($analysis['blocking_issues'] ?? [])
                ? 'Installer surface is clear to continue.'
                : 'Installer surface contains blocking issues that should be repaired before install.',
        ];
    }
}
