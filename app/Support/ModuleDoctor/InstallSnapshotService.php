<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\File;

class InstallSnapshotService
{
    public function snapshotModule(string $moduleName): ?string
    {
        $source = base_path('Modules/' . $moduleName);

        if (!File::isDirectory($source)) {
            return null;
        }

        $stamp = now()->format('Ymd_His');
        $target = storage_path('app/module-installer-backups/' . $moduleName . '/' . $stamp);
        File::ensureDirectoryExists(dirname($target));
        File::copyDirectory($source, $target);

        return $target;
    }
}
