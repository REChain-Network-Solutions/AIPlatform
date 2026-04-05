<?php

namespace App\Support\ModuleDoctor;

use Nwidart\Modules\Facades\Module;

class ModuleEnablementTruthService
{
    public function inspect(string $moduleName): array
    {
        $module = Module::find($moduleName);

        return [
            'status' => $module ? 'pass' : 'warn',
            'detail' => $module ? 'Module exists in the runtime registry.' : 'Module was not found in the runtime module registry.',
            'registered' => (bool) $module,
            'enabled' => $module ? (bool) $module->isEnabled() : false,
        ];
    }
}
