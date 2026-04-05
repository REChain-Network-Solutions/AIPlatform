<?php

namespace App\Support\ModuleDoctor;

class CompanyModuleSeederService
{
    public function preview(string $moduleName): array
    {
        return [
            'status' => 'pass',
            'detail' => 'Prepared company module_settings seed preview for ' . $moduleName . '.',
            'sql_preview' => "INSERT IGNORE INTO company_module_settings (company_id, module_name, status) SELECT id, '" . addslashes($moduleName) . "', 'active' FROM companies;",
        ];
    }
}
