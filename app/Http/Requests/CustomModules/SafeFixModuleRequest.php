<?php
namespace App\Http\Requests\CustomModules;

use Illuminate\Foundation\Http\FormRequest;

class SafeFixModuleRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'filePath' => 'required|string', 'repair_permissions' => 'nullable|boolean', 'repair_package_bridge' => 'nullable|boolean', 'repair_company_module_settings' => 'nullable|boolean', 'clear_caches' => 'nullable|boolean'
        ];
    }
}
