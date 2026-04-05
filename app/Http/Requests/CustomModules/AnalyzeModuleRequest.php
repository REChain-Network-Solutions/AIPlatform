<?php
namespace App\Http\Requests\CustomModules;

use Illuminate\Foundation\Http\FormRequest;

class AnalyzeModuleRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'filePath' => 'required|string'
        ];
    }
}
