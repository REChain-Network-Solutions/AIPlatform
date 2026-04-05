<?php
namespace App\Http\Requests\CustomModules;

use Illuminate\Foundation\Http\FormRequest;

class DeleteModuleRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [

        ];
    }
}
