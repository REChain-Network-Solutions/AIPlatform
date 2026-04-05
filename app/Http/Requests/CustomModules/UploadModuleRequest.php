<?php
namespace App\Http\Requests\CustomModules;

use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Http\UploadedFile;

class UploadModuleRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'file' => 'nullable|file|mimes:zip|max:51200',
            'download_file' => 'nullable|file|mimes:zip|max:51200',
            'module' => 'nullable|file|mimes:zip|max:51200',
            'zip' => 'nullable|file|mimes:zip|max:51200',
        ];
    }

    public function validatedUpload(): UploadedFile
    {
        return $this->file('file')
            ?? $this->file('download_file')
            ?? $this->file('module')
            ?? $this->file('zip');
    }
}
