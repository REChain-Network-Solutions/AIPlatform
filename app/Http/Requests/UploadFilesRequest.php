<?php

namespace App\Http\Requests;

use Illuminate\Contracts\Validation\ValidationRule;
use Illuminate\Foundation\Http\FormRequest;

class UploadFilesRequest extends FormRequest
{
    /**
     * Determine if the user is authorized to make this request.
     */
    public function authorize(): bool
    {
        return $this->user() !== null;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array<string, ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        return [
            'files'        => ['required', 'array', 'min:1', 'max:5'],
            'files.*'      => ['required', 'array:name,data'],
            'files.*.name' => ['required', 'string', 'max:255'],
            'files.*.data' => ['required', 'string', 'max:30000000'],
        ];
    }

    public function messages(): array
    {
        return [
            'files.max'        => __('A maximum of five files may be uploaded at once.'),
            'files.*.data.max' => __('Each encoded file must be smaller than 30 MB.'),
        ];
    }
}
