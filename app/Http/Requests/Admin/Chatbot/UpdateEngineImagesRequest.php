<?php

namespace App\Http\Requests\Admin\Chatbot;

use Illuminate\Foundation\Http\FormRequest;

class UpdateEngineImagesRequest extends FormRequest
{
    public function authorize(): bool
    {
        return $this->user()?->isAdmin() ?? false;
    }

    /**
     * @return array<string, \Illuminate\Contracts\Validation\ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        return [
            'engine_logo'      => 'sometimes|array',
            'engine_logo.*'    => 'nullable|file|mimes:svg,png,jpg,jpeg,gif,webp|max:5120',
        ];
    }

    /**
     * @return array<string, string>
     */
    public function messages(): array
    {
        return [
            'engine_logo.*.mimes'  => __('The engine logo must be an image (SVG, PNG, JPG, GIF or WebP).'),
            'engine_logo.*.max'    => __('The engine logo may not be greater than 5MB.'),
        ];
    }
}
