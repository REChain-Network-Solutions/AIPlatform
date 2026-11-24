<?php

namespace App\Http\Requests\Admin\Settings;

use Illuminate\Foundation\Http\FormRequest;

class UpdateHeygenSettingsRequest extends FormRequest
{
    /**
     * Determine if the user is authorized to make this request.
     */
    public function authorize(): bool
    {
        return $this->user()?->isAdmin() ?? false;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array<string, \Illuminate\Contracts\Validation\ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        return ['heygen_secret_key' => ['required', 'string']];
    }

    public function messages(): array
    {
        return [
            'heygen_secret_key.required' => __('The Heygen API key is required.'),
            'heygen_secret_key.string'   => __('The Heygen API key must be a string.'),
        ];
    }
}
