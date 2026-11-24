<?php

namespace App\Http\Requests;

use Illuminate\Contracts\Validation\ValidationRule;
use Illuminate\Foundation\Http\FormRequest;

class UpdateChannelRequest extends FormRequest
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
     * @return array<string, ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        return [
            'enable_beta' => ['required', 'boolean'],
        ];
    }

    /**
     * Normalize browser form submissions (jQuery sends "true"/"false" strings
     * url-encoded, which Laravel's strict boolean rule would reject). Anything
     * else is left untouched so validation still rejects it.
     */
    protected function prepareForValidation(): void
    {
        $value = $this->input('enable_beta');

        if (in_array($value, [true, 'true', 1, '1'], true)) {
            $this->merge(['enable_beta' => true]);
        } elseif (in_array($value, [false, 'false', 0, '0'], true)) {
            $this->merge(['enable_beta' => false]);
        }
    }

    public function messages(): array
    {
        return [
            'enable_beta.required' => __('Please choose an update channel.'),
            'enable_beta.boolean'  => __('The selected update channel is invalid.'),
        ];
    }
}
