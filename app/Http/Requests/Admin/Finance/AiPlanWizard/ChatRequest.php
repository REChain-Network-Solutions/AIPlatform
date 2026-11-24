<?php

namespace App\Http\Requests\Admin\Finance\AiPlanWizard;

use Illuminate\Contracts\Validation\ValidationRule;
use Illuminate\Foundation\Http\FormRequest;

class ChatRequest extends FormRequest
{
    /**
     * Determine if the user is authorized to make this request.
     */
    public function authorize(): bool
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array<string, ValidationRule|array|string>
     */
    public function rules(): array
    {
        return [
            'messages'           => 'required|array|min:1|max:30',
            'messages.*.role'    => 'required|in:user,assistant',
            'messages.*.content' => 'required|string|max:8000',
            'draft'              => 'nullable|array',
        ];
    }
}
