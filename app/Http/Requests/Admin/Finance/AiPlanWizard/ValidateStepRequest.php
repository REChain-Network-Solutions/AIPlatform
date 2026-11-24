<?php

namespace App\Http\Requests\Admin\Finance\AiPlanWizard;

use Illuminate\Contracts\Validation\ValidationRule;
use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Support\Arr;

class ValidateStepRequest extends FormRequest
{
    /**
     * @var array<string, array<int, string>>
     */
    public const STEP_FIELDS = [
        'basics'   => ['type', 'name', 'description', 'price', 'frequency', 'trial_days', 'active', 'is_featured'],
        'credits'  => ['credit_system_type', 'shared_credits_amount', 'credit_tier', 'credit_limits'],
        'features' => ['features', 'plan_ai_tools', 'plan_features'],
    ];

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
        $step = (string) $this->input('step');
        $fields = self::STEP_FIELDS[$step] ?? [];

        $rules = collect(StorePlanRequest::planRules())
            ->filter(fn ($rule, string $key): bool => in_array(Arr::first(explode('.', $key)), $fields, true))
            ->toArray();

        return array_merge($rules, [
            'step' => 'required|in:' . implode(',', array_keys(self::STEP_FIELDS)),
        ]);
    }

    /**
     * Get custom messages for validator errors.
     *
     * @return array<string, string>
     */
    public function messages(): array
    {
        return (new StorePlanRequest)->messages();
    }
}
