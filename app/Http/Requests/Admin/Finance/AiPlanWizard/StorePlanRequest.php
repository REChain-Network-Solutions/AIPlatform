<?php

namespace App\Http\Requests\Admin\Finance\AiPlanWizard;

use App\Enums\Plan\FrequencyEnum;
use App\Enums\Plan\TypeEnum;
use App\Services\Finance\AiPlanWizardService;
use Illuminate\Contracts\Validation\ValidationRule;
use Illuminate\Foundation\Http\FormRequest;

class StorePlanRequest extends FormRequest
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
        return self::planRules();
    }

    /**
     * Shared rule set, reused per-step by ValidateStepRequest.
     *
     * @return array<string, ValidationRule|array|string>
     */
    public static function planRules(): array
    {
        $frequencies = implode(',', array_column(FrequencyEnum::cases(), 'value'));
        $types = implode(',', array_column(TypeEnum::cases(), 'value'));

        return [
            'type'                  => 'required|in:' . $types,
            'name'                  => 'required|string|max:190',
            'description'           => 'required|string|max:15000',
            'price'                 => 'required|numeric|min:0',
            'frequency'             => 'required_if:type,' . TypeEnum::SUBSCRIPTION->value . '|nullable|in:' . $frequencies,
            'trial_days'            => 'nullable|integer|min:0',
            'active'                => 'nullable|boolean',
            'is_featured'           => 'nullable|boolean',
            'features'              => 'required|string|max:15000',
            'credit_system_type'    => 'required|in:separated,shared',
            'shared_credits_amount' => 'required_if:credit_system_type,shared|nullable|numeric|min:0',
            'credit_tier'           => [
                'nullable',
                static function (string $attribute, mixed $value, callable $fail): void {
                    if (is_numeric($value)) {
                        if ((float) $value <= 0) {
                            $fail(__('The credit tier multiplier must be greater than zero.'));
                        }

                        return;
                    }

                    if (! array_key_exists((string) $value, AiPlanWizardService::CREDIT_TIERS)) {
                        $fail(__('The selected credit tier is invalid.'));
                    }
                },
            ],
            'credit_limits'         => 'nullable|array:' . implode(',', AiPlanWizardService::CREDIT_CATEGORIES),
            'credit_limits.*'       => 'nullable|numeric|min:0',
            'plan_ai_tools'         => 'nullable|array',
            'plan_ai_tools.*'       => 'boolean',
            'plan_features'         => 'nullable|array',
            'plan_features.*'       => 'boolean',
        ];
    }

    /**
     * Get custom messages for validator errors.
     *
     * @return array<string, string>
     */
    public function messages(): array
    {
        return [
            'name.required'                     => __('Please give the plan a name.'),
            'description.required'              => __('Please add a short plan description.'),
            'price.required'                    => __('Please set a price for the plan.'),
            'frequency.required_if'             => __('Please choose a billing frequency.'),
            'features.required'                 => __('Please list the plan features.'),
            'shared_credits_amount.required_if' => __('Please set the shared credits amount.'),
        ];
    }
}
