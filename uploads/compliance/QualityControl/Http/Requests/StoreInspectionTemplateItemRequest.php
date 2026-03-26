<?php

namespace Modules\QualityControl\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreInspectionTemplateItemRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'item_name' => ['required', 'string', 'max:191'],
            'standard' => ['nullable', 'string', 'max:191'],
            'sort_order' => ['nullable', 'integer', 'min:0'],
            'is_required' => ['nullable', 'boolean'],
        ];
    }

    protected function prepareForValidation(): void
    {
        $this->merge([
            'sort_order' => (int) $this->input('sort_order', 0),
            'is_required' => (bool) $this->input('is_required', false),
        ]);
    }
}
