<?php

namespace Modules\WorkOrders\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class UpdateWorkOrderRequest extends FormRequest
{
    public function authorize(): bool { return true; }
    public function rules(): array {
        return [
            'status' => ['nullable','string','max:50'],
            'priority' => ['nullable','string','max:50'],
            'scheduled_for' => ['nullable','date'],
            'due_by' => ['nullable','date','after_or_equal:scheduled_for'],
            'notes' => ['nullable','string'],
        ];
    }
}
