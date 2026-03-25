<?php

namespace Modules\WorkOrders\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreAppointmentRequest extends FormRequest
{
    public function authorize(): bool { return true; }
    public function rules(): array {
        return [
            'work_order_id' => ['required','integer'],
            'technician_id' => ['nullable','integer'],
            'starts_at' => ['nullable','date'],
            'ends_at' => ['nullable','date','after_or_equal:starts_at'],
            'location' => ['nullable','string','max:255'],
            'status' => ['nullable','string','max:50'],
        ];
    }
}
