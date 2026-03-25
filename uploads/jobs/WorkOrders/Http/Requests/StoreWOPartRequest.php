<?php

namespace Modules\WorkOrders\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreWOPartRequest extends FormRequest
{
    public function authorize(): bool { return true; }
    public function rules(): array {
        return [
            'work_order_id' => ['required','integer'],
            'service_part_id' => ['nullable','integer'],
            'qty' => ['required','numeric','min:0'],
            'price' => ['required','numeric','min:0'],
        ];
    }
}
