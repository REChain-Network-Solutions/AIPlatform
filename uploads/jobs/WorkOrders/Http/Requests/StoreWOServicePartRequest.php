<?php

namespace Modules\WorkOrders\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreWOServicePartRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return {
            "work_order_id": [
                        "required",
                        "integer",
                        "min:1"
            ],
            "part_id": [
                        "required",
                        "integer",
                        "min:1"
            ],
            "qty": [
                        "required",
                        "numeric",
                        "min:0.01"
            ]
};
    }
}
