<?php

namespace Modules\WorkOrders\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreWOServiceTaskRequest extends FormRequest
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
            "name": [
                        "required",
                        "string",
                        "max:255"
            ],
            "hours": [
                        "nullable",
                        "numeric",
                        "min:0"
            ]
};
    }
}
