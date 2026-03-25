<?php

namespace Modules\WorkOrders\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class UpdateWORequestRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return {
            "customer_id": [
                        "sometimes",
                        "integer",
                        "min:1"
            ],
            "summary": [
                        "sometimes",
                        "string",
                        "max:500"
            ],
            "requested_at": [
                        "sometimes",
                        "nullable",
                        "date"
            ]
};
    }
}
