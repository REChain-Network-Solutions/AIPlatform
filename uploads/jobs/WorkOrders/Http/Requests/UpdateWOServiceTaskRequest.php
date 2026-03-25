<?php

namespace Modules\WorkOrders\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class UpdateWOServiceTaskRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return {
            "name": [
                        "sometimes",
                        "string",
                        "max:255"
            ],
            "hours": [
                        "sometimes",
                        "numeric",
                        "min:0"
            ]
};
    }
}
