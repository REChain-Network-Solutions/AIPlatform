<?php

namespace Modules\WorkOrders\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class UpdateWOTypeRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return {
            "name": [
                        "sometimes",
                        "string",
                        "max:120"
            ]
};
    }
}
