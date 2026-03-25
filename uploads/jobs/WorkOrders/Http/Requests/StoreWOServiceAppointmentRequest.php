<?php

namespace Modules\WorkOrders\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreWOServiceAppointmentRequest extends FormRequest
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
            "starts_at": [
                        "required",
                        "date"
            ],
            "ends_at": [
                        "nullable",
                        "date",
                        "after_or_equal:starts_at"
            ],
            "assignee_id": [
                        "nullable",
                        "integer",
                        "min:1"
            ]
};
    }
}
