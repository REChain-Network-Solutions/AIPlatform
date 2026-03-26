<?php

namespace Modules\QualityControl\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreInspectionScheduleRequest extends FormRequest
{
    public function authorize(): bool { return true; }
    public function rules(): array { return []; }
}
