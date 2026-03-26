<?php

namespace Modules\CustomerFeedback\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreNpsSurvey extends FormRequest
{
    public function authorize()
    {
        return user()->permission('add_feedback') != 'none';
    }

    public function rules()
    {
        return [
            'title' => 'required|string|max:255',
            'description' => 'nullable|string|max:1000',
            'question' => 'nullable|string|max:500',
            'meta' => 'nullable|array',
        ];
    }
}
