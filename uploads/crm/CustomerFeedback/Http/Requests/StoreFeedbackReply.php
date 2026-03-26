<?php

namespace Modules\CustomerFeedback\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreFeedbackReply extends FormRequest
{
    public function authorize()
    {
        return user()->permission('add_feedback_reply') != 'none';
    }

    public function rules()
    {
        return [
            'message' => 'required|string|max:5000',
            'message_html' => 'nullable|string|max:5000',
            'is_internal' => 'sometimes|boolean',
            'source_channel' => 'nullable|in:email,portal,api,auto',
            'files' => 'nullable|array',
            'files.*' => 'file|max:10240',
        ];
    }
}
