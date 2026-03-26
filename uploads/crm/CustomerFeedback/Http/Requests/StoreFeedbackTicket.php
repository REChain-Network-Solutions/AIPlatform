<?php

namespace Modules\CustomerFeedback\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreFeedbackTicket extends FormRequest
{
    public function authorize()
    {
        return user()->permission('add_feedback') != 'none';
    }

    public function rules()
    {
        return [
            'title' => 'required|string|max:255',
            'description' => 'required|string|max:5000',
            'user_id' => 'required|integer|exists:users,id',
            'agent_id' => 'nullable|integer|exists:users,id',
            'feedback_type' => 'nullable|in:complaint,feedback,survey_response',
            'status' => 'nullable|in:open,in_progress,resolved,closed,pending',
            'priority' => 'nullable|in:low,medium,high,critical',
            'channel_id' => 'nullable|integer|exists:feedback_channels,id',
            'group_id' => 'nullable|integer|exists:feedback_groups,id',
            'type_id' => 'nullable|integer|exists:feedback_types,id',
            'nps_score' => 'nullable|integer|min:1|max:10',
            'csat_score' => 'nullable|integer|min:1|max:5',
            'custom_meta' => 'nullable|array',
        ];
    }
}
