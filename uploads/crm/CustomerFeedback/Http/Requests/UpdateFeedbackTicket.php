<?php

namespace Modules\CustomerFeedback\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class UpdateFeedbackTicket extends FormRequest
{
    public function authorize()
    {
        return user()->permission('edit_feedback') != 'none';
    }

    public function rules()
    {
        return [
            'title' => 'sometimes|required|string|max:255',
            'description' => 'sometimes|required|string|max:5000',
            'status' => 'sometimes|required|in:open,in_progress,resolved,closed,pending',
            'priority' => 'sometimes|required|in:low,medium,high,critical',
            'agent_id' => 'nullable|integer|exists:users,id',
            'group_id' => 'nullable|integer|exists:feedback_groups,id',
            'type_id' => 'nullable|integer|exists:feedback_types,id',
        ];
    }
}
