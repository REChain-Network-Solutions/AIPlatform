<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;

class FeedbackReplyTemplate extends BaseModel
{
    use HasCompany;

    protected $table = 'feedback_reply_templates';
    protected $fillable = [
        'name',
        'description',
        'message',
        'reply_type',
        'status',
        'company_id'
    ];

    protected $casts = [
        'status' => 'boolean',
    ];

    const TYPE_AUTO = 'auto';
    const TYPE_MANUAL = 'manual';
}
