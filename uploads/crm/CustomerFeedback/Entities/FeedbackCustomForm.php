<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;

class FeedbackCustomForm extends BaseModel
{
    use HasCompany;

    protected $table = 'feedback_custom_forms';
    protected $fillable = [
        'name',
        'description',
        'fields',
        'status',
        'company_id'
    ];

    protected $casts = [
        'fields' => 'array',
        'status' => 'boolean',
    ];
}
