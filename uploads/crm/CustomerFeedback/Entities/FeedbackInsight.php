<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * FeedbackInsight - AI-generated insights and analysis
 */
class FeedbackInsight extends BaseModel
{
    use HasCompany;

    protected $table = 'feedback_insights';
    protected $fillable = [
        'feedback_ticket_id',
        'insight_type',
        'title',
        'description',
        'confidence_score',
        'suggested_action',
        'tags',
        'metadata',
        'company_id'
    ];

    protected $casts = [
        'tags' => 'array',
        'metadata' => 'array',
        'confidence_score' => 'float',
    ];

    const TYPE_SENTIMENT = 'sentiment';
    const TYPE_CATEGORY = 'category';
    const TYPE_PRIORITY = 'priority';
    const TYPE_ACTION = 'action';
    const TYPE_TREND = 'trend';

    public function ticket(): BelongsTo
    {
        return $this->belongsTo(FeedbackTicket::class, 'feedback_ticket_id');
    }
}
