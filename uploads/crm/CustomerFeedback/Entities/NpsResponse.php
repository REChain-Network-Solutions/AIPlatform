<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * NpsResponse - Individual NPS survey responses
 */
class NpsResponse extends BaseModel
{
    use HasCompany;

    protected $table = 'nps_responses';
    protected $fillable = [
        'nps_survey_id',
        'feedback_ticket_id',
        'user_id',
        'score',
        'feedback',
        'metadata',
        'company_id'
    ];

    protected $casts = [
        'metadata' => 'array',
    ];

    public function survey(): BelongsTo
    {
        return $this->belongsTo(NpsSurvey::class, 'nps_survey_id');
    }

    public function ticket(): BelongsTo
    {
        return $this->belongsTo(FeedbackTicket::class, 'feedback_ticket_id');
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id');
    }

    public function getPromotorAttribute(): bool
    {
        return $this->score >= 9;
    }

    public function getPassiveAttribute(): bool
    {
        return $this->score >= 7 && $this->score <= 8;
    }

    public function getDetractorAttribute(): bool
    {
        return $this->score <= 6;
    }
}
