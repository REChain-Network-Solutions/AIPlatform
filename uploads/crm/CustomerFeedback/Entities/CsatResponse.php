<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * CsatResponse - Individual CSAT survey responses
 */
class CsatResponse extends BaseModel
{
    use HasCompany;

    protected $table = 'csat_responses';
    protected $fillable = [
        'csat_survey_id',
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
        return $this->belongsTo(CsatSurvey::class, 'csat_survey_id');
    }

    public function ticket(): BelongsTo
    {
        return $this->belongsTo(FeedbackTicket::class, 'feedback_ticket_id');
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id');
    }
}
