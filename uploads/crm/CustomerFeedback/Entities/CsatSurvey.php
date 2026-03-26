<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\HasMany;

/**
 * CsatSurvey - Customer Satisfaction survey definitions
 */
class CsatSurvey extends BaseModel
{
    use HasCompany;

    protected $table = 'csat_surveys';
    protected $fillable = [
        'title',
        'description',
        'question',
        'scale_min',
        'scale_max',
        'meta',
        'status',
        'company_id'
    ];

    protected $casts = [
        'meta' => 'array',
        'status' => 'boolean',
    ];

    const DEFAULT_QUESTION = 'How satisfied are you with our service?';

    public function responses(): HasMany
    {
        return $this->hasMany(CsatResponse::class, 'csat_survey_id');
    }
}
