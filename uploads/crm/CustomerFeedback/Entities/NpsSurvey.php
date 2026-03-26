<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\HasMany;

/**
 * NpsSurvey - Net Promoter Score survey definitions
 */
class NpsSurvey extends BaseModel
{
    use HasCompany;

    protected $table = 'nps_surveys';
    protected $fillable = [
        'title',
        'description',
        'question',
        'meta',
        'status',
        'company_id'
    ];

    protected $casts = [
        'meta' => 'array',
        'status' => 'boolean',
    ];

    const DEFAULT_QUESTION = 'How likely are you to recommend us to a friend or colleague?';

    public function responses(): HasMany
    {
        return $this->hasMany(NpsResponse::class, 'nps_survey_id');
    }

    public function getPromotorCountAttribute()
    {
        return $this->responses()->where('score', '>=', 9)->count();
    }

    public function getPassiveCountAttribute()
    {
        return $this->responses()->whereBetween('score', [7, 8])->count();
    }

    public function getDetractorCountAttribute()
    {
        return $this->responses()->where('score', '<=', 6)->count();
    }
}
