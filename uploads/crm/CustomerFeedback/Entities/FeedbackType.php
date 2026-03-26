<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\HasMany;

/**
 * FeedbackType - Classification types for feedback tickets
 */
class FeedbackType extends BaseModel
{
    use HasCompany;

    protected $table = 'feedback_types';
    protected $fillable = ['name', 'slug', 'description', 'type_category', 'status', 'company_id'];
    protected $casts = [
        'status' => 'boolean',
    ];

    // Category types
    const CATEGORY_COMPLAINT = 'complaint';
    const CATEGORY_FEEDBACK = 'feedback';
    const CATEGORY_SURVEY = 'survey';

    public function tickets(): HasMany
    {
        return $this->hasMany(FeedbackTicket::class, 'type_id');
    }

    public function scopeComplaints($query)
    {
        return $query->where('type_category', self::CATEGORY_COMPLAINT);
    }

    public function scopeFeedback($query)
    {
        return $query->where('type_category', self::CATEGORY_FEEDBACK);
    }

    public function scopeSurveys($query)
    {
        return $query->where('type_category', self::CATEGORY_SURVEY);
    }
}
