<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\HasMany;

class FeedbackTagList extends BaseModel
{
    use HasCompany;

    protected $table = 'feedback_tags_list';
    protected $fillable = ['name', 'description', 'color', 'company_id'];

    public function tags(): HasMany
    {
        return $this->hasMany(FeedbackTag::class, 'tag_id');
    }
}
