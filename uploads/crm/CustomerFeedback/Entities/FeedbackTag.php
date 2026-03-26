<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class FeedbackTag extends BaseModel
{
    use HasCompany;

    protected $table = 'feedback_tags';
    protected $fillable = ['feedback_id', 'tag_id', 'company_id'];

    public function ticket(): BelongsTo
    {
        return $this->belongsTo(FeedbackTicket::class, 'feedback_id');
    }

    public function tagList(): BelongsTo
    {
        return $this->belongsTo(FeedbackTagList::class, 'tag_id');
    }
}
