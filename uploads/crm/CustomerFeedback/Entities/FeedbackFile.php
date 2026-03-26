<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class FeedbackFile extends BaseModel
{
    use HasCompany;

    protected $table = 'feedback_files';
    protected $fillable = [
        'feedback_id',
        'reply_id',
        'filename',
        'file_path',
        'file_size',
        'mime_type',
        'company_id'
    ];

    public function ticket(): BelongsTo
    {
        return $this->belongsTo(FeedbackTicket::class, 'feedback_id');
    }

    public function reply(): BelongsTo
    {
        return $this->belongsTo(FeedbackReply::class, 'reply_id');
    }
}
