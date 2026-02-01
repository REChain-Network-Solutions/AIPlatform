<?php

namespace App\Extensions\AISocialMedia\System\Models;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class ScheduledPost extends Model
{
    protected $table = 'scheduled_posts';

    protected $fillable = [
        'automation_platform_id',
        'user_id',
        'command_running',
        'last_run_date',
        'company_id',
        'platform',
        'products',
        'campaign_name',
        'campaign_target',
        'topics',
        'is_seo',
        'tone',
        'length',
        'is_email',
        'is_repeated',
        'repeat_period',
        'repeat_start_date',
        'repeat_time',
        'visual_format',
        'visual_ratio',
        'posted_at',
        'prompt',
        'content',
        'media',
        'auto_generate',
    ];

    protected $casts = [
        'auto_generate' => 'boolean',
    ];

    public function automationPlatform(): BelongsTo
    {
        return $this->belongsTo(AutomationPlatform::class, 'automation_platform_id');
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id');
    }
}
