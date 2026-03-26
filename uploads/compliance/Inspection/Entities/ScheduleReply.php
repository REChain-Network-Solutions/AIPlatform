<?php

namespace Modules\Inspection\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Scopes\ActiveScope;
use Illuminate\Database\Eloquent\SoftDeletes;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class ScheduleReply extends BaseModel
{
    use SoftDeletes;

    protected $table = 'inspection_schedule_replies';
    protected $dates = ['deleted_at'];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id')->withoutGlobalScope(ActiveScope::class);
    }

    public function files(): HasMany
    {
        return $this->hasMany(ScheduleFile::class, 'schedule_reply_id');
    }

    public function schedule(): BelongsTo
    {
        return $this->belongsTo(Schedule::class);
    }



}
