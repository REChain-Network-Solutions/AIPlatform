<?php

namespace Modules\Inspection\Entities;

use Carbon\Carbon;
use App\Models\User;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use App\Scopes\ActiveScope;
use Illuminate\Support\Facades\DB;
use Illuminate\Notifications\Notifiable;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Modules\Inspection\Entities\ScheduleItems;
use Modules\Inspection\Entities\ScheduleReply;
use Illuminate\Database\Eloquent\Relations\BelongsTo;


class Schedule extends BaseModel
{

    use Notifiable;
    use HasCompany;

    protected $table = 'inspection_schedules';
    protected $dates = ['issue_date'];
    // protected $appends = ['created_on','inspected', 'closed'];



    public function items(): HasMany
    {
        return $this->hasMany(ScheduleItems::class, 'schedule_id');
    }

    public function worker(): BelongsTo
    {
        return $this->belongsTo(User::class, 'worker_id')->withoutGlobalScope(ActiveScope::class);
    }

    public function inspector(): BelongsTo
    {
        return $this->belongsTo(User::class, 'inspect_by')->withoutGlobalScope(ActiveScope::class);
    }

    public function agent(): BelongsTo
    {
        return $this->belongsTo(User::class, 'agent_id')->withoutGlobalScope(ActiveScope::class);
    }

    public function reply(): HasMany
    {
        return $this->hasMany(ScheduleReply::class, 'schedule_id');
    }
    // public function getIssueDateAttribute()
    // {
    //     if (is_null($this->issue_date)) {
    //         return '';
    //     }

    //     return Carbon::parse($this->issue_date)->format('d F, Y');
    // }

    // public function getCreatedOnAttribute()
    // {
    //     $setting = company();

    //     if (!is_null($this->created_at)) {
    //         return $this->created_at->timezone($setting->timezone)->format('d M Y H:i');
    //     }

    //     return '';
    // }

    // public function getInspectedAttribute()
    // {
    //     $setting = company();

    //     if (!is_null($this->inspected_at)) {
    //         return $this->inspected_at->timezone($setting->timezone)->format('d M Y H:i');
    //     }

    //     return '';
    // }

    // public function getClosedAttribute()
    // {
    //     $setting = company();

    //     if (!is_null($this->closed_at)) {
    //         return $this->closed_at->timezone($setting->timezone)->format('d M Y H:i');
    //     }

    //     return '';
    // }
}
