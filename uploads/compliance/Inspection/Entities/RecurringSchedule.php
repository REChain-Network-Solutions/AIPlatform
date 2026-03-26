<?php

namespace Modules\Inspection\Entities;

use Carbon\Carbon;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use App\Scopes\ActiveScope;
use App\Models\ModuleSetting;
use Modules\Units\Entities\Floor;
use Modules\Units\Entities\Tower;
use Illuminate\Notifications\Notifiable;
use Modules\Inspection\Entities\Schedule;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Modules\Inspection\Entities\InspectionTemplateStandar;

class RecurringSchedule extends BaseModel
{
    use Notifiable, HasCompany;

    protected $table = 'inspection_schedule_recurring';
    protected $dates = ['issue_date', 'next_schedule_date'];

    const MODULE_NAME = 'inspection';
    const ROTATION_COLOR = [
        'daily' => 'success',
        'weekly' => 'info',
        'bi-weekly' => 'warning',
        'monthly' => 'secondary',
        'quarterly' => 'light',
        'half-yearly' => 'dark',
        'annually' => 'success',
    ];

    public static function addModuleSetting($company)
    {
        // create admin, employee and client module settings
        $roles = ['admin', 'employee'];

        ModuleSetting::createRoleSettingEntry(self::MODULE_NAME, $roles, $company);

    }

    public function recurrings(): HasMany
    {
        return $this->hasMany(Schedule::class, 'schedule_recurring_id');
    }

    public function items(): HasMany
    {
        return $this->hasMany(RecurringScheduleItems::class, 'schedule_recurring_id');
    }

    public function floor(): BelongsTo
    {
        return $this->belongsTo(Floor::class, 'floor_id');
    }

    public function tower(): BelongsTo
    {
        return $this->belongsTo(Tower::class, 'tower_id');
    }

    public function getIssueOnAttribute()
    {
        if (is_null($this->issue_date)) {
            return '';
        }

        return Carbon::parse($this->issue_date)->format('d F, Y');

    }



}
