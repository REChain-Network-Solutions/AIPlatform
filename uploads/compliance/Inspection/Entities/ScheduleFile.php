<?php
namespace Modules\Inspection\Entities;

use App\Models\BaseModel;
use App\Traits\IconTrait;
use Modules\Inspection\Entities\ScheduleReply;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class ScheduleFile extends BaseModel
{
    use IconTrait;

    /**
     * Inspection module table (prefixed to avoid collisions with other modules / core).
     */
    protected $table = 'inspection_schedule_files';

    const FILE_PATH = 'schedule-files';

    protected $appends = ['file_url', 'icon'];

    public function getFileUrlAttribute()
    {
        return (!is_null($this->external_link)) ? $this->external_link : asset_url_local_s3('schedule-files/' . $this->schedule_reply_id . '/' . $this->hashname);
    }

    public function reply(): BelongsTo
    {
        return $this->belongsTo(ScheduleReply::class);
    }

}
